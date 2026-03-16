import os
import sys
import glob

# Add parent directory to path so we can import eval_model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import gym
from gym import spaces
import eval_model
import datasets
import estimators as estimators_lib
from eval_model import MakeMade, GenerateQuery


# Configuration class replacing command-line arguments (argparse)
class Config:
    def __init__(self):
        self.dataset = 'tpch'  # Options: 'dmv', 'tpch'
        self.fc_hiddens = 128
        self.layers = 4
        self.residual = True    #Must match training config, set to true for resmade
        self.direct_io = False
        self.input_encoding = 'binary'
        self.output_encoding = 'one_hot'
        self.column_masking = False
        self.inv_order = False
        # Add other parameters from train_model.py as needed

class NaruEstimator:
    def __init__(self, model_path, device='cpu'):
        self.cfg = Config()
        eval_model.args = self.cfg  # For compatibility with MakeMade
        self.device = device
        
        # 1. Load data (essential to know column metadata)
        # For example: this calls LoadDmv from datasets.py
        if self.cfg.dataset == 'dmv-tiny':
            self.table = datasets.LoadDmv('dmv-tiny.csv')
        elif self.cfg.dataset == 'tpch':
            self.table = datasets.LoadTpch('tpch_lineitem_10k.csv')
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")
        
        # 2. Build model structure (initialized with random weights)
        # We reuse the MakeMade function logic from eval_model.py
        # Note: 'seed' and 'fixed_ordering' should match training configuration
        self.model = MakeMade(
            scale=self.cfg.fc_hiddens,
            cols_to_train=self.table.columns,
            seed=0, 
            fixed_ordering=None
        ).to(device)
        
        # 3. Load trained weights
        print(f"Loading Naru model from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # 4. Create ProgressiveSampling estimator
        # This wrapper handles the actual sampling logic for cardinality estimation
        self.estimator = estimators_lib.ProgressiveSampling(
            self.model,
            self.table,
            50, # psample size, in eval_model.py it's set to 2000, but it tremendously slows down the gym training process
            device=device,
            shortcircuit=self.cfg.column_masking
        )

    def estimate_cardinality(self, query_cols, query_ops, query_vals):
        """
        Estimates the number of rows matching the query.
        
        Args:
            query_cols: list of Column objects (e.g., [Column(Make), Column(Year)])
            query_ops: list of operators (e.g., ['=', '>='])
            query_vals: list of values (e.g., ['Toyota', 2010])
            
        Returns:
            Estimated cardinality (int/float)
        """
        with torch.no_grad():
            card_tensor = self.estimator.Query(query_cols, query_ops, query_vals)
            
            # Ensure we get a scalar value, not a tensor
            if isinstance(card_tensor, torch.Tensor):
                card = card_tensor.detach().cpu().item()
            else:
                card = float(card_tensor)
        
        return card

def calculate_cost(naru, query, current_indexes):
    """
    Calculates the execution cost of a query given a set of indexes.
    
    Args:
        naru: Instance of NaruEstimator
        query: Tuple of (cols, ops, vals)
        current_indexes: List of available indexes, e.g., [['Make'], ['Model', 'Year']]
    """
    cols, ops, vals = query
    
    # 1. Ask Naru for selectivity estimation
    estimated_rows_raw = naru.estimate_cardinality(cols, ops, vals)

    if hasattr(estimated_rows_raw, 'item'):
        estimated_rows = float(estimated_rows_raw.item())
    else:
        estimated_rows = float(estimated_rows_raw)

    total_rows = naru.table.cardinality  # Assuming 100% selectivity means all rows

    
    # Selectivity = (estimated rows) / (total rows)
    # This is the key metric provided by the neural network
    
    # 2. Simplified Cost Model
    # Base case: Full Table Scan (cost proportional to total rows)
    cost = total_rows * 1.0 
    
    # Check if any index can optimize this query
    best_index_cost = cost
    
    for index in current_indexes:
        # Check if the index is applicable (simple prefix matching)
        # E.g., if query filters on 'Make', an index on ('Make', 'Model') is applicable.
        if is_index_applicable(index, cols):
            # Heuristic: Index Scan Cost ≈ log(N) + (Selected Rows)
            # We multiply by a factor (e.g., 0.1) to represent that index access is faster than scanning
            RANDOM_IO_PENALTY = 2.0
            index_scan_cost = np.log2(total_rows) + (estimated_rows*RANDOM_IO_PENALTY)
            
            if index_scan_cost < best_index_cost:   
                best_index_cost = index_scan_cost
                
    return best_index_cost

def is_index_applicable(index_cols, query_cols):
    """
    Helper to check if an index covers the query columns.
    Simple implementation: check if the first column of the index is in the query.
    """
    query_col_names = [c.name for c in query_cols]
    # If the leading column of the index is present in the query predicates, it's usable.
    if index_cols[0] in query_col_names:
        return True
    return False

def generate_workload_query(table, rng, dataset_name='tpch'):
    """
    Generates a random query independent of global arguments.
    
    Args:
        table: The table object containing data and columns.
        rng: Numpy random state generator.
        dataset_name: Name of the dataset (examples: 'tpch','dmv') to handle specific logic.
    """
    # 1. Determine number of filters (1 to 3)
    num_filters = rng.randint(1, 4)
    
    # 2. Select random columns for the query
    # We avoid picking the same column twice
    chosen_cols = rng.choice(table.columns, num_filters, replace=False)
    
    query_cols = []
    query_ops = []
    query_vals = []
    
    # Access the underlying pandas DataFrame for sampling values
    df = table.data
    
    for col in chosen_cols:
        # Randomly select an operator
        op = rng.choice(['=', '>=', '<=', '>'])
        
        # Sample a real value from the actual data to ensure the query is meaningful
        # We sample a random row and take the value of the current column
        random_row_idx = rng.randint(0, len(df))
        val = df.iloc[random_row_idx][col.name]
        
        # Handle specific type conversions if necessary (e.g., for TPCH dates)
        if dataset_name == 'tpch' and 'DATE' in col.name.upper():
             if not isinstance(val, np.datetime64):
                 val = np.datetime64(val)

        query_cols.append(col)
        query_ops.append(op)
        query_vals.append(val)
        
    return query_cols, query_ops, query_vals

class IndexEnv(gym.Env):
    """
    Custom Gym Environment for Database Index Selection.
    The agent learns to select the optimal set of indexes to minimize query cost.
    """
    
    def __init__(self, naru_estimator):
        super(IndexEnv, self).__init__()
        
        self.naru = naru_estimator
        self.dataset_name = self.naru.cfg.dataset
        self.n_cols = len(self.naru.table.columns)
        
        # --- Define Action and Observation Spaces ---
        
        # Action Space: Discrete choice.
        # The agent chooses an integer 'i' (0 to 2 * n_cols).
        # Actions 0 to n_cols-1: Create Index on column 'i'.
        # Actions n_cols to 2*n_cols-1: Drop Index on column 'i - n_cols'.
        # Action 2 * n_cols: STOP (Finish optimization early).
        self.action_space = gym.spaces.Discrete(self.n_cols * 2 + 1)
        
        # Observation Space: Binary vector.
        # Represents the current configuration: [0, 1, 0, ...]
        # 1 means an index exists on that column, 0 means it does not.
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.n_cols,), 
            dtype=np.float32
        )
        
        # Initial state: No indexes created
        self.state = np.zeros(self.n_cols, dtype=np.float32)
        
        # Random generator for consistent workload generation
        self.rng = np.random.RandomState(42)  # Seeded for determinism
        
        # Generate a deterministic workload of M queries
        self.workload_size = 10
        self.workload = []
        for _ in range(self.workload_size):
            self.workload.append(generate_workload_query(self.naru.table, self.rng, self.dataset_name))
            
        # Pre-calculate baseline cost (Average Cost with NO Indexes)
        self.baseline_cost = 0.0
        for q in self.workload:
            c = calculate_cost(self.naru, q, [])
            if isinstance(c, torch.Tensor):
                self.baseline_cost += float(c.item())
            else:
                self.baseline_cost += float(c)
        self.baseline_cost /= self.workload_size
        print(f"Environment initialized with deterministic workload of size {self.workload_size}. Baseline Average Cost: {self.baseline_cost:.2f}")

        # Episode length management
        self.current_step = 0
        self.max_steps_per_episode = 50

    def step(self, action):
        """
        Executes one step in the environment.
        1. Applies the action (Explicit Create/Drop Index, or STOP).
        2. Evaluates the fixed workload.
        3. Calculates reward based on relative improvement against baseline, with strict invalid action penalties.
        """
        
        invalid_step = False
        terminated = False
        
        # 1. Update State (Create/Drop Index/Stop)
        if action == self.n_cols * 2:
            # Action: STOP
            terminated = True
        elif action < self.n_cols:
            # Action: Create Index
            if self.state[action] == 1.0:
                # Already exists - invalid action
                invalid_step = True
            else:
                self.state[action] = 1.0
        else:
            # Action: Drop Index
            target_col = action - self.n_cols
            if self.state[target_col] == 0.0:
                # Does not exist - invalid action
                invalid_step = True
            else:
                self.state[target_col] = 0.0
        
        # 3. Calculate Cost (using Naru Estimator) over the entire Workload
        # Identify which columns currently have an active index
        active_indexes = []
        for i, has_index in enumerate(self.state):
            if has_index == 1.0:
                active_indexes.append([self.naru.table.columns[i].name])
        
        # Calculate the average cost of the deterministic workload
        total_cost = 0.0
        for current_query in self.workload:
            try:
                raw_cost = calculate_cost(self.naru, current_query, active_indexes)
            except Exception as e:
                print(f"Warning: Cost calculation failed with error: {e}")
                # Use a moderately high cost on failure so we don't destroy gradients
                raw_cost = self.baseline_cost * 2.0 if self.baseline_cost > 0 else 10000.0

            if isinstance(raw_cost, torch.Tensor):
                cost = float(raw_cost.item())
            else:
                cost = float(raw_cost)
            total_cost += cost

        avg_workload_cost = total_cost / self.workload_size

        # 4. Compute Reward (Reward Shaping)
        # We calculate relative cost against the No-Index baseline cost.
        # Cost ratio will be around 1.0 when no indexes are used, and < 1.0 when good indexes are used.
        if self.baseline_cost > 0:
            cost_ratio = avg_workload_cost / self.baseline_cost
        else:
            cost_ratio = 1.0

        # We add a penalty for maintaining too many indexes.
        # Tuned penalty to be relative to the [-1.0, 0.0] scale.
        PENALTY_PER_INDEX = 0.05  # Each index costs 5% of the baseline cost in penalty
        penalty = np.sum(self.state) * PENALTY_PER_INDEX
        
        # RL agents maximize reward, so we use negative cost ratio.
        # Reward will typically start around -1.0 and move towards ~0.0
        reward = float(-cost_ratio - penalty)
        
        # Add a massive penalty for taking an invalid action (idempotent loops)
        if invalid_step:
            reward = -2.0  # High relative punishment
            terminated = True # Force early exit so it can't farm bad states
            
        # Add a very small reward bonus for stopping early to lock in good indexes
        if terminated and not invalid_step:
            reward += 0.5 # Give a stronger incentive to use the STOP button

        # 5. Check Termination
        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Combine terminated and truncated for 'done' flag
        done = terminated or truncated
        
        # Auxiliary information (useful for logging/debugging)
        info = {
            "query_cost": cost, 
            "num_indexes": float(np.sum(self.state))
        }
        
        # Return format for Gym API (v26+ uses terminated, truncated)
        # If using older SB3 versions, use: return self.state, reward, done, info
        return self.state, reward, done, False, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        Called at the beginning of each training episode.
        """
        super().reset(seed=seed)
        # The workload generation is now completely separated and deterministic (done in __init__).
        # State resetting is just putting indexes back to zero.
        # Reset state to no indexes
        self.state = np.zeros(self.n_cols, dtype=np.float32)
        self.current_step = 0

        # Return observation and empty info dict
        return self.state, {}
    
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import ProgressBarCallback
    import os

    # Model file
    # IMPORTANT: Ensure this path matches where your trained model is saved.
    # You may need to adjust the pattern based on your training setup.
    # Search in parent directory for models
    parent_dir_main = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir_main)  # Change working directory so datasets can be found
    search_pattern = "models/tpch*.pt"  # Now use relative path since we're in parent dir
    found_files = glob.glob(search_pattern)

    # Check if any model files were found
    if not found_files:
        print(f"ERROR: No model files found matching pattern {search_pattern}")
        print("Please run 'train_model.py' first to generate a checkpoint.")
        # Create a dummy file for syntax checking purposes
        with open("dummy_index_env.py", "w") as f:
            f.write("# Dummy file created due to missing model files.\n")
        sys.exit(1)

    MODEL_PATH = found_files[0]  # Use the first found model file
    print(f"Using model file: {MODEL_PATH}")
    
    # --- 2. Initialize Environment ---
    print("Initializing Naru Estimator...")
    # Note: Ensure NaruEstimator is initialized with the correct device (cuda/cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try-except block to handle cases where model file is missing during development
    try:
        naru = NaruEstimator(model_path=MODEL_PATH, device=device)
        env = IndexEnv(naru)
        
        # Re-enable gradients for training (ProgressiveSampling disables them)
        torch.set_grad_enabled(True)
        
        # Optional: Check if the environment follows Gym API standards
        print("Checking Gym environment compliance...")
        # check_env(env) # Uncomment to run strict checks
        
        # --- 3. Train Agent ---
        print("Starting PPO Agent training...")
        
        # Initialize PPO (Proximal Policy Optimization) agent
        # MlpPolicy is suitable for vector inputs (our state)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log="./tensorboard_logs/")
        
        # Train the agent
        # total_timesteps determines how many interactions the agent has with the environment
        print("Training for 10,000 timesteps...")
        model.learn(total_timesteps=10000, callback=ProgressBarCallback())
        
        print("Training finished.")
        
        # --- 4. Save and Test ---
        model.save("index_advisor_ppo")
        print("Agent saved to 'index_advisor_ppo.zip'")
        
        # Simple test run
        obs, _ = env.reset()
        print("\nTesting untrained agent (10 random baseline steps) just to verify mechanics:")
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            if action == env.n_cols * 2:
                action_type = "STOP"
                target_col = "Optimization Complete"
            else:
                action_type = "Create" if action < env.n_cols else "Drop"
                target_col = env.naru.table.columns[action % env.n_cols].name
            
            print(f"Step {i+1}: Action ({action_type} '{target_col}') -> Reward (Cost Ratio & Penalty): {reward:.4f}, Active Indexes: {info['num_indexes']}")
            if done:
                obs, _ = env.reset()
    # Catch exceptions related to model loading
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Tip: Make sure the model architecture in NaruEstimator (Config) matches the trained model.")