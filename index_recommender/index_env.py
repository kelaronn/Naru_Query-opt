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

# Import your existing modules
import datasets
import estimators as estimators_lib
# Assuming made.py, transformer.py, etc., are in the same folder
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
            num_samples=2000, # psample size
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
        card = self.estimator.Query(query_cols, query_ops, query_vals)
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
    estimated_rows = naru.estimate_cardinality(cols, ops, vals)
    total_rows = naru.table.cardinality
    
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
            index_scan_cost = (np.log2(total_rows) + estimated_rows) * 0.1
            
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
        # The agent chooses an integer 'i' (0 to n_cols-1).
        # This action toggles the index on the i-th column (Create <-> Drop).
        self.action_space = gym.spaces.Discrete(self.n_cols)
        
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
        self.rng = np.random.Generator(np.random.PCG64(42))
        
        # Episode length management
        self.current_step = 0
        self.max_steps_per_episode = 50 

    def step(self, action):
        """
        Executes one step in the environment.
        1. Applies the action (update index config).
        2. Generates a workload (query).
        3. Calculates cost (reward).
        """
        
        # 1. Update State (Toggle index status)
        # If 0 -> 1 (Create Index), If 1 -> 0 (Drop Index)
        self.state[action] = 1.0 - self.state[action]
        
        # 2. Generate Workload (Query)
        # We generate a random query to evaluate the current index configuration
        query = generate_workload_query(self.naru.table, self.rng, self.dataset_name)
        
        # 3. Calculate Cost (using Naru Estimator)
        # Identify which columns currently have an active index
        active_indexes = []
        for i, has_index in enumerate(self.state):
            if has_index == 1.0:
                # For simplicity, we assume single-column indexes here.
                # Complex composite indexes would require a larger action space.
                active_indexes.append([self.naru.table.columns[i].name])
        
        # Calculate the estimated cost of the query with the current indexes
        cost = calculate_cost(self.naru, query, active_indexes)
        
        # 4. Compute Reward
        # RL agents maximize reward, so we use negative cost.
        # We add a penalty for maintaining too many indexes (storage/write overhead).
        # Penalty factor (e.g., 100.0) needs tuning based on cost magnitude.
        maintenance_penalty = np.sum(self.state) * 50.0 
        
        reward = -cost - maintenance_penalty
        
        # 5. Check Termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Combine terminated and truncated for 'done' flag (compatibility with older Gym/SB3)
        done = terminated or truncated
        
        # Auxiliary information (useful for logging/debugging)
        info = {
            "query_cost": cost, 
            "num_indexes": np.sum(self.state)
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
        
        # Reset state to no indexes
        self.state = np.zeros(self.n_cols, dtype=np.float32)
        self.current_step = 0
        
        # Return observation and empty info dict
        return self.state, {}
    
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    import os

    # Model file
    # IMPORTANT: Ensure this path matches where your trained model is saved.
    # You may need to adjust the pattern based on your training setup.
    search_pattern = "models/tpch*.pt"
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
        
        # Optional: Check if the environment follows Gym API standards
        print("Checking Gym environment compliance...")
        # check_env(env) # Uncomment to run strict checks
        
        # --- 3. Train Agent ---
        print("Starting PPO Agent training...")
        
        # Initialize PPO (Proximal Policy Optimization) agent
        # MlpPolicy is suitable for vector inputs (our state)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        
        # Train the agent
        # total_timesteps determines how many interactions the agent has with the environment
        model.learn(total_timesteps=10000)
        
        print("Training finished.")
        
        # --- 4. Save and Test ---
        model.save("index_advisor_ppo")
        print("Agent saved to 'index_advisor_ppo.zip'")
        
        # Simple test run
        obs, _ = env.reset()
        print("\nTesting trained agent:")
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            print(f"Step {i}: Action (Toggle Col {action}) -> Reward: {reward:.2f}, Indexes: {info['num_indexes']}")
            if done:
                obs, _ = env.reset()

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Tip: Make sure the model architecture in NaruEstimator (Config) matches the trained model.")