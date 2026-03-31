import os
import sys
import glob
import psycopg2

# Add parent directory to path so we can import eval_model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.logger import configure
import time

import eval_model
import datasets
import estimators as estimators_lib
from eval_model import MakeMade, GenerateQuery


# Configuration class replacing command-line arguments (argparse)
class Config:
    def __init__(self):
        self.dataset = 'tpch_pg'  # Use PG dataset
        self.db_url = 'postgresql://postgres:postgres@localhost:5432/naru_db'
        self.fc_hiddens = 128
        self.layers = 4
        self.residual = True    #Must match training config, set to true for resmade
        self.direct_io = False
        self.input_encoding = 'binary'
        self.output_encoding = 'one_hot'
        self.column_masking = False
        self.inv_order = False

class NaruEstimator:
    def __init__(self, model_path, device='cpu'):
        self.cfg = Config()
        eval_model.args = self.cfg  # For compatibility with MakeMade
        self.device = device
        
        # 1. Load data
        if self.cfg.dataset == 'dmv-tiny':
            self.table = datasets.LoadDmv('dmv-tiny.csv')
        elif self.cfg.dataset == 'tpch_pg':
            self.table = datasets.LoadTpchFromPostgres(self.cfg.db_url)
        elif self.cfg.dataset == 'tpch':
            self.table = datasets.LoadTpch('tpch_lineitem_10k.csv')
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")
        
        # 2. Build model structure
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
        self.estimator = estimators_lib.ProgressiveSampling(
            self.model,
            self.table,
            50,
            device=device,
            shortcircuit=self.cfg.column_masking
        )

    def estimate_cardinality(self, query_cols, query_ops, query_vals):
        with torch.no_grad():
            card_tensor = self.estimator.Query(query_cols, query_ops, query_vals)
            if isinstance(card_tensor, torch.Tensor):
                card = card_tensor.detach().cpu().item()
            else:
                card = float(card_tensor)
        return card

def calculate_cost(naru, query, pg_conn=None, analyze=False):
    """
    Calculates the execution cost of a query using PostgreSQL's EXPLAIN.
    If analyze is True, executes the query and returns Actual Execution Time (ms).
    Otherwise returns Planner Total Cost.
    """
    cols, ops, vals = query
    
    if pg_conn is None:
        return 10000.0 # Fallback
        
    where_clauses = []
    for col, op, val in zip(cols, ops, vals):
        col_name = col.name
        # format value appropriately
        if isinstance(val, (str, np.datetime64)):
            # convert np.datetime64 to standard string if needed
            val_str = str(val).split('T')[0]
            formatted_val = f"'{val_str}'"
        else:
            formatted_val = str(val)
        where_clauses.append(f'"{col_name}" {op} {formatted_val}')
        
    where_str = " AND ".join(where_clauses)
    
    if analyze:
        sql_query = f"EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM lineitem WHERE {where_str};"
    else:
        sql_query = f"EXPLAIN (FORMAT JSON) SELECT * FROM lineitem WHERE {where_str};"
    
    try:
        with pg_conn.cursor() as cur:
            cur.execute(sql_query)
            result = cur.fetchone()
            if analyze:
                # Actual time taken for the query in ms
                execution_time = result[0][0].get('Execution Time', 10000.0)
                return float(execution_time)
            else:
                plan = result[0][0]['Plan']
                total_cost = plan['Total Cost']
                return float(total_cost)
    except Exception as e:
        print(f"Error executing EXPLAIN: {e}")
        # rollback on error so subsequent queries don't fail
        try:
            pg_conn.rollback()
        except:
            pass
        return 10000.0

def generate_workload_query(table, rng, dataset_name='tpch'):
    num_filters = rng.randint(1, 4)
    chosen_cols = rng.choice(table.columns, num_filters, replace=False)
    
    query_cols = []
    query_ops = []
    query_vals = []
    
    df = table.data
    
    for col in chosen_cols:
        op = rng.choice(['=', '>=', '<=', '>'])
        random_row_idx = rng.randint(0, len(df))
        val = df.iloc[random_row_idx][col.name]
        
        if 'tpch' in dataset_name and 'DATE' in col.name.upper():
             if not isinstance(val, np.datetime64):
                 val = np.datetime64(val)

        query_cols.append(col)
        query_ops.append(op)
        query_vals.append(val)
        
    return query_cols, query_ops, query_vals

def estimate_cost_with_naru(naru_estimator, query, index_state):
    """
    Calculates the estimated cost of a query based on NARU cardinality and index state.
    If an index is present, it returns the index scan cost, otherwise it returns the full table scan cost.
    """
    query_cols, ops, vals = query
    card = naru_estimator.estimate_cardinality(query_cols, ops, vals)
    
    total_rows = len(naru_estimator.table.data)
    
    # Check if any of the query columns have an active index
    active_index_found = False
    for col in query_cols:
        col_idx = naru_estimator.table.columns.index(col)
        if index_state[col_idx] == 1.0:
            active_index_found = True
            break
            
    if active_index_found:
        # Index scan cost: random I/O + fetching subset of rows
        # E.g. base cost + (cardinality * cost_per_row)
        cost = 50.0 + (card * 4.0)
    else:
        # Full table scan cost: sequential I/O of all rows
        cost = float(total_rows * 1.5)
        
    return float(cost)

class IndexEnv(gym.Env):
    def __init__(self, naru_estimator):
        super(IndexEnv, self).__init__()
        self.naru = naru_estimator
        self.dataset_name = self.naru.cfg.dataset
        self.n_cols = len(self.naru.table.columns)
        
        self.action_space = gym.spaces.Discrete(self.n_cols * 2 + 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_cols,), dtype=np.float32)
        self.state = np.zeros(self.n_cols, dtype=np.float32)
        
        self.rng = np.random.RandomState(42)  # Seeded for determinism
        
        # Connect to Postgres
        self.db_url = self.naru.cfg.db_url
        try:
            self.pg_conn = psycopg2.connect(self.db_url)
            self.pg_conn.autocommit = True
            print("Successfully connected to PostgreSQL for cost estimation.")
            
            # Drop all potential RL indexes before starting
            with self.pg_conn.cursor() as cur:
                for col in self.naru.table.columns:
                    cur.execute(f"DROP INDEX IF EXISTS idx_rl_{col.name.lower()};")
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            self.pg_conn = None

        # Generate a deterministic workload of M queries
        self.workload_size = 10
        self.workload = []
        for _ in range(self.workload_size):
            self.workload.append(generate_workload_query(self.naru.table, self.rng, self.dataset_name))
            
        # State: 1 (tanulás NARU alapon), 2 (finomhangolás Postgres alapon)
        self.training_phase = 1 
        
        # Pre-calculate baseline cost (Average Cost with NO Indexes) for both phases
        self.baseline_cost_pg_time = 0.0
        self.baseline_cost_naru = 0.0
        empty_index_state = np.zeros(self.n_cols, dtype=np.float32)
        
        for q in self.workload:
            c_pg = calculate_cost(self.naru, q, pg_conn=self.pg_conn, analyze=True)
            self.baseline_cost_pg_time += float(c_pg)
            c_naru = estimate_cost_with_naru(self.naru, q, empty_index_state)
            self.baseline_cost_naru += float(c_naru)
            
        self.baseline_cost_pg_time /= self.workload_size
        self.baseline_cost_naru /= self.workload_size
        
        print(f"Environment initialized with deterministic workload of size {self.workload_size}.")
        print(f"Baseline Time (PG Analyze): {self.baseline_cost_pg_time:.2f} ms, Baseline Cost (NARU): {self.baseline_cost_naru:.2f}")

        self.current_step = 0
        self.max_steps_per_episode = 50

    def step(self, action):
        invalid_step = False
        terminated = False
        
        # 1. Update State (Create/Drop Index/Stop)
        if action == self.n_cols * 2:
            terminated = True
        elif action < self.n_cols:
            if self.state[action] == 1.0:
                invalid_step = True
            else:
                self.state[action] = 1.0
                col_name = self.naru.table.columns[action].name
                index_name = f"idx_rl_{col_name.lower()}"
                if self.training_phase == 2 and self.pg_conn:
                    try:
                        with self.pg_conn.cursor() as cur:
                            cur.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON lineitem ("{col_name}");')
                    except Exception as e:
                        print(f"Failed to create index {index_name}: {e}")
        else:
            target_col = action - self.n_cols
            if self.state[target_col] == 0.0:
                invalid_step = True
            else:
                self.state[target_col] = 0.0
                col_name = self.naru.table.columns[target_col].name
                index_name = f"idx_rl_{col_name.lower()}"
                if self.training_phase == 2 and self.pg_conn:
                    try:
                        with self.pg_conn.cursor() as cur:
                            cur.execute(f"DROP INDEX IF EXISTS {index_name};")
                    except Exception as e:
                        print(f"Failed to drop index {index_name}: {e}")
        
        # 3. Calculate Cost using PostgreSQL EXPLAIN over the Workload
        total_value = 0.0
        
        for current_query in self.workload:
            try:
                if self.training_phase == 1:
                    raw_val = estimate_cost_with_naru(self.naru, current_query, self.state)
                else:
                    raw_val = calculate_cost(self.naru, current_query, pg_conn=self.pg_conn, analyze=True)
            except Exception as e:
                # Büntetés hiba esetén
                raw_val = (self.baseline_cost_naru * 2.0) if self.training_phase == 1 else (self.baseline_cost_pg_time * 2.0)
            total_value += float(raw_val)

        avg_workload_val = total_value / self.workload_size

        # Skálázási trükk: a Phase 2-es (ms) időket leképezzük a Phase 1-es (NARU) nagyságrendre!
        # Így a DQN Q-értékei nem zavarodnak meg a fázisváltáskor.
        if self.training_phase == 2:
            time_ratio = avg_workload_val / self.baseline_cost_pg_time
            scaled_cost = time_ratio * self.baseline_cost_naru
            current_baseline_scaled = self.baseline_cost_naru
        else:
            scaled_cost = avg_workload_val
            current_baseline_scaled = self.baseline_cost_naru

        # 4. Compute Reward (Reward Shaping) - Improvement over baseline
        improvement = current_baseline_scaled - scaled_cost

        # Penalty for maintaining too many indexes. 5% of baseline cost per index.
        PENALTY_PER_INDEX = current_baseline_scaled * 0.05
        penalty = np.sum(self.state) * PENALTY_PER_INDEX
        
        reward = float(improvement - penalty)
        
        if invalid_step:
            # Huge penalty relative to baseline
            reward = float(-current_baseline_scaled * 2.0)
            terminated = True
            
        if terminated and not invalid_step:
            # Small bonus for cleanly stopping
            reward += float(current_baseline_scaled * 0.1)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode
        done = terminated or truncated
        
        info = {
            "query_cost": float(avg_workload_val), 
            "num_indexes": float(np.sum(self.state))
        }
        
        return self.state, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state to no indexes physically and locally
        if self.training_phase == 2 and self.pg_conn:
            with self.pg_conn.cursor() as cur:
                for col in self.naru.table.columns:
                    cur.execute(f"DROP INDEX IF EXISTS idx_rl_{col.name.lower()};")

        self.state = np.zeros(self.n_cols, dtype=np.float32)
        self.current_step = 0

        return self.state, {}
    
    def close(self):
        if hasattr(self, 'pg_conn') and self.pg_conn:
            self.pg_conn.close()
        super().close()

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import ProgressBarCallback
    
    parent_dir_main = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir_main)
    search_pattern = "models/tpch*.pt"
    found_files = glob.glob(search_pattern)

    if not found_files:
        print(f"ERROR: No model files found matching pattern {search_pattern}")
        # Need to put an absolute path dummy check here or run Naru training first.
        # Fallback for checking compilation ONLY
        # sys.exit(1)
        MODEL_PATH = "dummy.pt"
    else:
        MODEL_PATH = found_files[0]
        
    print(f"Using model file: {MODEL_PATH}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        naru = NaruEstimator(model_path=MODEL_PATH, device=device)
        env = IndexEnv(naru)
        
        torch.set_grad_enabled(True)
        
        print("\n--- PHASE 1: Traning based on NARU estimated costs ---")
        env.training_phase = 1
        
        # RL Model cseréje PPO-ról DQN-re
        # A DQN egy off-policy algoritmus, replay bufferrel és Q-hálózattal. Diszkrét téren kiváló.
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0005, 
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3
        )
        
        # Leghatékonyabb megoldás: Egyedi logger, ami egyetlen timestamp mappába gyűjti mindkét fázist
        # A 'log' a txt/log szöveges mentést, a 'csv' a táblázatos mentést készíti a folyamatsávról
        run_name = f"DQN_Run_{int(time.time())}"
        log_dir = f"./tensorboard_logs/{run_name}"
        custom_logger = configure(log_dir, ["stdout", "tensorboard", "log", "csv"])
        model.set_logger(custom_logger)
        
        print(f"Tensorboard naplózás helye: {log_dir}")
        print("Training for 4,000 timesteps directly from Naru estimator...")
        model.learn(total_timesteps=4000, callback=ProgressBarCallback())
        
        print("\n--- PHASE 2: Finomhangolás PostgreSQL EXPLAIN értékekkel ---")
        env.training_phase = 2
        # Töröljük a fizikai indexeket a biztonság kedvéért fázisváltáskor
        env.reset()
        
        print("Training for 1,000 timesteps using Postgres Database...")
        # Reset_num_timesteps=False, hogy lássuk a görbén a folytatást (tb log)
        model.learn(total_timesteps=1000, callback=ProgressBarCallback(), reset_num_timesteps=False)
        
        print("Training finished.")
        model.save("index_advisor_dqn")
        print("Agent saved to 'index_advisor_dqn.zip'")
        
        obs, _ = env.reset()
        print("\nTesting trained agent (10 random steps):")
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            if action == env.n_cols * 2:
                action_type = "STOP"
                target_col = "Optimization Complete"
            else:
                action_type = "Create" if action < env.n_cols else "Drop"
                target_col = env.naru.table.columns[action % env.n_cols].name
            
            print(f"Step {i+1}: Action ({action_type} '{target_col}') -> Reward: {reward:.2f}, Cost: {info['query_cost']:.2f}, Active Indexes: {info['num_indexes']}")
            if done:
                obs, _ = env.reset()
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nAn error occurred during execution: {e}")