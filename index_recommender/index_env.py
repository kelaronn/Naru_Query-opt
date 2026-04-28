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
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.logger import configure
import time

import eval_model
import datasets
import estimators as estimators_lib
from eval_model import MakeMade, GenerateQuery



class EpsilonBumpCallback(BaseCallback):
    """
    Directly overrides model.exploration_rate every step during Phase 2.
    This is necessary because SB3's internal scheduler computes progress_remaining < 0
    when num_timesteps > _total_timesteps (Phase 2 only runs 1000 steps), immediately
    resetting epsilon to 0.05 regardless of any manual setting.
    """
    def __init__(self, phase2_start_step: int, phase2_steps: int,
                 bump_start: float = 0.3, bump_end: float = 0.05):
        super().__init__()
        self.phase2_start_step = phase2_start_step
        self.phase2_steps = phase2_steps
        self.bump_start = bump_start
        self.bump_end = bump_end

    def _on_step(self) -> bool:
        elapsed = self.model.num_timesteps - self.phase2_start_step
        progress = min(max(elapsed / self.phase2_steps, 0.0), 1.0)
        self.model.exploration_rate = self.bump_end + (self.bump_start - self.bump_end) * (1.0 - progress)
        return True


# Configuration class replacing command-line arguments (argparse)
class Config:
    def __init__(self):
        self.dataset = 'tpch_pg'  # Use PG dataset
        self.db_url = 'postgresql://postgres:postgres@localhost:5432/naru_db'
        self.fc_hiddens = 128
        self.layers = 4
        self.residual = False    #Must match training config, set to true for resmade
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

def calculate_cost(naru, query, pg_conn=None, analyze=False, table_name='lineitem'):
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
        sql_query = f"EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM {table_name} WHERE {where_str};"
    else:
        sql_query = f"EXPLAIN (FORMAT JSON) SELECT * FROM {table_name} WHERE {where_str};"
    
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
    
class MaskedDQN(DQN):
        """
        A custom DQN wrapper that enforces action masking during both:
        1. Epsilon-greedy exploration (sample from valid actions only)
        2. Q-value inference (argmax over valid actions only)
        """
        def predict(self, observation, state=None, episode_start=None, deterministic=False):
            # 1. Get the action mask from the unwrapped environment
            env = self.get_env().envs[0]
            mask = env.unwrapped.action_masks()
            
            # 2. Get Q-values from the neural network
            observation_tensor = torch.as_tensor(observation).reshape(1, -1).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(observation_tensor).cpu().numpy()[0]
            
            # 3. Apply the mask: set Q-values of illegal actions to -infinity
            q_values[~mask] = -np.inf
            
            # 4. Epsilon-greedy exploration (if not deterministic)
            exploration = self.exploration_rate
            
            # EXPLORATION BUMP FOR PHASE 2:
            # If we are in phase 2 and just started it, bump exploration to 30% manually
            # to break out of the phase 1 local minimum.
            if env.unwrapped.training_phase == 2 and self.num_timesteps < 4300:
                exploration = max(exploration, 0.3)
                
            if not deterministic and np.random.rand() < exploration:
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
            else:
                # Argmax over the masked Q-values (legal actions only)
                action = np.argmax(q_values)
                
            return np.array([action]), None

class IndexEnv(gym.Env):
    def __init__(self, naru_estimator):
        super(IndexEnv, self).__init__()
        self.naru = naru_estimator
        self.dataset_name = self.naru.cfg.dataset
        self.n_cols = len(self.naru.table.columns)
        # Table name used in all SQL statements — change here if you switch datasets.
        self.table_name = 'lineitem'
        
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
                    cur.execute(f"DROP INDEX IF EXISTS idx_rl_{self.table_name}_{col.name.lower()};");
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            self.pg_conn = None

        # Generate dynamic workload
        self.workload_size = 10
        self.workload = []
        
        # Pre-declare baseline variables
        self.baseline_cost_pg_time = 0.0
        self.baseline_cost_naru = 0.0
        self.penalty_per_index = 0.0
        self.detailed_log_file = None
        
        # Compute the penalty scaling factor (alpha) once (database parameter)
        self.alpha = self._compute_alpha()

        # Generate the initial workload and calculate baselines
        self._generate_dynamic_workload()
        
        # State: 1 (learning based on NARU), 2 (finetuning based on Postgres)
        self.training_phase = 1 
        self.current_step = 0
        self.max_steps_per_episode = 50

    def set_log_path(self, log_path):
        """Set the file path for logging."""
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.detailed_log_file = open(log_path, 'w')
        # Header for CSV style
        self.detailed_log_file.write("Phase,Step,State,Action,Reward\n")

    def step(self, action):
        terminated = False

        # 1. Update State (Create/Drop Index/Stop)
        # NOTE: illegal actions (create where index exists, drop where none exists)
        # are prevented at the algorithm level via action_masks(). This branch
        # acts as a silent safety-net only — it should never be reached during
        # normal MaskablePPO training, but DQN's epsilon-greedy hits it frequently.
        state_changed = False
        if action == self.n_cols * 2:
            terminated = True
        elif action < self.n_cols:
            if self.state[action] == 0.0:   # safety: only act if truly valid
                self.state[action] = 1.0
                state_changed = True
                col_name = self.naru.table.columns[action].name
                index_name = f"idx_rl_{self.table_name}_{col_name.lower()}"
                if self.training_phase == 2 and self.pg_conn:
                    try:
                        with self.pg_conn.cursor() as cur:
                            cur.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {self.table_name} ("{col_name}");')
                    except Exception as e:
                        print(f"Failed to create index {index_name}: {e}")
        else:
            target_col = action - self.n_cols
            if self.state[target_col] == 1.0:  # safety: only act if truly valid
                self.state[target_col] = 0.0
                state_changed = True
                col_name = self.naru.table.columns[target_col].name
                index_name = f"idx_rl_{self.table_name}_{col_name.lower()}"
                if self.training_phase == 2 and self.pg_conn:
                    try:
                        with self.pg_conn.cursor() as cur:
                            cur.execute(f"DROP INDEX IF EXISTS {index_name};")
                    except Exception as e:
                        print(f"Failed to drop index {index_name}: {e}")

        # 3. Calculate Cost (Skip if state hasn't changed to save time and avoid Jitter)
        if not state_changed and not terminated and hasattr(self, 'previous_avg_workload_val'):
            avg_workload_val = self.previous_avg_workload_val
            reward = 0.0
        else:
            total_value = 0.0
    
            for current_query in self.workload:
                try:
                    if self.training_phase == 1:
                        raw_val = estimate_cost_with_naru(self.naru, current_query, self.state)
                    else:
                        raw_val = calculate_cost(self.naru, current_query, pg_conn=self.pg_conn, analyze=True, table_name=self.table_name)
                except Exception as e:
                    raw_val = (self.baseline_cost_naru * 2.0) if self.training_phase == 1 else (self.baseline_cost_pg_time * 2.0)
                total_value += float(raw_val)
    
            avg_workload_val = total_value / self.workload_size
            self.previous_avg_workload_val = avg_workload_val
    
            # Scaling trick: map Phase 2 (ms) times to Phase 1 (NARU) scale!
            if self.training_phase == 2:
                time_ratio = avg_workload_val / self.baseline_cost_pg_time
                scaled_cost = time_ratio * self.baseline_cost_naru
                current_baseline_scaled = self.baseline_cost_naru
            else:
                scaled_cost = avg_workload_val
                current_baseline_scaled = self.baseline_cost_naru
    
            # 4. Compute Reward (Reward Shaping) - Improvement over baseline
            
        # W (Workload Frequency): Represents how many times the workload runs over the lifetime of the index.
        # In real-world index tuning (like MS AutoAdmin), storage costs (penalty) are paid once, 
        # while query execution savings are accumulated over thousands of runs.
        # We assume the average query is run 100 times relative to the cost of storing the index.
            W_LIFETIME = 100.0
            
            improvement = current_baseline_scaled - scaled_cost
            total_lifetime_savings = improvement * W_LIFETIME
    
        # Penalty: theoretically calibrated I/O overhead per B-Tree index
            penalty = np.sum(self.state) * self.penalty_per_index
    
        # The Absolute Evaluate Reward
            current_total_reward = float(total_lifetime_savings - penalty)
            
        # FIX FOR EXPLOIT (Reward Farming): Use Differential Reward!
        # The agent only gets points for IMPROVING the state, and loses points for making it worse.
            if not hasattr(self, 'previous_total_reward'):
                self.previous_total_reward = 0.0
                
            reward = current_total_reward - self.previous_total_reward
            self.previous_total_reward = current_total_reward

        if terminated:
            pass  # No STOP bonus: differential reward already guides the agent.
                  # A bonus here causes premature stopping for reward-farming.

        # REWARD NORMALISATION: relative to maximum possible single-step improvement.
        # ×2.0 → 10% improvement ≈ +0.2, 100% improvement ≈ +2.0
        # Preserves gradient between "good" (+1.0) and "excellent" (+2.0) — no information loss from clipping.
        W_LIFETIME = 100.0  # must match the value used above
        reward = reward / (self.baseline_cost_naru * W_LIFETIME + 1e-9) * 2.0
        # Safety clip ±3: catches any degenerate edge cases without clipping normal rewards.
        reward = float(np.clip(reward, -10.0, 10.0))

        self.current_step += 1
        # Gymnasium standard: terminated = agent chose STOP, truncated = step budget exceeded.
        # Keeping them separate preserves the Q-value bootstrap at episode boundary
        # (a truncated state is NOT a terminal state, so gamma * V(s') is kept).
        truncated = (self.current_step >= self.max_steps_per_episode) and not terminated

        info = {
            "query_cost": float(avg_workload_val),
            "num_indexes": float(np.sum(self.state))
        }

        # Writing to the open log file without terminal spam
        if hasattr(self, 'detailed_log_file') and self.detailed_log_file and not self.detailed_log_file.closed:
            state_str = "[" + " ".join(f"{x:.1f}" for x in self.state) + "]"
            log_msg = f"{self.training_phase},{self.current_step},{state_str},{action},{reward:.4f}\n"
            self.detailed_log_file.write(log_msg)
            self.detailed_log_file.flush() # Ensures that it is written to the disk immediately

        return self.state, reward, terminated, truncated, info

    def _generate_dynamic_workload(self):
        """
        Generates a new random workload and calculates the corresponding 
        baseline costs (cost without any indexes) and penalty logic.
        Called in __init__ and at the start of every episode in reset().
        """
        self.workload = []
        for _ in range(self.workload_size):
            self.workload.append(generate_workload_query(self.naru.table, self.rng, self.dataset_name))
            
        self.baseline_cost_pg_time = 0.0
        self.baseline_cost_naru = 0.0
        empty_index_state = np.zeros(self.n_cols, dtype=np.float32)
        
        for q in self.workload:
            c_pg = calculate_cost(self.naru, q, pg_conn=self.pg_conn, analyze=True, table_name=self.table_name)
            self.baseline_cost_pg_time += float(c_pg)
            c_naru = estimate_cost_with_naru(self.naru, q, empty_index_state)
            self.baseline_cost_naru += float(c_naru)
            
        self.baseline_cost_pg_time /= self.workload_size
        self.baseline_cost_naru /= self.workload_size
        
        # Recalculate penalty for this specific workload's baseline
        self.penalty_per_index = float(self.baseline_cost_naru * self.alpha)

    def _compute_alpha(self) -> float:
        """
        Computes alpha, the ratio of B-Tree sequential maintenance overhead,
        based on PostgreSQL configuration and table size. 
        Only needs to be computed once at initialization.
        """
        AVG_BTREE_ENTRY_BYTES = 10   # 6-byte ItemId pointer + ~4 byte key (int/date/float)
        BTREE_FILL_FACTOR     = 0.9  # PostgreSQL B-Tree fillfactor

        rpc         = 1.1    # SSD default (random_page_cost)
        spc         = 1.0    # seq_page_cost default
        block_size  = 8192
        n_rows      = len(self.naru.table.data)
        # Fallback: estimate table_pages from in-memory data assuming ~100 B/row average
        AVG_ROW_BYTES_FALLBACK = 100
        table_pages = max(1, int(np.ceil(n_rows * AVG_ROW_BYTES_FALLBACK / block_size)))

        if self.pg_conn:
            try:
                with self.pg_conn.cursor() as cur:
                    cur.execute("SHOW random_page_cost;")
                    rpc = float(cur.fetchone()[0])

                    cur.execute("SHOW seq_page_cost;")
                    spc = float(cur.fetchone()[0])

                    cur.execute("SHOW block_size;")
                    block_size = int(cur.fetchone()[0])

                    # reltuples and relpages are O(1) — updated by ANALYZE, no full scan.
                    cur.execute(
                        "SELECT reltuples::BIGINT, relpages::BIGINT FROM pg_class "
                        "WHERE relname = %s AND relkind = 'r';",
                        (self.table_name,)
                    )
                    row = cur.fetchone()
                    if row:
                        if int(row[0]) > 0:
                            n_rows = int(row[0])
                        if int(row[1]) > 0:
                            table_pages = int(row[1])
            except Exception as e:
                print(f"[Penalty Calibration] Warning: PG parameter query failed ({e}). "
                      "Using SSD defaults and in-memory estimates.")

        # Effective usable bytes per index page (fillfactor accounts for B-Tree splits)
        usable_bytes_per_index_page = block_size * BTREE_FILL_FACTOR
        index_pages = max(1, int(np.ceil(n_rows * AVG_BTREE_ENTRY_BYTES
                                         / usable_bytes_per_index_page)))

        # α: fraction of the sequential full-table-scan cost that one index's
        #    random I/O overhead represents.
        #    Denominator: table_pages × spc  (spc is per PAGE, not per row!)
        alpha = (index_pages * rpc) / max(table_pages * spc, 1.0)

        print(
            f"[Penalty Calibration] table={self.table_name}, n_rows={n_rows:,}, "
            f"table_pages={table_pages}, index_pages={index_pages} "
            f"(fill={BTREE_FILL_FACTOR}), rpc={rpc}, spc={spc}, "
            f"block_size={block_size}B, alpha={alpha:.5f} "
            f"(= {alpha*100:.3f}% of NARU baseline per index)"
        )
        return float(alpha)

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of currently valid actions.

        Used by MaskablePPO (sb3-contrib) to eliminate illegal moves before
        action sampling and Q/policy network inference — the agent never even
        sees an illegal action in its probability distribution.

        Action encoding
        ---------------
        [0,       n_cols)   CREATE index on column i  — valid iff state[i] == 0
        [n_cols, 2*n_cols)  DROP   index on column i  — valid iff state[i] == 1
        [2*n_cols]          STOP                       — always valid
        """
        mask = np.zeros(self.n_cols * 2 + 1, dtype=bool)
        for i in range(self.n_cols):
            if self.state[i] == 0.0:
                mask[i] = True               # CREATE valid: slot is empty
            else:
                mask[self.n_cols + i] = True  # DROP  valid: slot is occupied
        mask[self.n_cols * 2] = True          # STOP is always a legal choice
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state to no indexes physically and locally
        if self.training_phase == 2 and self.pg_conn:
            with self.pg_conn.cursor() as cur:
                for col in self.naru.table.columns:
                    cur.execute(f"DROP INDEX IF EXISTS idx_rl_{self.table_name}_{col.name.lower()};")

        self.state = np.zeros(self.n_cols, dtype=np.float32)
        self.current_step = 0
        self.previous_total_reward = 0.0
        if hasattr(self, 'previous_avg_workload_val'):
            del self.previous_avg_workload_val
        
        # Dynamically sample a fresh 10 queries and compute their un-indexed baselines
        self._generate_dynamic_workload()

        return self.state, {}
    
    def close(self):
        if hasattr(self, 'detailed_log_file') and self.detailed_log_file and not self.detailed_log_file.closed:
            self.detailed_log_file.close()
        if hasattr(self, 'pg_conn') and self.pg_conn:
            self.pg_conn.close()
        super().close()

if __name__ == "__main__":
    from stable_baselines3.common.callbacks import ProgressBarCallback
    
    # --- Custom DQN Action Masking ---
    
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

        # Creating folders for logs (common folder for phases)
        run_name = f"DQN_Masked_Run_{int(time.time())}"
        log_dir = f"./tensorboard_logs/{run_name}"
        os.makedirs(log_dir, exist_ok=True)
        # Setting the detailed per-step logging for the env!
        env.set_log_path(os.path.join(log_dir, "log_perstep.txt"))

        print("\n--- PHASE 1: Training based on NARU estimated costs ---")
        env.training_phase = 1

        # Using our customized DQN that respects the action_masks() function
        model = MaskedDQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0005,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            max_grad_norm=10,          # Gradient clipping: prevents exploding gradients
        )
        
        # Most efficient solution: Custom logger that collects both phases in a single timestamp folder
        # 'log' saves the text/log, 'csv' saves the tabular data of the process
        custom_logger = configure(log_dir, ["stdout", "tensorboard", "log", "csv"])
        model.set_logger(custom_logger)
        
        print(f"Tensorboard logging location: {log_dir}")
        print("Training for 4,000 timesteps directly from Naru estimator...")
        model.learn(total_timesteps=4000, callback=ProgressBarCallback())
        
        print("\n--- PHASE 2: Finetuning using PostgreSQL EXPLAIN values ---")
        env.training_phase = 2
        # Reset replay buffer: Phase 1 Naru Q-values mislead the agent in Phase 2 —
        # as epsilon decreases, the agent exploits wrong Phase 1 intuitions ("create all indexes")
        # which Postgres actually penalises. A clean buffer forces Postgres-only learning.
        if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
            model.replay_buffer.reset()
        # Delete physical indexes for safety during phase transition
        env.reset()
        
        # CRITICAL FIX: SB3 DQN's internal _on_step() runs AFTER our callback every step and
        # calls: self.exploration_rate = self.exploration_schedule(progress_remaining)
        # This ALWAYS overrides our callback's value back to 0.05!
        # Solution: Replace exploration_schedule with an identity lambda that just returns the
        # current exploration_rate unchanged → _on_step becomes a no-op for epsilon,
        # and EpsilonBumpCallback has FULL control over exploration_rate.
        model.exploration_schedule = lambda _: model.exploration_rate
        
        print("Training for 2,000 timesteps using Postgres Database...")
        phase1_steps = 4000
        phase2_steps = 2000
        epsilon_callback = EpsilonBumpCallback(
            phase2_start_step=phase1_steps,
            phase2_steps=phase2_steps,
            bump_start=0.3,
            bump_end=0.05
        )
        # reset_num_timesteps=False → TensorBoard remains linear AND learning_starts stays satisfied
        model.learn(
            total_timesteps=phase2_steps,
            callback=[ProgressBarCallback(), epsilon_callback],
            reset_num_timesteps=False
        )

        print("Training finished.")
        # Restore a picklable exploration_schedule before saving.
        # Our lambda captures 'model' (which holds env with psycopg2 connection → not picklable).
        # Replacing it with a simple constant function resolves the TypeError on model.save().
        final_eps = model.exploration_rate
        model.exploration_schedule = lambda _: final_eps
        model.save("index_advisor_dqn_masked")
        print("Agent saved to 'index_advisor_dqn_masked.zip'")
        
        obs, _ = env.reset()
        print("\nTesting trained agent (10 steps, deterministic):")
        for i in range(10):
            # Custom MaskedDQN handles the mask internally inside predict()
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)  # predict() returns numpy array; columns[] needs a scalar
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
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