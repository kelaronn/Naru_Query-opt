import glob
import torch
import numpy as np
import os
import sys

# Beállítjuk a munkakönyvtárat és az import útvonalat
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)  # Munkakönyvtár = szülőkönyvtár (ahol a datasets/ van)

from stable_baselines3 import DQN

# Import the environment and estimator from your existing file
# Ensure index_env.py is in the same directory!
from index_env import IndexEnv, NaruEstimator, MaskedDQN, calculate_cost, estimate_cost_with_naru

def format_query(q):
    cols, ops, vals = q
    where_clauses = []
    for col, op, val in zip(cols, ops, vals):
        if isinstance(val, (str, np.datetime64)):
            val_str = str(val).split('T')[0]
            formatted_val = f"'{val_str}'"
        else:
            formatted_val = str(val)
        where_clauses.append(f'"{col.name}" {op} {formatted_val}')
    return " AND ".join(where_clauses)

def main():
    # --- 1. Setup Environment (Same as in training) ---
    print("Initializing Environment...")
    
    # Locate the Naru model checkpoint automatically
    model_pattern = os.path.join(parent_dir, "models", "tpch*.pt")
    model_files = glob.glob(model_pattern)
    if not model_files:
        print("Error: Naru model (.pt) not found in 'models/' directory!")
        return
    naru_model_path = model_files[0]
    print(f"Using Naru model: {naru_model_path}")
    
    # Determine device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the Naru Estimator
    # This loads the neural network that predicts cardinality
    naru = NaruEstimator(model_path=naru_model_path, device=device)
    
    # Create the Gym Environment using the estimator
    # Dinamikus workload: a tesztelés is véletlenszerű lekérdezéseken fut
    env = IndexEnv(naru, dynamic_workload=False)

    env.rng = np.random.RandomState(None)  # Ensure reproducibility if needed
    
    # Re-enable gradients for inference (ProgressiveSampling disables them)
    torch.set_grad_enabled(True)
    
    # --- 2. Load the Trained Agent ---
    print("Loading Trained DQN Agent...")
    agent_path = os.path.join(parent_dir, "index_advisor_dqn_masked_best")
    
    if not os.path.exists(f"{agent_path}.zip"):
        print(f"Error: Trained agent file '{agent_path}.zip' not found.")
        print("Please run 'index_env.py' first to train the model.")
        return

    # Load the model from the zip file
    model = MaskedDQN.load(agent_path, env=env)

    # --- 3. Testing (Inference Loop) ---
    # Beállítjuk a Phase 2-t (Postgres valódi végrehajtás). Fontos, hogy ez reset előtt legyen!
    env.training_phase = 2 
    
    # Reset environment with a fixed seed to ensure consistent comparison
    env.rng = np.random.RandomState(42)
    env._generate_dynamic_workload()
    obs, _ = env.reset(seed=42)
    
    log_lines = []
    
    print("\n=== STARTING INFERENCE TEST ===")
    log_lines.append("=== STARTING INFERENCE TEST ===")
    print("Running inference on test queries...")
    log_lines.append("Running inference on test queries...")
    
    # Kiszámoljuk az indexek előtti időket ténylegesen Postgres-ben!
    print("\n--- Evaluating queries BEFORE index creation (PostgreSQL EXPLAIN ANALYZE) ---")
    log_lines.append("\n--- Evaluating queries BEFORE index creation (PostgreSQL EXPLAIN ANALYZE) ---")
    before_costs = []
    for idx, q in enumerate(env.workload):
        q_str = format_query(q)
        # Itt ha a training_phase == 2 és van pg_conn, akkor a calculate_cost ténylegesen futtatja a Postgres EXPLAIN ANALYZE-t
        if env.training_phase == 2 and env.pg_conn:
            time_val, blks_val = calculate_cost(env.naru, q, pg_conn=env.pg_conn, analyze=True, table_name=env.table_name)
            cost = time_val
        else:
            cost = estimate_cost_with_naru(env.naru, q, np.zeros(env.n_cols))
        before_costs.append(cost)
        
        msg = f"Query {idx+1}: {cost:.4f} ms | {q_str}"
        print(msg)
        log_lines.append(msg)
    
    # Get column names for better logging
    col_names = [c.name for c in naru.table.columns]
    
    header = f"\n{'STEP':<6} | {'ACTION (Toggle Index)':<25} | {'REWARD (Cost)':<15} | {'ACTIVE INDEXES'}"
    separator = "-" * 100
    print(header)
    print(separator)
    log_lines.append(header)
    log_lines.append(separator)
    
    # Run for a fixed number of steps (e.g., 50) or until 'done'
    for i in range(10):
        # Predict the next action based on the observation
        # deterministic=True ensures the agent uses its best known strategy (no random exploration)
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)  # cast because predict() returns a numpy array
        
        # Execute the action in the environment (This actually CREATES indexes in Postgres since phase=2)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Logging Results ---
        # Decode action to column name and action type
        if action == len(col_names) * 2:
            action_type = "STOP"
            target_col = "Optimization Finished"
        elif action < len(col_names):
            action_type = "Create Index"
            target_col = col_names[action]
        else:
            action_type = "Drop Index"
            target_col = col_names[action - len(col_names)]
            
        action_str = f"{action_type} ({target_col})"
        
        # Decode observation (state) to list of active indexes
        # obs is a binary vector: [0, 1, 0, ...] -> 1 means index is active
        # De az új obs kétszer akkora! Az első fele a state.
        active_state = obs[:len(col_names)]
        active_indexes = [col_names[idx] for idx, val in enumerate(active_state) if val == 1.0]
        
        # Format the active indexes list as a string
        active_str = ", ".join(active_indexes) if active_indexes else "None"
        
        msg = f"{i+1:<6} | {action_str:<25} | {reward:<15.4f} | {active_str}"
        print(msg)
        log_lines.append(msg)
        
        if done:
            msg_done = "Episode finished early."
            print(msg_done)
            log_lines.append(msg_done)
            break

    print(separator)
    log_lines.append(separator)
    
    # Kiszámoljuk az indexek utáni időket Postgresben!
    print("\n--- Evaluating queries AFTER index creation (PostgreSQL EXPLAIN ANALYZE) ---")
    log_lines.append("\n--- Evaluating queries AFTER index creation (PostgreSQL EXPLAIN ANALYZE) ---")
    after_costs = []
    for idx, q in enumerate(env.workload):
        q_str = format_query(q)
        if env.training_phase == 2 and env.pg_conn:
            time_val, blks_val = calculate_cost(env.naru, q, pg_conn=env.pg_conn, analyze=True, table_name=env.table_name)
            cost = time_val
        else:
            cost = estimate_cost_with_naru(env.naru, q, env.state)
        after_costs.append(cost)
        
        improvement = before_costs[idx] - cost
        msg = f"Query {idx+1}: {cost:.4f} ms (Improvement: {improvement:.4f} ms) | {q_str}"
        print(msg)
        log_lines.append(msg)

    avg_before = sum(before_costs) / len(before_costs)
    avg_after = sum(after_costs) / len(after_costs)
    total_improvement = avg_before - avg_after
    
    summary_msg = f"\nAverage Cost BEFORE: {avg_before:.4f} ms | AFTER: {avg_after:.4f} ms | AVG IMPROVEMENT: {total_improvement:.4f} ms"
    print(summary_msg)
    log_lines.append(summary_msg)

    print("\n=== TEST FINISHED ===")
    log_lines.append("\n=== TEST FINISHED ===")
    
    # Log to a text file
    log_file_path = os.path.join(parent_dir, "test_agent_results.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nResults successfully logged to: {log_file_path}")

if __name__ == "__main__":
    main()