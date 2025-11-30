import glob
import torch
import numpy as np
import os
import sys

# Beállítjuk a munkakönyvtárat és az import útvonalat
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)  # Munkakönyvtár = szülőkönyvtár (ahol a datasets/ van)

from stable_baselines3 import PPO

# Import the environment and estimator from your existing file
# Ensure index_env.py is in the same directory!
from index_env import IndexEnv, NaruEstimator

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
    env = IndexEnv(naru)

    env.rng = np.random.RandomState(None)  # Ensure reproducibility if needed
    
    # Re-enable gradients for inference (ProgressiveSampling disables them)
    torch.set_grad_enabled(True)
    
    # --- 2. Load the Trained Agent ---
    print("Loading Trained PPO Agent...")
    agent_path = os.path.join(os.path.dirname(__file__), "index_advisor_ppo")
    
    if not os.path.exists(f"{agent_path}.zip"):
        print(f"Error: Trained agent file '{agent_path}.zip' not found.")
        print("Please run 'index_env.py' first to train the model.")
        return

    # Load the model from the zip file
    model = PPO.load(agent_path)

    # --- 3. Testing (Inference Loop) ---
    print("\n=== STARTING INFERENCE TEST ===")
    print(f"Running inference on test queries...\n")
    
    # Reset environment to get a FRESH set of queries (workload)
    # The agent has never seen these specific queries before
    obs, _ = env.reset()
    
    # Get column names for better logging
    col_names = [c.name for c in naru.table.columns]
    
    print(f"{'STEP':<6} | {'ACTION (Toggle Index)':<25} | {'REWARD (Cost)':<15} | {'ACTIVE INDEXES'}")
    print("-" * 100)
    
    # Run for a fixed number of steps (e.g., 10) or until 'done'
    for i in range(10):
        # Predict the next action based on the observation
        # deterministic=True ensures the agent uses its best known strategy (no random exploration)
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute the action in the environment
        obs, reward, done, _, info = env.step(action)
        
        # --- Logging Results ---
        # Decode action to column name
        target_col = col_names[action]
        
        # Decode observation (state) to list of active indexes
        # obs is a binary vector: [0, 1, 0, ...] -> 1 means index is active
        active_indexes = [col_names[idx] for idx, val in enumerate(obs) if val == 1.0]
        
        # Format the active indexes list as a string
        active_str = ", ".join(active_indexes) if active_indexes else "None"
        
        print(f"{i+1:<6} | {target_col:<25} | {reward:<15.4f} | {active_str}")
        
        if done:
            print("Episode finished early.")
            break

    print("-" * 100)
    print("\n=== TEST FINISHED ===")

if __name__ == "__main__":
    main()