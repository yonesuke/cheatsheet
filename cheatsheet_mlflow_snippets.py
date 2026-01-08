# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlflow",
#     "pandas",
# ]
# ///

import mlflow
import os
import shutil
import random
import math
import pathlib

def simulate_training_run(run_name, params):
    """
    Simulates a machine learning training run.
    """
    print(f"Starting run: {run_name} with params: {params}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log Hyperparameter Configuration
        mlflow.log_params(params)
        
        # Simulation parameters
        epochs = params.get("epochs", 10)
        base_lr = params.get("learning_rate", 0.01)
        decay = params.get("decay", 0.9)
        
        # 2. Training Loop (Simulating steps)
        for epoch in range(epochs):
            # Simulate "training" - calculating loss
            # Add some randomness to make it look real
            noise = random.uniform(-0.02, 0.02)
            train_loss = (1.0 - (0.8 * (1 - decay**epoch))) + noise
            val_loss = train_loss * 1.1 + random.uniform(0, 0.05)
            accuracy = 0.5 + (0.45 * (1 - decay**epoch))
            
            # Log metrics for this step
            # 'step' is crucial for visualizing the curve correctly in UI
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy
            }, step=epoch)
            
            # (Optional) Log checkpoints or artifacts occasionally
            if epoch % 5 == 0:
                with open(f"checkpoint_epoch_{epoch}.txt", "w") as f:
                    f.write(f"Model state at epoch {epoch}")
                mlflow.log_artifact(f"checkpoint_epoch_{epoch}.txt", artifact_path="checkpoints")

        # 3. Log Final Artifacts (e.g., best model, config)
        with open("final_model_config.json", "w") as f:
            f.write(str(params))
        mlflow.log_artifact("final_model_config.json")
        
    print(f"Finished run: {run_name}")

def analyze_experiments(experiment_name):
    """
    Demonstrates how to programmatically find the best run and retrieve data.
    """
    print("\n--- Post-Experiment Analysis ---")
    
    # Get Experiment Details
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found.")
        return

    # 1. SearchRuns: Get all runs into a Pandas DataFrame
    # Using SQL-like filter string
    df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.val_loss ASC"] # Sort by validation loss to find "best"
    )
    
    if df.empty:
        print("No finished runs found.")
        return

    print(f"Found {len(df)} runs.")
    
    # Display top 3 runs
    subset = df[["run_id", "params.learning_rate", "metrics.val_loss", "metrics.accuracy"]]
    print("\nTop 3 Runs (lowest validation loss):")
    print(subset.head(3))
    
    # 2. Get the "Best" Run
    best_run = df.iloc[0]
    best_run_id = best_run["run_id"]
    best_val_loss = best_run["metrics.val_loss"]
    
    print(f"\nBest Run ID: {best_run_id}")
    print(f"Best Val Loss: {best_val_loss}")
    
    # 3. Retrieve Artifacts from the Best Run
    print("\nDownloading artifacts from the best run...")
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=best_run_id,
            artifact_path="final_model_config.json"
        )
        print(f"Artifact downloaded to: {local_path}")
        with open(local_path, 'r') as f:
            print(f"Config content: {f.read()}")
    except Exception as e:
        print(f"Failed to download artifact: {e}")

def main():
    # Setup Experiment
    # Using a local directory relative to the script for persistent storage
    # This allows the user to inspect the 'mlruns' directory if they want
    cwd = pathlib.Path.cwd()
    tracking_uri = f"file://{cwd}/mlruns_snippets"
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "snippet_demo_experiment"
    mlflow.set_experiment(experiment_name)
    
    print(f"Tracking URI set to: {tracking_uri}")
    print(f"Experiment active: {experiment_name}")

    # Simulate a Grid Search or Random Search
    hyperparams_list = [
        {"learning_rate": 0.1, "epochs": 20, "decay": 0.8},
        {"learning_rate": 0.01, "epochs": 20, "decay": 0.95}, # This should be better
        {"learning_rate": 0.001, "epochs": 20, "decay": 0.99},
    ]

    for i, params in enumerate(hyperparams_list):
        simulate_training_run(f"trial_{i}", params)

    # Perform Analysis
    analyze_experiments(experiment_name)

if __name__ == "__main__":
    main()
