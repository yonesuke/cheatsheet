# MLflow Cheatsheet

A guide to using MLflow's Python client "raw" (without framework integrations) for maximum control, including detailed post-experiment analysis.

> [!TIP]
> A runnable example script demonstrating these concepts (including a realistic training loop and analysis) is available at [cheatsheet_mlflow_snippets.py](file:///home/yoneda/cheatsheet/cheatsheet_mlflow_snippets.py). You can run it directly with `uv run cheatsheet_mlflow_snippets.py`.

## Setup

```python
import mlflow

# Set the tracking URI to a local directory or a server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or "file:///path/to/mlruns"
mlflow.set_experiment("my_experiment_name")
```

## Raw Tracking API

Avoiding `mlflow.autolog()` for explicit control over what gets logged.

### Basic Run Context

```python
import mlflow

# Start a run. Using 'with' matches the run lifecycle to the block scope.
with mlflow.start_run(run_name="manual_run_v1") as run:
    
    # 1. Log Parameters (Configuration)
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam"
    })
    
    # 2. Log Metrics (Looping)
    for epoch in range(10):
        # ... training code ...
        loss = 0.5 * (0.9 ** epoch) # Dummy loss
        accuracy = 0.5 + (0.05 * epoch)
        
        # step is optional but essential for time-series plots in UI
        mlflow.log_metrics({"train_loss": loss, "val_acc": accuracy}, step=epoch)
        
    # 3. Log Artifacts (Files)
    with open("output.txt", "w") as f:
        f.write("Some result data")
    
    # Log a local file to the run's artifact store
    mlflow.log_artifact("output.txt", artifact_path="results")
    
    # Log a model (if you want to use pyfunc or just save the pickle manually)
    # mlflow.sklearn.log_model(model, "model") # Integrated way (avoid if strictly "raw")
    # Raw way:
    # with open("model.pkl", "wb") as f: pickle.dump(model, f)
    # mlflow.log_artifact("model.pkl", "models")
```

### Nested Runs

Useful for hyperparameter search or cross-validation within a "parent" experiment run.

```python
with mlflow.start_run(run_name="hyperparam_search_parent") as parent_run:
    for lr in [0.01, 0.001]:
        with mlflow.start_run(run_name=f"trial_lr_{lr}", nested=True) as child_run:
            mlflow.log_param("lr", lr)
            # ... train and log metrics ...
```

## Post-Experiment Analysis & Retrieval

This is capable of finding the "best" run programmatically after experiments effectively acting as a model registry/leaderboard query.

### Searching Runs

`mlflow.search_runs` returns a Pandas `DataFrame`.

```python
import mlflow

# Get all runs from the experiment
experiment_id = mlflow.get_experiment_by_name("my_experiment_name").experiment_id

# Returns a Pandas DataFrame
df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.optimizer = 'adam' AND metrics.val_acc > 0.8", # SQL-like filtering
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    order_by=["metrics.val_acc DESC"] # Sort by metric to find best
)

print(df[["run_id", "params.learning_rate", "metrics.val_acc"]].head())
```

### Fetching the Best Run Automatically

Pattern to get the run ID and artifacts of the absolute best model.

```python
# 1. Search and Sort
best_run_df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.train_loss ASC"], # or "metrics.val_acc DESC"
    max_results=1
)

if not best_run_df.empty:
    best_run = best_run_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_loss = best_run["metrics.train_loss"]
    
    print(f"Best Run ID: {best_run_id}, Loss: {best_loss}")
    
    # 2. Retrieve Artifacts from that run
    # If you logged a file "config.json" or "model.pkl"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id, 
        artifact_path="results/output.txt"
    )
    print(f"Artifact downloaded to: {local_path}")
    
    # Validating the content
    with open(local_path, "r") as f:
        print(f.read())
else:
    print("No runs found.")
```

### Useful Filter Strings

| Requirement | Filter String |
| :--- | :--- |
| Specific Param | `params.model_type = 'cnn'` |
| Metric Threshold | `metrics.accuracy >= 0.95` |
| Tag check | `tags.release_candidate = 'true'` |
| Combination | `params.lr < 0.01 AND metrics.loss < 0.1` |
