from pathlib import Path

from src import config


def log_policy_run(name: str, params: dict, metrics: dict, artifacts=()):
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment("retraining-policies")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for a in artifacts:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
