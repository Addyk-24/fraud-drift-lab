from pathlib import Path

from src import config


def log_policy_run(name: str, params: dict, metrics: dict, artifacts=()):
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("retraining-policies")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            for a in artifacts:
                if Path(a).exists():
                    mlflow.log_artifact(str(a))
    except Exception as exc:
        print(f"[tracking] MLflow logging skipped: {type(exc).__name__}: {str(exc)[:120]}")
