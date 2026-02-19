from __future__ import annotations

from contextlib import contextmanager
import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str):
    with mlflow.start_run(run_name=run_name) as run:
        yield run
