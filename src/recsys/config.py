from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

default_raw = PROJECT_ROOT / "movielens_data"
alt_raw = PROJECT_ROOT / "movie_lens_data"
raw_path = default_raw if default_raw.exists() else alt_raw
DATA_RAW = Path(os.getenv("RECSYS_DATA_DIR", raw_path))
DATA_PROCESSED = Path(os.getenv("RECSYS_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
ARTIFACTS_DIR = Path(os.getenv("RECSYS_ARTIFACTS_DIR", PROJECT_ROOT / "models"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))


@dataclass(frozen=True)
class SplitConfig:
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    min_ratings: int = 5
    min_item_ratings: int = 5


@dataclass(frozen=True)
class SamplingConfig:
    max_users: int | None = 5000
    max_items: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class LightGCNConfig:
    embedding_dim: int = 64
    n_layers: int = 2
    lr: float = 1e-3
    reg: float = 1e-4
    epochs: int = 15
    batch_size: int = 2048
    top_k: int = 200
    seed: int = 42


@dataclass(frozen=True)
class DeepFMConfig:
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 15
    batch_size: int = 1024
    seed: int = 42
    patience: int = 3
    min_delta: float = 1e-4


@dataclass(frozen=True)
class EvalConfig:
    k: int = 10
    rating_threshold: float = 4.0
