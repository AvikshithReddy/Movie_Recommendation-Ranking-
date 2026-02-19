from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read_csv(path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols, nrows=nrows)


def load_ratings(data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    path = data_dir / "ratings.csv"
    df = _read_csv(path, usecols=["userId", "movieId", "rating", "timestamp"], nrows=nrows)
    df["userId"] = df["userId"].astype("int32")
    df["movieId"] = df["movieId"].astype("int32")
    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def load_movies(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "movies.csv"
    df = _read_csv(path, usecols=["movieId", "title", "genres"])
    df["movieId"] = df["movieId"].astype("int32")
    return df


def load_tags(data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    path = data_dir / "tags.csv"
    df = _read_csv(path, usecols=["userId", "movieId", "tag", "timestamp"], nrows=nrows)
    df["userId"] = df["userId"].astype("int32")
    df["movieId"] = df["movieId"].astype("int32")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df
