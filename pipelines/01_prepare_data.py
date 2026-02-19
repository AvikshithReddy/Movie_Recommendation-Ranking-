from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recsys.config import DATA_PROCESSED, DATA_RAW, SamplingConfig, SplitConfig
from recsys.io import load_movies, load_ratings, load_tags
from recsys.sampling import sample_users
from recsys.splits import time_split


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(DATA_RAW)
    movies = load_movies(DATA_RAW)
    tags = load_tags(DATA_RAW)

    sampling_cfg = SamplingConfig()
    ratings = ratings.drop_duplicates(subset=["userId", "movieId", "timestamp"])
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]
    ratings = sample_users(ratings, sampling_cfg.max_users, sampling_cfg.seed)
    valid_users = set(ratings["userId"].unique())
    valid_items = set(ratings["movieId"].unique())
    tags = tags[tags["userId"].isin(valid_users) & tags["movieId"].isin(valid_items)].copy()

    split_cfg = SplitConfig()
    train, val, test = time_split(
        ratings,
        test_ratio=split_cfg.test_ratio,
        val_ratio=split_cfg.val_ratio,
        min_ratings=split_cfg.min_ratings,
        min_item_ratings=split_cfg.min_item_ratings,
    )

    train.to_parquet(DATA_PROCESSED / "ratings_train.parquet", index=False)
    val.to_parquet(DATA_PROCESSED / "ratings_val.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "ratings_test.parquet", index=False)
    movies.to_parquet(DATA_PROCESSED / "movies.parquet", index=False)
    tags.to_parquet(DATA_PROCESSED / "tags.parquet", index=False)

    summary = {
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "items": int(ratings["movieId"].nunique()),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "sampling": {
            "max_users": sampling_cfg.max_users,
            "seed": sampling_cfg.seed,
        },
        "split": {
            "test_ratio": split_cfg.test_ratio,
            "val_ratio": split_cfg.val_ratio,
            "min_ratings": split_cfg.min_ratings,
            "min_item_ratings": split_cfg.min_item_ratings,
        },
    }

    with open(DATA_PROCESSED / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
