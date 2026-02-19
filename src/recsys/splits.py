from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    min_ratings: int = 5,
    min_item_ratings: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_ratio + val_ratio >= 1.0:
        raise ValueError("test_ratio + val_ratio must be < 1")

    user_counts = ratings.groupby("userId")["movieId"].transform("count")
    item_counts = ratings.groupby("movieId")["userId"].transform("count")
    filtered = ratings[(user_counts >= min_ratings) & (item_counts >= min_item_ratings)].copy()

    filtered = filtered.sort_values(["userId", "timestamp", "movieId"])
    filtered["rank"] = filtered.groupby("userId").cumcount()
    filtered["user_count"] = filtered.groupby("userId")["movieId"].transform("count")

    test_cut = (filtered["user_count"] * (1 - test_ratio)).astype(int)
    val_cut = (filtered["user_count"] * (1 - test_ratio - val_ratio)).astype(int)

    split = np.where(filtered["rank"] >= test_cut, "test", np.where(filtered["rank"] >= val_cut, "val", "train"))
    filtered["split"] = split

    train = filtered[filtered["split"] == "train"].drop(columns=["rank", "user_count", "split"])
    val = filtered[filtered["split"] == "val"].drop(columns=["rank", "user_count", "split"])
    test = filtered[filtered["split"] == "test"].drop(columns=["rank", "user_count", "split"])

    return train, val, test
