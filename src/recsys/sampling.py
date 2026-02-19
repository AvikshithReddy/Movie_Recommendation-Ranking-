from __future__ import annotations

import pandas as pd


def sample_users(ratings: pd.DataFrame, max_users: int | None, seed: int = 42) -> pd.DataFrame:
    if max_users is None:
        return ratings
    users = ratings["userId"].drop_duplicates().sample(n=min(max_users, ratings["userId"].nunique()), random_state=seed)
    return ratings[ratings["userId"].isin(users)].copy()
