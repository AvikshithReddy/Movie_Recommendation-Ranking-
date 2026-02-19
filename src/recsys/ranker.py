from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("userId").size().tolist()


@dataclass
class RankerArtifacts:
    model: object
    feature_cols: list[str]


def train_ranker(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
):
    from lightgbm import LGBMRanker

    group = build_group_sizes(train_df)

    model = LGBMRanker(
        objective="lambdarank",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=200,
        max_depth=-1,
        min_data_in_leaf=20,
        lambda_l1=0.0,
        lambda_l2=1.0,
    )

    model.fit(
        train_df[feature_cols],
        train_df[label_col],
        group=group,
    )

    return model


def predict_ranker(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    preds = model.predict(df[feature_cols])
    out = df[["userId", "movieId"]].copy()
    out["score"] = preds
    return out
