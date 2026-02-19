from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, DATA_PROCESSED, DeepFMConfig
from recsys.deepfm import DeepFM, predict_deepfm
from recsys.features import apply_feature_stats, build_feature_frame, filter_tags_by_time


FEATURE_COLS = [
    "score",
    "user_rating_count",
    "user_rating_mean",
    "item_rating_count",
    "item_rating_mean",
    "genre_overlap",
    "embedding_dot",
    "user_recency_days",
    "item_recency_days",
    "log_user_rating_count",
    "log_item_rating_count",
    "user_rating_mean_centered",
    "item_rating_mean_centered",
    "item_popularity_pct",
    "user_tag_count",
    "item_tag_count",
]


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    user_factors = np.load(ARTIFACTS_DIR / "embeddings" / "user_factors.npy")
    item_factors = np.load(ARTIFACTS_DIR / "embeddings" / "item_factors.npy")
    return user_factors, item_factors


def load_maps() -> tuple[dict[int, int], dict[int, int]]:
    user_map = pd.read_csv(ARTIFACTS_DIR / "user_map.csv")
    item_map = pd.read_csv(ARTIFACTS_DIR / "item_map.csv")
    user_id_map = dict(zip(user_map["userId"].astype(int), user_map["userIndex"].astype(int)))
    item_id_map = dict(zip(item_map["movieId"].astype(int), item_map["itemIndex"].astype(int)))
    return user_id_map, item_id_map


def add_indices(df: pd.DataFrame, user_id_map: dict[int, int], item_id_map: dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["userId"].map(user_id_map)
    df["item_idx"] = df["movieId"].map(item_id_map)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df


def load_ranker(num_users: int, num_items: int) -> DeepFM:
    with open(ARTIFACTS_DIR / "deepfm_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = DeepFMConfig(
        embedding_dim=meta["config"]["embedding_dim"],
        hidden_dims=tuple(meta["config"]["hidden_dims"]),
        dropout=meta["config"]["dropout"],
        lr=meta["config"]["lr"],
        epochs=meta["config"]["epochs"],
        batch_size=meta["config"]["batch_size"],
        patience=meta["config"].get("patience", 3),
        min_delta=meta["config"].get("min_delta", 1e-4),
    )

    model = DeepFM(num_users, num_items, len(FEATURE_COLS), cfg)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "deepfm_ranker.pt", map_location="cpu"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    candidates = pd.read_parquet(DATA_PROCESSED / f"candidates_{args.split}.parquet")
    movies = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
    tags = pd.read_parquet(DATA_PROCESSED / "tags.parquet")
    ratings_train = pd.read_parquet(DATA_PROCESSED / "ratings_train.parquet")
    train_max_ts = int(ratings_train["timestamp"].max())
    tags = filter_tags_by_time(tags, train_max_ts)

    user_factors, item_factors = load_embeddings()
    user_id_map, item_id_map = load_maps()

    features = build_feature_frame(
        candidates,
        ratings_train,
        movies,
        tags,
        user_factors,
        item_factors,
        user_id_map,
        item_id_map,
    )
    with open(ARTIFACTS_DIR / "feature_stats.json", "r", encoding="utf-8") as f:
        feature_stats = json.load(f)
    features = apply_feature_stats(features, FEATURE_COLS, feature_stats)
    features = add_indices(features, user_id_map, item_id_map)

    model = load_ranker(len(user_id_map), len(item_id_map))
    ranked = predict_deepfm(model, features, FEATURE_COLS)

    ranked = ranked.sort_values(["userId", "score"], ascending=[True, False])
    ranked["rank"] = ranked.groupby("userId").cumcount() + 1
    topk = ranked[ranked["rank"] <= args.top_k]

    out_path = DATA_PROCESSED / f"recommendations_{args.split}.parquet"
    topk.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
