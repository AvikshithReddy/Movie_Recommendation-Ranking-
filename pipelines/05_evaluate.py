from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, DATA_PROCESSED, DeepFMConfig, EvalConfig
from recsys.deepfm import DeepFM, predict_deepfm
from recsys.eval import coverage_at_k, map_at_k, ndcg_at_k, precision_recall_at_k
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
    test = pd.read_parquet(DATA_PROCESSED / "ratings_test.parquet")
    movies = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
    tags = pd.read_parquet(DATA_PROCESSED / "tags.parquet")
    candidates = pd.read_parquet(DATA_PROCESSED / "candidates_test.parquet")

    cfg = EvalConfig()
    truth = test[test["rating"] >= cfg.rating_threshold]

    retrieval_precision, retrieval_recall = precision_recall_at_k(candidates, truth, k=cfg.k)
    retrieval_ndcg = ndcg_at_k(candidates, truth, k=cfg.k)
    retrieval_mapk = map_at_k(candidates, truth, k=cfg.k)
    retrieval_coverage = coverage_at_k(candidates, total_items=movies["movieId"].nunique(), k=cfg.k)

    print("Retrieval metrics")
    print(f"Precision@{cfg.k}: {retrieval_precision:.4f}")
    print(f"Recall@{cfg.k}: {retrieval_recall:.4f}")
    print(f"NDCG@{cfg.k}: {retrieval_ndcg:.4f}")
    print(f"MAP@{cfg.k}: {retrieval_mapk:.4f}")
    print(f"Coverage@{cfg.k}: {retrieval_coverage:.4f}")

    user_factors, item_factors = load_embeddings()
    user_id_map, item_id_map = load_maps()

    ratings_train = pd.read_parquet(DATA_PROCESSED / "ratings_train.parquet")
    train_max_ts = int(ratings_train["timestamp"].max())
    tags = filter_tags_by_time(tags, train_max_ts)

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

    precision, recall = precision_recall_at_k(ranked, truth, k=cfg.k)
    ndcg = ndcg_at_k(ranked, truth, k=cfg.k)
    mapk = map_at_k(ranked, truth, k=cfg.k)
    coverage = coverage_at_k(ranked, total_items=movies["movieId"].nunique(), k=cfg.k)

    print("Ranker metrics")
    print(f"Precision@{cfg.k}: {precision:.4f}")
    print(f"Recall@{cfg.k}: {recall:.4f}")
    print(f"NDCG@{cfg.k}: {ndcg:.4f}")
    print(f"MAP@{cfg.k}: {mapk:.4f}")
    print(f"Coverage@{cfg.k}: {coverage:.4f}")

    if ndcg < retrieval_ndcg:
        print("Ranker underperforms retrieval. Consider switching to SASRec or XGBoost ranker.")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "k": cfg.k,
        "rating_threshold": cfg.rating_threshold,
        "retrieval": {
            "precision_at_k": float(retrieval_precision),
            "recall_at_k": float(retrieval_recall),
            "ndcg_at_k": float(retrieval_ndcg),
            "map_at_k": float(retrieval_mapk),
            "coverage_at_k": float(retrieval_coverage),
        },
        "ranker": {
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "ndcg_at_k": float(ndcg),
            "map_at_k": float(mapk),
            "coverage_at_k": float(coverage),
        },
        "model_selection": {
            "ranker_better_than_retrieval": bool(ndcg >= retrieval_ndcg),
            "suggestion": "If ranker underperforms retrieval, consider switching to SASRec or XGBoost ranker.",
        },
    }

    out_path = DATA_PROCESSED / "evaluation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
