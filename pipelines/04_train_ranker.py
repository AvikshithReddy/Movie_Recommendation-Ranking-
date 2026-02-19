from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, DATA_PROCESSED, DeepFMConfig, EvalConfig, MLFLOW_TRACKING_URI
from recsys.deepfm import predict_deepfm, train_deepfm
from recsys.eval import map_at_k, ndcg_at_k, precision_recall_at_k
from recsys.features import apply_feature_stats, build_feature_frame, compute_feature_stats, filter_tags_by_time
from recsys.mlflow_utils import mlflow_run, setup_mlflow


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


def add_labels(candidates: pd.DataFrame, truth: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    truth_pos = truth[truth["rating"] >= rating_threshold][["userId", "movieId"]]
    truth_pos["label"] = 1
    merged = candidates.merge(truth_pos, on=["userId", "movieId"], how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)
    return merged


def add_indices(df: pd.DataFrame, user_id_map: dict[int, int], item_id_map: dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["userId"].map(user_id_map)
    df["item_idx"] = df["movieId"].map(item_id_map)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df


def main() -> None:
    train = pd.read_parquet(DATA_PROCESSED / "ratings_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "ratings_val.parquet")
    movies = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
    tags = pd.read_parquet(DATA_PROCESSED / "tags.parquet")
    train_max_ts = int(train["timestamp"].max())
    tags = filter_tags_by_time(tags, train_max_ts)

    cand_train = pd.read_parquet(DATA_PROCESSED / "candidates_train.parquet")
    cand_val = pd.read_parquet(DATA_PROCESSED / "candidates_val.parquet")

    user_factors, item_factors = load_embeddings()
    user_id_map, item_id_map = load_maps()

    cfg = EvalConfig()
    ranker_cfg = DeepFMConfig()

    train_labeled = add_labels(cand_train, train, cfg.rating_threshold)
    val_labeled = add_labels(cand_val, val, cfg.rating_threshold)

    train_features = build_feature_frame(
        train_labeled,
        train,
        movies,
        tags,
        user_factors,
        item_factors,
        user_id_map,
        item_id_map,
    )
    val_features = build_feature_frame(
        val_labeled,
        train,
        movies,
        tags,
        user_factors,
        item_factors,
        user_id_map,
        item_id_map,
    )

    feature_stats = compute_feature_stats(train_features, FEATURE_COLS)
    train_features = apply_feature_stats(train_features, FEATURE_COLS, feature_stats)
    val_features = apply_feature_stats(val_features, FEATURE_COLS, feature_stats)

    train_features = add_indices(train_features, user_id_map, item_id_map)
    val_features = add_indices(val_features, user_id_map, item_id_map)

    pos = float(train_labeled["label"].sum())
    neg = float(len(train_labeled) - pos)
    pos_weight = neg / pos if pos > 0 else 1.0

    model, history = train_deepfm(
        train_features,
        val_features,
        FEATURE_COLS,
        num_users=len(user_id_map),
        num_items=len(item_id_map),
        config=ranker_cfg,
    )

    ranker_path = ARTIFACTS_DIR / "deepfm_ranker.pt"
    torch.save(model.state_dict(), ranker_path)

    meta = {
        "num_users": len(user_id_map),
        "num_items": len(item_id_map),
        "feature_cols": FEATURE_COLS,
        "config": {
            "embedding_dim": ranker_cfg.embedding_dim,
            "hidden_dims": list(ranker_cfg.hidden_dims),
            "dropout": ranker_cfg.dropout,
            "lr": ranker_cfg.lr,
            "epochs": ranker_cfg.epochs,
            "batch_size": ranker_cfg.batch_size,
            "patience": ranker_cfg.patience,
            "min_delta": ranker_cfg.min_delta,
        },
        "pos_weight": pos_weight,
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
    }
    with open(ARTIFACTS_DIR / "deepfm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(ARTIFACTS_DIR / "feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)

    val_preds = predict_deepfm(model, val_features, FEATURE_COLS)
    precision, recall = precision_recall_at_k(val_preds, val[val["rating"] >= cfg.rating_threshold], k=cfg.k)
    ndcg = ndcg_at_k(val_preds, val[val["rating"] >= cfg.rating_threshold], k=cfg.k)
    mapk = map_at_k(val_preds, val[val["rating"] >= cfg.rating_threshold], k=cfg.k)

    setup_mlflow(MLFLOW_TRACKING_URI, "recsys_ranker")
    with mlflow_run("deepfm_ranker"):
        import mlflow

        mlflow.log_params(
            {
                "embedding_dim": ranker_cfg.embedding_dim,
                "hidden_dims": ",".join(map(str, ranker_cfg.hidden_dims)),
                "dropout": ranker_cfg.dropout,
                "lr": ranker_cfg.lr,
                "epochs": ranker_cfg.epochs,
                "batch_size": ranker_cfg.batch_size,
                "patience": ranker_cfg.patience,
                "min_delta": ranker_cfg.min_delta,
                "k": cfg.k,
            }
        )
        mlflow.log_metrics({"precision_at_k": precision, "recall_at_k": recall, "ndcg_at_k": ndcg, "map_at_k": mapk})
        mlflow.log_metric("pos_weight", pos_weight)
        for step, (train_loss, val_loss) in enumerate(
            zip(history["train_loss"], history["val_loss"]), start=1
        ):
            mlflow.log_metric("train_loss", train_loss, step=step)
            mlflow.log_metric("val_loss", val_loss, step=step)
        if history.get("best_epoch"):
            mlflow.log_metric("best_epoch", history["best_epoch"])
        if history.get("best_val_loss") is not None:
            mlflow.log_metric("best_val_loss", history["best_val_loss"])
        mlflow.log_artifact(str(ranker_path))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "deepfm_meta.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "feature_stats.json"))


if __name__ == "__main__":
    main()
