from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from recsys.config import ARTIFACTS_DIR, DATA_PROCESSED, DATA_RAW, LightGCNConfig, MLFLOW_TRACKING_URI
from recsys.io import load_ratings
from recsys.lightgcn import build_interactions, build_norm_adj, train_lightgcn
from recsys.mlflow_utils import mlflow_run, setup_mlflow


def _load_train() -> pd.DataFrame:
    path = DATA_PROCESSED / "ratings_train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return load_ratings(DATA_RAW)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ratings = _load_train()
    cfg = LightGCNConfig()

    interactions, user_id_map, item_id_map = build_interactions(ratings, rating_threshold=4.0)
    num_users = len(user_id_map)
    num_items = len(item_id_map)

    adj = build_norm_adj(interactions, num_users, num_items)

    start = time.time()
    model = train_lightgcn(
        interactions,
        num_users,
        num_items,
        adj,
        embedding_dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
        lr=cfg.lr,
        reg=cfg.reg,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    duration = time.time() - start

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    adj = adj.to(device)
    model.eval()
    with torch.no_grad():
        user_factors, item_factors = model(adj)
        user_factors = user_factors.cpu().numpy().astype("float32")
        item_factors = item_factors.cpu().numpy().astype("float32")

    (ARTIFACTS_DIR / "embeddings").mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "embeddings" / "user_factors.npy", user_factors)
    np.save(ARTIFACTS_DIR / "embeddings" / "item_factors.npy", item_factors)

    user_map = pd.DataFrame({"userId": list(user_id_map.keys()), "userIndex": list(user_id_map.values())})
    item_map = pd.DataFrame({"movieId": list(item_id_map.keys()), "itemIndex": list(item_id_map.values())})
    user_map.to_csv(ARTIFACTS_DIR / "user_map.csv", index=False)
    item_map.to_csv(ARTIFACTS_DIR / "item_map.csv", index=False)

    model_path = ARTIFACTS_DIR / "lightgcn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": cfg.embedding_dim,
            "n_layers": cfg.n_layers,
        },
        model_path,
    )

    setup_mlflow(MLFLOW_TRACKING_URI, "recsys_retrieval")
    with mlflow_run("lightgcn_retrieval"):
        import mlflow

        mlflow.log_params(
            {
                "embedding_dim": cfg.embedding_dim,
                "n_layers": cfg.n_layers,
                "lr": cfg.lr,
                "reg": cfg.reg,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
            }
        )
        mlflow.log_metrics(
            {
                "train_users": num_users,
                "train_items": num_items,
                "train_interactions": int(len(interactions)),
                "train_duration_sec": duration,
            }
        )
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "user_map.csv"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "item_map.csv"))


if __name__ == "__main__":
    main()
