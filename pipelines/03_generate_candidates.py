from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from recsys.config import ARTIFACTS_DIR, DATA_PROCESSED, LightGCNConfig


def load_maps() -> tuple[dict[int, int], np.ndarray]:
    user_map = pd.read_csv(ARTIFACTS_DIR / "user_map.csv")
    item_map = pd.read_csv(ARTIFACTS_DIR / "item_map.csv")
    user_id_map = dict(zip(user_map["userId"].astype(int), user_map["userIndex"].astype(int)))
    item_map = item_map.sort_values("itemIndex")
    item_index_to_id = item_map["movieId"].astype(int).to_numpy()
    return user_id_map, item_index_to_id


def build_user_pos(ratings: pd.DataFrame, user_id_map: dict[int, int], item_index_to_id: np.ndarray) -> list[set[int]]:
    num_users = len(user_id_map)
    item_id_to_index = {int(mid): idx for idx, mid in enumerate(item_index_to_id)}
    user_pos = [set() for _ in range(num_users)]
    for row in ratings.itertuples():
        uidx = user_id_map.get(int(row.userId))
        iidx = item_id_to_index.get(int(row.movieId))
        if uidx is None or iidx is None:
            continue
        if row.rating >= 4.0:
            user_pos[uidx].add(iidx)
    return user_pos


def load_split(split: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"ratings_{split}.parquet"
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    ratings = load_split(args.split)
    users = ratings["userId"].drop_duplicates().tolist()

    user_id_map, item_index_to_id = load_maps()
    cfg = LightGCNConfig()

    user_factors = np.load(ARTIFACTS_DIR / "embeddings" / "user_factors.npy")
    item_factors = np.load(ARTIFACTS_DIR / "embeddings" / "item_factors.npy")

    filter_pos = None
    if args.split != "train":
        train_ratings = pd.read_parquet(DATA_PROCESSED / "ratings_train.parquet")
        filter_pos = build_user_pos(train_ratings, user_id_map, item_index_to_id)

    rows = []
    batch_size = 256
    user_ids = [int(uid) for uid in users if int(uid) in user_id_map]
    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start : start + batch_size]
        batch_idx = np.array([user_id_map[uid] for uid in batch_user_ids], dtype=int)
        scores = user_factors[batch_idx] @ item_factors.T

        for row_idx, user_id in enumerate(batch_user_ids):
            user_scores = scores[row_idx]
            if filter_pos is not None:
                for item_idx in filter_pos[batch_idx[row_idx]]:
                    user_scores[item_idx] = -np.inf
            topk_idx = np.argpartition(-user_scores, cfg.top_k - 1)[: cfg.top_k]
            topk_idx = topk_idx[np.argsort(-user_scores[topk_idx])]
            for item_idx in topk_idx:
                rows.append(
                    {
                        "userId": int(user_id),
                        "movieId": int(item_index_to_id[item_idx]),
                        "score": float(user_scores[item_idx]),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_parquet(DATA_PROCESSED / f"candidates_{args.split}.parquet", index=False)


if __name__ == "__main__":
    main()
