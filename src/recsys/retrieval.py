from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class RetrievalArtifacts:
    model: object
    user_id_map: Dict[int, int]
    item_id_map: Dict[int, int]
    user_index_map: Dict[int, int]
    item_index_map: Dict[int, int]
    user_items: sparse.csr_matrix


def build_interaction_matrix(
    ratings: pd.DataFrame, rating_threshold: float = 4.0
) -> Tuple[sparse.coo_matrix, Dict[int, int], Dict[int, int]]:
    implicit_df = ratings[ratings["rating"] >= rating_threshold]
    user_ids, user_index = pd.factorize(implicit_df["userId"])
    item_ids, item_index = pd.factorize(implicit_df["movieId"])

    data = np.ones(len(implicit_df), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (user_ids, item_ids)), shape=(len(user_index), len(item_index))
    )
    user_id_map = {int(uid): int(idx) for idx, uid in enumerate(user_index)}
    item_id_map = {int(iid): int(idx) for idx, iid in enumerate(item_index)}
    return matrix, user_id_map, item_id_map


def train_als(
    interactions: sparse.coo_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 20,
    alpha: float = 40.0,
):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )
    weighted = (interactions * alpha).tocsr()
    model.fit(weighted)
    return model


def build_artifacts(
    model,
    user_id_map: Dict[int, int],
    item_id_map: Dict[int, int],
    user_items: sparse.csr_matrix,
) -> RetrievalArtifacts:
    user_index_map = {v: k for k, v in user_id_map.items()}
    item_index_map = {v: k for k, v in item_id_map.items()}
    return RetrievalArtifacts(
        model=model,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        user_index_map=user_index_map,
        item_index_map=item_index_map,
        user_items=user_items,
    )


def recommend_for_user(
    artifacts: RetrievalArtifacts,
    user_id: int,
    top_k: int = 200,
    filter_items: set[int] | None = None,
    filter_already_liked_items: bool = True,
) -> list[tuple[int, float]]:
    if user_id not in artifacts.user_id_map:
        return []
    user_idx = artifacts.user_id_map[user_id]
    item_ids, scores = artifacts.model.recommend(
        user_idx,
        artifacts.user_items,
        N=top_k,
        filter_already_liked_items=filter_already_liked_items,
    )
    results = []
    for item_idx, score in zip(item_ids, scores):
        movie_id = artifacts.item_index_map.get(int(item_idx))
        if movie_id is None:
            continue
        if filter_items and movie_id in filter_items:
            continue
        results.append((movie_id, float(score)))
    return results
