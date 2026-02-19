from __future__ import annotations

import numpy as np
import pandas as pd


def _group_truth(df: pd.DataFrame) -> dict[int, set[int]]:
    truth = {}
    for uid, group in df.groupby("userId"):
        truth[int(uid)] = set(group["movieId"].tolist())
    return truth


def precision_recall_at_k(recs: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> tuple[float, float]:
    truth_map = _group_truth(truth)
    precisions = []
    recalls = []

    for uid, group in recs.groupby("userId"):
        preds = group.sort_values("score", ascending=False).head(k)["movieId"].tolist()
        if not preds:
            continue
        true_items = truth_map.get(int(uid), set())
        if not true_items:
            continue
        hits = len(set(preds) & true_items)
        precisions.append(hits / k)
        recalls.append(hits / len(true_items))

    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def _dcg(relevances: list[int]) -> float:
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)))


def ndcg_at_k(recs: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> float:
    truth_map = _group_truth(truth)
    scores = []

    for uid, group in recs.groupby("userId"):
        preds = group.sort_values("score", ascending=False).head(k)["movieId"].tolist()
        true_items = truth_map.get(int(uid), set())
        if not true_items:
            continue
        rels = [1 if item in true_items else 0 for item in preds]
        dcg = _dcg(rels)
        ideal_rels = sorted(rels, reverse=True)
        idcg = _dcg(ideal_rels)
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def map_at_k(recs: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> float:
    truth_map = _group_truth(truth)
    scores = []

    for uid, group in recs.groupby("userId"):
        preds = group.sort_values("score", ascending=False).head(k)["movieId"].tolist()
        true_items = truth_map.get(int(uid), set())
        if not true_items:
            continue

        hits = 0
        precision_sum = 0.0
        for idx, item in enumerate(preds, start=1):
            if item in true_items:
                hits += 1
                precision_sum += hits / idx
        if hits > 0:
            scores.append(precision_sum / min(len(true_items), k))

    return float(np.mean(scores)) if scores else 0.0


def coverage_at_k(recs: pd.DataFrame, total_items: int, k: int = 10) -> float:
    if total_items == 0:
        return 0.0
    topk = (
        recs.sort_values(["userId", "score"], ascending=[True, False])
        .groupby("userId")
        .head(k)
    )
    unique_items = topk["movieId"].nunique()
    return float(unique_items / total_items)
