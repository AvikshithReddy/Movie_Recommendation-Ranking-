import pandas as pd

from recsys.eval import map_at_k, ndcg_at_k, precision_recall_at_k


def test_metrics_basic():
    recs = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [10, 11, 12, 20, 21],
            "score": [0.9, 0.8, 0.1, 0.7, 0.6],
        }
    )
    truth = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 21],
        }
    )

    precision, recall = precision_recall_at_k(recs, truth, k=2)
    ndcg = ndcg_at_k(recs, truth, k=2)
    mapk = map_at_k(recs, truth, k=2)

    assert precision > 0
    assert recall > 0
    assert ndcg > 0
    assert mapk > 0
