import pandas as pd

from recsys.splits import time_split


def test_time_split_no_overlap():
    data = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "movieId": [10, 11, 12, 13, 20, 21, 22, 23, 24],
            "rating": [4, 3, 5, 4, 4, 2, 5, 4, 3],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4, 5],
        }
    )
    train, val, test = time_split(data, test_ratio=0.25, val_ratio=0.25, min_ratings=1)

    train_pairs = set(zip(train.userId, train.movieId))
    val_pairs = set(zip(val.userId, val.movieId))
    test_pairs = set(zip(test.userId, test.movieId))

    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)
