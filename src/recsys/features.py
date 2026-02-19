from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def build_user_item_stats(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    global_mean = float(ratings["rating"].mean())
    user_stats = ratings.groupby("userId").agg(
        user_rating_count=("rating", "count"),
        user_rating_mean=("rating", "mean"),
        user_last_ts=("timestamp", "max"),
    )
    item_stats = ratings.groupby("movieId").agg(
        item_rating_count=("rating", "count"),
        item_rating_mean=("rating", "mean"),
        item_last_ts=("timestamp", "max"),
    )
    item_stats["item_popularity_pct"] = item_stats["item_rating_count"].rank(pct=True)
    return user_stats.reset_index(), item_stats.reset_index(), global_mean


def build_tag_features(tags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tags.empty:
        return (
            pd.DataFrame(columns=["userId", "user_tag_count"]),
            pd.DataFrame(columns=["movieId", "item_tag_count"]),
        )
    user_tags = tags.groupby("userId").agg(user_tag_count=("tag", "count")).reset_index()
    item_tags = tags.groupby("movieId").agg(item_tag_count=("tag", "count")).reset_index()
    return user_tags, item_tags


def filter_tags_by_time(tags: pd.DataFrame, max_timestamp: int) -> pd.DataFrame:
    if tags.empty:
        return tags
    return tags[tags["timestamp"] <= max_timestamp].copy()


def build_genre_map(movies: pd.DataFrame) -> dict[int, set[str]]:
    genre_map: dict[int, set[str]] = {}
    for _, row in movies.iterrows():
        genres = row["genres"]
        if pd.isna(genres):
            genre_map[int(row["movieId"])] = set()
        else:
            genre_map[int(row["movieId"])] = set(genres.split("|"))
    return genre_map


def build_user_genres(ratings: pd.DataFrame, genre_map: dict[int, set[str]]) -> Dict[int, set[str]]:
    user_genres: dict[int, set[str]] = {}
    for uid, group in ratings.groupby("userId"):
        genres = set()
        for mid in group["movieId"].tolist():
            genres |= genre_map.get(int(mid), set())
        user_genres[int(uid)] = genres
    return user_genres


def genre_overlap(user_genres: Dict[int, set[str]], genre_map: dict[int, set[str]], user_id: int, movie_id: int) -> float:
    user_set = user_genres.get(int(user_id), set())
    item_set = genre_map.get(int(movie_id), set())
    if not item_set:
        return 0.0
    return len(user_set & item_set) / len(item_set)


def add_embedding_features(
    df: pd.DataFrame,
    user_embedding: np.ndarray,
    item_embedding: np.ndarray,
    user_id_map: dict[int, int],
    item_id_map: dict[int, int],
) -> pd.DataFrame:
    def embed_dot(row) -> float:
        uid = row["userId"]
        mid = row["movieId"]
        uidx = user_id_map.get(int(uid))
        iidx = item_id_map.get(int(mid))
        if uidx is None or iidx is None:
            return 0.0
        return float(np.dot(user_embedding[uidx], item_embedding[iidx]))

    df = df.copy()
    df["embedding_dot"] = df.apply(embed_dot, axis=1)
    return df


def add_genre_overlap_feature(
    df: pd.DataFrame,
    user_genres: Dict[int, set[str]],
    genre_map: dict[int, set[str]],
) -> pd.DataFrame:
    df = df.copy()
    df["genre_overlap"] = df.apply(
        lambda row: genre_overlap(user_genres, genre_map, row["userId"], row["movieId"]), axis=1
    )
    return df


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_recency_days"] = (df["user_last_ts"].max() - df["user_last_ts"]) / 86400.0
    df["item_recency_days"] = (df["item_last_ts"].max() - df["item_last_ts"]) / 86400.0
    return df


def build_feature_frame(
    candidates: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None,
    user_embedding: np.ndarray,
    item_embedding: np.ndarray,
    user_id_map: dict[int, int],
    item_id_map: dict[int, int],
) -> pd.DataFrame:
    user_stats, item_stats, global_mean = build_user_item_stats(ratings)
    genre_map = build_genre_map(movies)
    user_genres = build_user_genres(ratings, genre_map)

    df = candidates.merge(user_stats, on="userId", how="left").merge(item_stats, on="movieId", how="left")
    if tags is not None:
        user_tags, item_tags = build_tag_features(tags)
        df = df.merge(user_tags, on="userId", how="left").merge(item_tags, on="movieId", how="left")
    df = add_genre_overlap_feature(df, user_genres, genre_map)
    df = add_embedding_features(df, user_embedding, item_embedding, user_id_map, item_id_map)
    df = add_recency_features(df)

    df["log_user_rating_count"] = np.log1p(df["user_rating_count"])
    df["log_item_rating_count"] = np.log1p(df["item_rating_count"])
    df["user_rating_mean_centered"] = df["user_rating_mean"] - global_mean
    df["item_rating_mean_centered"] = df["item_rating_mean"] - global_mean

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for col in feature_cols:
        mean = float(df[col].mean())
        std = float(df[col].std())
        if std == 0.0 or np.isnan(std):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_feature_stats(df: pd.DataFrame, feature_cols: list[str], stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in stats:
            continue
        df[col] = (df[col] - stats[col]["mean"]) / stats[col]["std"]
    return df
