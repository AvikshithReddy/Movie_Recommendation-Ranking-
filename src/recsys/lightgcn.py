from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class LightGCNArtifacts:
    user_id_map: Dict[int, int]
    item_id_map: Dict[int, int]
    user_index_map: Dict[int, int]
    item_index_map: Dict[int, int]


def build_interactions(
    ratings: pd.DataFrame, rating_threshold: float = 4.0
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    implicit_df = ratings[ratings["rating"] >= rating_threshold].copy()
    user_codes, user_index = pd.factorize(implicit_df["userId"])
    item_codes, item_index = pd.factorize(implicit_df["movieId"])
    implicit_df["user_idx"] = user_codes
    implicit_df["item_idx"] = item_codes

    user_id_map = {int(uid): int(idx) for idx, uid in enumerate(user_index)}
    item_id_map = {int(iid): int(idx) for idx, iid in enumerate(item_index)}
    return implicit_df, user_id_map, item_id_map


def build_norm_adj(
    interactions: pd.DataFrame, num_users: int, num_items: int
) -> torch.Tensor:
    user_idx = interactions["user_idx"].to_numpy()
    item_idx = interactions["item_idx"].to_numpy()

    rows = np.concatenate([user_idx, item_idx + num_users])
    cols = np.concatenate([item_idx + num_users, user_idx])

    num_nodes = num_users + num_items
    deg = np.bincount(rows, minlength=num_nodes)
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

    values = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    values_t = torch.tensor(values, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values_t, (num_nodes, num_nodes)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, n_layers: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return out[: self.num_users], out[self.num_users :]


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


def sample_batch(
    interactions: pd.DataFrame,
    num_items: int,
    user_pos_sets: list[set[int]],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(interactions), size=batch_size)
    users = interactions["user_idx"].to_numpy()[idx]
    pos_items = interactions["item_idx"].to_numpy()[idx]

    neg_items = rng.integers(0, num_items, size=batch_size)
    for i in range(batch_size):
        tries = 0
        while neg_items[i] in user_pos_sets[users[i]] and tries < 10:
            neg_items[i] = rng.integers(0, num_items)
            tries += 1
    return users, pos_items, neg_items


def train_lightgcn(
    interactions: pd.DataFrame,
    num_users: int,
    num_items: int,
    adj: torch.Tensor,
    embedding_dim: int = 64,
    n_layers: int = 2,
    lr: float = 1e-3,
    reg: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 1024,
    seed: int = 42,
    device: str | None = None,
) -> LightGCN:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LightGCN(num_users, num_items, embedding_dim, n_layers).to(device)
    adj = adj.to(device)

    user_pos_sets = [set() for _ in range(num_users)]
    for row in interactions.itertuples():
        user_pos_sets[row.user_idx].add(row.item_idx)

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    steps_per_epoch = max(1, len(interactions) // batch_size)
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = sample_batch(
                interactions, num_items, user_pos_sets, batch_size, rng
            )
            users_t = torch.tensor(users, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)

            user_emb, item_emb = model(adj)

            loss = bpr_loss(user_emb[users_t], item_emb[pos_t], item_emb[neg_t])
            reg_loss = reg * (
                user_emb[users_t].pow(2).sum()
                + item_emb[pos_t].pow(2).sum()
                + item_emb[neg_t].pow(2).sum()
            ) / batch_size
            total_loss = loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return model


def build_artifacts(
    user_id_map: Dict[int, int],
    item_id_map: Dict[int, int],
) -> LightGCNArtifacts:
    user_index_map = {v: k for k, v in user_id_map.items()}
    item_index_map = {v: k for k, v in item_id_map.items()}
    return LightGCNArtifacts(
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        user_index_map=user_index_map,
        item_index_map=item_index_map,
    )
