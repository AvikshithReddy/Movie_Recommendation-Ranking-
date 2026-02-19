from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recsys.config import DeepFMConfig


class RankDataset(Dataset):
    def __init__(self, user_idx: np.ndarray, item_idx: np.ndarray, dense: np.ndarray, labels: np.ndarray):
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.dense = dense
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.user_idx[idx],
            self.item_idx[idx],
            self.dense[idx],
            self.labels[idx],
        )


class DeepFM(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_dense: int, config: DeepFMConfig):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, config.embedding_dim)
        self.item_emb = nn.Embedding(num_items, config.embedding_dim)
        self.dense_proj = nn.Linear(num_dense, config.embedding_dim)

        dnn_input_dim = config.embedding_dim * 2 + num_dense
        layers = []
        in_dim = dnn_input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        d = self.dense_proj(dense)

        fields = torch.stack([u, i, d], dim=1)  # (B, F, K)
        sum_emb = fields.sum(dim=1)
        fm = 0.5 * (sum_emb.pow(2) - fields.pow(2).sum(dim=1)).sum(dim=1, keepdim=True)

        dnn_in = torch.cat([u, i, dense], dim=1)
        dnn_out = self.dnn(dnn_in)

        out = fm + dnn_out
        return out.squeeze(1)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_deepfm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    num_users: int,
    num_items: int,
    config: DeepFMConfig,
) -> tuple[DeepFM, dict[str, list[float]]]:
    device = _device()
    torch.manual_seed(config.seed)

    train_ds = RankDataset(
        train_df["user_idx"].to_numpy(),
        train_df["item_idx"].to_numpy(),
        train_df[feature_cols].to_numpy(dtype=np.float32),
        train_df["label"].to_numpy(dtype=np.float32),
    )
    val_ds = RankDataset(
        val_df["user_idx"].to_numpy(),
        val_df["item_idx"].to_numpy(),
        val_df[feature_cols].to_numpy(dtype=np.float32),
        val_df["label"].to_numpy(dtype=np.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = DeepFM(num_users, num_items, len(feature_cols), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = neg / pos if pos > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for user_idx, item_idx, dense, labels in train_loader:
            user_idx = user_idx.to(device).long()
            item_idx = item_idx.to(device).long()
            dense = dense.to(device).float()
            labels = labels.to(device).float()

            preds = model(user_idx, item_idx, dense)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for u, i, d, labels in val_loader:
                u = u.to(device).long()
                i = i.to(device).long()
                d = d.to(device).float()
                labels = labels.to(device).float()
                preds = model(u, i, d)
                loss = loss_fn(preds, labels)
                val_loss += float(loss.item())
                val_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_batches)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_loss - config.min_delta:
            best_loss = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss if best_loss < float("inf") else None

    return model, history


def predict_deepfm(model: DeepFM, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    device = _device()
    model = model.to(device)
    model.eval()

    user_idx = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long, device=device)
    item_idx = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long, device=device)
    dense = torch.tensor(df[feature_cols].to_numpy(dtype=np.float32), device=device)

    with torch.no_grad():
        scores = model(user_idx, item_idx, dense).cpu().numpy()

    out = df[["userId", "movieId"]].copy()
    out["score"] = scores
    return out
