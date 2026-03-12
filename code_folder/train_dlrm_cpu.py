# ============================================================
# train_dlrm_cpu.py
# CPU-Optimized DLRM Training (16GB RAM Safe)
# ============================================================

import os
import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score
from tqdm import tqdm


# =========================
# CONFIG
# =========================

PARQUET_PATH = "/app/dataset/ml_features_dataset.parquet"   # CHANGE THIS
LABEL_COL = "label"                     # CHANGE IF NEEDED

EXCLUDE_FEATURES = [
    "cat_feature_17","cat_feature_16","int_feature_7",
    "int_feature_10","cat_feature_6","partition_file","loaded_at"
]

NUM_SHARDS = 16
SHARD_DIR = "dlrm_shards"

HASH_BUCKET_SIZE = 300_000
EMB_DIM_OPTIONS = [8, 16]
LR_OPTIONS = [0.0005, 0.001]
DROPOUT_OPTIONS = [0.0, 0.2]
BATCH_OPTIONS = [2048]
EPOCHS = 2

DEVICE = "cpu"


# =========================
# SHARDING
# =========================

def shard_parquet(parquet_path: str, out_dir: str, num_shards: int):
    os.makedirs(out_dir, exist_ok=True)
    print("Sharding dataset...")

    lf = pl.scan_parquet(parquet_path).with_row_index("__idx")

    for i in range(num_shards):
        shard = (
            lf.filter((pl.col("__idx") % num_shards) == i)
              .drop("__idx")
        )
        shard_path = os.path.join(out_dir, f"shard_{i:02d}.parquet")
        shard.sink_parquet(shard_path)
        print(f"Shard {i+1}/{num_shards} written.")

    print("Sharding complete.")


# =========================
# DATASET
# =========================

class ShardedDataset(IterableDataset):

    def __init__(
        self,
        shard_paths: List[str],
        dense_cols: List[str],
        cat_cols: List[str],
        label_col: str,
        batch_size: int,
        shuffle: bool = True
    ):
        self.shard_paths = shard_paths
        self.dense_cols = dense_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):

        shards = list(self.shard_paths)
        if self.shuffle:
            random.shuffle(shards)

        for sp in shards:

            df = pl.read_parquet(sp)

            # Dense
            X_dense = df.select(
                [pl.col(c).cast(pl.Float32).fill_null(0.0) for c in self.dense_cols]
            ).to_numpy()
            X_dense = torch.tensor(X_dense, dtype=torch.float32)

            # Sparse (hashing)
            sparse_tensors = []
            for c in self.cat_cols:
                col = (
                    df[c]
                    .cast(pl.Utf8)
                    .fill_null("UNK")
                    .hash() % HASH_BUCKET_SIZE
                )
                sparse_tensors.append(
                    torch.tensor(col.to_numpy(), dtype=torch.long)
                )

            X_sparse = torch.stack(sparse_tensors, dim=1)

            y = torch.tensor(
                df[self.label_col].cast(pl.Float32).fill_null(0.0).to_numpy(),
                dtype=torch.float32
            )

            n = len(df)
            for i in range(0, n, self.batch_size):
                yield (
                    X_dense[i:i+self.batch_size],
                    X_sparse[i:i+self.batch_size],
                    y[i:i+self.batch_size]
                )


# =========================
# MODEL
# =========================

class DLRM(nn.Module):

    def __init__(self, num_dense, num_sparse, emb_dim, dropout):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(HASH_BUCKET_SIZE, emb_dim)
            for _ in range(num_sparse)
        ])

        # Bottom MLP
        self.bottom = nn.Sequential(
            nn.Linear(num_dense, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )

        F = num_sparse + 1
        num_pairs = F * (F - 1) // 2
        top_input = emb_dim + num_pairs

        self.top = nn.Sequential(
            nn.Linear(top_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dense, x_sparse):

        dense_out = self.bottom(x_dense)

        embs = [emb(x_sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        V = torch.stack([dense_out] + embs, dim=1)

        G = torch.bmm(V, V.transpose(1,2))

        idx_i, idx_j = torch.triu_indices(G.size(1), G.size(1), offset=1)
        interactions = G[:, idx_i, idx_j]

        x = torch.cat([dense_out, interactions], dim=1)

        return self.top(x).squeeze(1)


# =========================
# METRICS
# =========================

def evaluate(model, loader):

    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for x_dense, x_sparse, y in loader:
            p = model(x_dense, x_sparse)
            ys.append(y.numpy())
            ps.append(p.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_hat = (y_pred >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "logloss": float(log_loss(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_hat)),
        "recall": float(recall_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat))
    }


# =========================
# TRAINING
# =========================

def train_model(train_loader, val_loader, num_dense, num_sparse, emb_dim, lr, dropout):

    model = DLRM(num_dense, num_sparse, emb_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for x_dense, x_sparse, y in tqdm(train_loader):

            optimizer.zero_grad()
            preds = model(x_dense, x_sparse)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics = evaluate(model, val_loader)

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        print("Validation:", metrics)

    return model, metrics


# =========================
# MAIN
# =========================

def main():

    if not os.path.exists(SHARD_DIR):
        shard_parquet(PARQUET_PATH, SHARD_DIR, NUM_SHARDS)

    shard_paths = [os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR)]

    # Infer columns
    sample = pl.read_parquet(shard_paths[0])
    all_cols = sample.columns

    dense_cols = [c for c in all_cols if "int_feature" in c and c not in EXCLUDE_FEATURES]
    cat_cols = [c for c in all_cols if "cat_feature" in c and c not in EXCLUDE_FEATURES]

    split = int(0.8 * len(shard_paths))
    train_shards = shard_paths[:split]
    val_shards = shard_paths[split:]

    best_auc = -1
    best_config = None
    best_model = None
    best_metrics = None

    for emb_dim in EMB_DIM_OPTIONS:
        for lr in LR_OPTIONS:
            for dropout in DROPOUT_OPTIONS:
                for batch in BATCH_OPTIONS:

                    print("\nTesting:", emb_dim, lr, dropout, batch)

                    train_ds = ShardedDataset(train_shards, dense_cols, cat_cols, LABEL_COL, batch)
                    val_ds = ShardedDataset(val_shards, dense_cols, cat_cols, LABEL_COL, batch, shuffle=False)

                    train_loader = DataLoader(train_ds, batch_size=None)
                    val_loader = DataLoader(val_ds, batch_size=None)

                    model, metrics = train_model(
                        train_loader,
                        val_loader,
                        len(dense_cols),
                        len(cat_cols),
                        emb_dim,
                        lr,
                        dropout
                    )

                    if metrics["roc_auc"] > best_auc:
                        best_auc = metrics["roc_auc"]
                        best_config = (emb_dim, lr, dropout, batch)
                        best_model = model
                        best_metrics = metrics

    print("\nBEST CONFIG:", best_config)
    print("BEST METRICS:", best_metrics)

    torch.save(best_model.state_dict(), "dlrm_model.pt")

    with open("metrics.json", "w") as f:
        json.dump(best_metrics, f)

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()