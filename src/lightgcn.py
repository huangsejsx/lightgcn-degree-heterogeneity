import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_DIR = "data/processed"
OUT_DIR = "results"
EMBED_DIM = 64
N_LAYERS = 2
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 20
TOP_K = [10, 20]
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_user_items(df):
    return df.groupby("user_idx")["item_idx"].apply(set).to_dict()

def sample_negative(user, num_items, train_user_items):
    seen = train_user_items[user]
    while True:
        neg_item = random.randint(0, num_items - 1)
        if neg_item not in seen:
            return neg_item

def build_norm_adj(train, num_users, num_items, device):
    user = torch.tensor(train["user_idx"].values, dtype=torch.long)
    item = torch.tensor(train["item_idx"].values, dtype=torch.long) + num_users
    row = torch.cat([user, item])
    col = torch.cat([item, user])
    deg = torch.zeros(num_users + num_items, dtype=torch.float32)
    deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col])
    adj = torch.sparse_coo_tensor(indices, values, (num_users + num_items, num_users + num_items))
    return adj.coalesce().to(device)

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, n_layers, norm_adj):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        user_final, item_final = torch.split(final_emb, [self.num_users, self.num_items])
        return user_final, item_final

    def forward(self, users, pos_items, neg_items):
        user_final, item_final = self.propagate()
        u_emb = user_final[users]
        pos_emb = item_final[pos_items]
        neg_emb = item_final[neg_items]
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        return pos_scores, neg_scores

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

def train_one_epoch(model, optimizer, train_edges, train_user_items, num_items, device):
    model.train()
    np.random.shuffle(train_edges)
    total_loss = 0.0
    num_batches = 0
    for start in range(0, len(train_edges), BATCH_SIZE):
        batch = train_edges[start:start + BATCH_SIZE]
        users = batch[:, 0]
        pos_items = batch[:, 1]
        neg_items = np.array([sample_negative(int(u), num_items, train_user_items) for u in users])
        users = torch.tensor(users, dtype=torch.long, device=device)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)
        optimizer.zero_grad()
        pos_scores, neg_scores = model(users, pos_items, neg_items)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def recall_at_k(ranked_items, true_item, k):
    return 1.0 if true_item in ranked_items[:k] else 0.0

def ndcg_at_k(ranked_items, true_item, k):
    top_k = ranked_items[:k]
    if true_item not in top_k:
        return 0.0
    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)

def evaluate(model, test, train_user_items, device):
    model.eval()
    recalls = {k: [] for k in TOP_K}
    ndcgs = {k: [] for k in TOP_K}
    with torch.no_grad():
        user_emb, item_emb = model.propagate()
        item_emb_t = item_emb.t()
        for _, row in test.iterrows():
            user = int(row["user_idx"])
            true_item = int(row["item_idx"])
            scores = torch.matmul(user_emb[user], item_emb_t).cpu().numpy()
            seen_items = train_user_items.get(user, set())
            scores[list(seen_items)] = -np.inf
            max_k = max(TOP_K)
            ranked_items = np.argpartition(-scores, max_k)[:max_k]
            ranked_items = ranked_items[np.argsort(-scores[ranked_items])].tolist()
            for k in TOP_K:
                recalls[k].append(recall_at_k(ranked_items, true_item, k))
                ndcgs[k].append(ndcg_at_k(ranked_items, true_item, k))
    results = []
    for k in TOP_K:
        results.append({
            "model": "LightGCN",
            "K": k,
            "Recall": float(np.mean(recalls[k])),
            "NDCG": float(np.mean(ndcgs[k]))
        })
    return pd.DataFrame(results)

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    num_users = int(max(train["user_idx"].max(), test["user_idx"].max()) + 1)
    num_items = int(max(train["item_idx"].max(), test["item_idx"].max()) + 1)
    print("Users:", num_users)
    print("Items:", num_items)
    print("Building normalized adjacency...")
    norm_adj = build_norm_adj(train, num_users, num_items, device)
    train_user_items = build_user_items(train)
    train_edges = train[["user_idx", "item_idx"]].values
    model = LightGCN(num_users, num_items, EMBED_DIM, N_LAYERS, norm_adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Training LightGCN...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, train_edges, train_user_items, num_items, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")
    print("Evaluating LightGCN...")
    results_df = evaluate(model, test, train_user_items, device)
    results_df.to_csv(f"{OUT_DIR}/lightgcn_metrics.csv", index=False)
    print(results_df)
    with torch.no_grad():
        user_emb, item_emb = model.propagate()
        np.save(f"{OUT_DIR}/lightgcn_user_embeddings.npy", user_emb.cpu().numpy())
        np.save(f"{OUT_DIR}/lightgcn_item_embeddings.npy", item_emb.cpu().numpy())
    print("Saved results/lightgcn_metrics.csv")
    print("Saved results/lightgcn_user_embeddings.npy")
    print("Saved results/lightgcn_item_embeddings.npy")

if __name__ == "__main__":
    main()