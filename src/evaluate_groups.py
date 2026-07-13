import os
import math
import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
RESULT_DIR = "results"
TOP_K = [10, 20]

def recall_at_k(ranked_items, true_item, k):
    return 1.0 if true_item in ranked_items[:k] else 0.0

def ndcg_at_k(ranked_items, true_item, k):
    top_k = ranked_items[:k]
    if true_item not in top_k:
        return 0.0
    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)

def build_user_items(df):
    return df.groupby("user_idx")["item_idx"].apply(set).to_dict()

def evaluate_lightgcn_by_groups():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    user_degrees = pd.read_csv(f"{DATA_DIR}/user_degrees.csv")
    item_degrees = pd.read_csv(f"{DATA_DIR}/item_degrees.csv")
    user_emb = np.load(f"{RESULT_DIR}/lightgcn_user_embeddings.npy")
    item_emb = np.load(f"{RESULT_DIR}/lightgcn_item_embeddings.npy")
    train_user_items = build_user_items(train)
    user_group_map = dict(zip(user_degrees["user_idx"], user_degrees["degree_group"]))
    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    user_group_records = []
    item_group_records = []
    for _, row in test.iterrows():
        user = int(row["user_idx"])
        true_item = int(row["item_idx"])
        scores = user_emb[user] @ item_emb.T
        seen_items = train_user_items.get(user, set())
        scores[list(seen_items)] = -np.inf
        max_k = max(TOP_K)
        ranked_items = np.argpartition(-scores, max_k)[:max_k]
        ranked_items = ranked_items[np.argsort(-scores[ranked_items])].tolist()
        user_group = user_group_map.get(user, "unknown")
        item_group = item_group_map.get(true_item, "unknown")
        for k in TOP_K:
            rec = recall_at_k(ranked_items, true_item, k)
            ndcg = ndcg_at_k(ranked_items, true_item, k)
            user_group_records.append({
                "model": "LightGCN",
                "group_type": "user_degree",
                "group": user_group,
                "K": k,
                "Recall": rec,
                "NDCG": ndcg
            })
            item_group_records.append({
                "model": "LightGCN",
                "group_type": "item_degree",
                "group": item_group,
                "K": k,
                "Recall": rec,
                "NDCG": ndcg
            })
    user_group_df = pd.DataFrame(user_group_records)
    item_group_df = pd.DataFrame(item_group_records)
    user_summary = user_group_df.groupby(["model", "group_type", "group", "K"], as_index=False)[["Recall", "NDCG"]].mean()
    item_summary = item_group_df.groupby(["model", "group_type", "group", "K"], as_index=False)[["Recall", "NDCG"]].mean()
    user_summary.to_csv(f"{RESULT_DIR}/lightgcn_user_group_metrics.csv", index=False)
    item_summary.to_csv(f"{RESULT_DIR}/lightgcn_item_group_metrics.csv", index=False)
    print("User group metrics:")
    print(user_summary)
    print("\nItem group metrics:")
    print(item_summary)
    print("\nSaved results/lightgcn_user_group_metrics.csv")
    print("Saved results/lightgcn_item_group_metrics.csv")

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    evaluate_lightgcn_by_groups()

if __name__ == "__main__":
    main()