import os
import math
import pandas as pd
from collections import defaultdict

DATA_DIR = "data/processed"
OUT_DIR = "results"
TOP_SIM_ITEMS = 100

def recall_at_k(recommended, ground_truth, k):
    return 1.0 if ground_truth in recommended[:k] else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    if ground_truth not in recommended_k:
        return 0.0
    rank = recommended_k.index(ground_truth) + 1
    return 1.0 / math.log2(rank + 1)

def build_item_similarity(train):
    user_items = train.groupby("user_idx")["item_idx"].apply(list).to_dict()
    item_count = defaultdict(int)
    co_count = defaultdict(lambda: defaultdict(float))
    for items in user_items.values():
        unique_items = list(set(items))
        for item in unique_items:
            item_count[item] += 1
        for i in unique_items:
            for j in unique_items:
                if i == j:
                    continue
                co_count[i][j] += 1.0 / math.log2(len(unique_items) + 1)
    item_sim = {}
    for i, related_items in co_count.items():
        sims = []
        for j, cij in related_items.items():
            sim = cij / math.sqrt(item_count[i] * item_count[j])
            sims.append((j, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:TOP_SIM_ITEMS]
        item_sim[i] = sims
    return item_sim

def recommend(user, train_user_items, item_sim, all_items, topn=20):
    seen_items = train_user_items.get(user, set())
    scores = defaultdict(float)
    for item in seen_items:
        for sim_item, sim_score in item_sim.get(item, []):
            if sim_item not in seen_items:
                scores[sim_item] += sim_score
    if len(scores) < topn:
        for item in all_items:
            if item not in seen_items and item not in scores:
                scores[item] += 0.0
            if len(scores) >= topn:
                break
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:topn]]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    print("Building item similarity...")
    item_sim = build_item_similarity(train)
    print("Item similarity built.")
    train_user_items = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    all_items = train["item_idx"].value_counts().index.tolist()
    results = []
    for k in [10, 20]:
        recalls = []
        ndcgs = []
        for _, row in test.iterrows():
            user = row["user_idx"]
            true_item = row["item_idx"]
            recommended = recommend(user, train_user_items, item_sim, all_items, topn=20)
            recalls.append(recall_at_k(recommended, true_item, k))
            ndcgs.append(ndcg_at_k(recommended, true_item, k))
        results.append({
            "model": "ItemCF",
            "K": k,
            "Recall": sum(recalls) / len(recalls),
            "NDCG": sum(ndcgs) / len(ndcgs)
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/itemcf_metrics.csv", index=False)
    print(results_df)
    print("Saved results to results/itemcf_metrics.csv")

if __name__ == "__main__":
    main()