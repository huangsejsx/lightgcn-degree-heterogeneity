import os
import math
import pandas as pd

DATA_DIR = "data/processed"
OUT_DIR = "results"

def recall_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    return 1.0 if ground_truth in recommended_k else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    if ground_truth not in recommended_k:
        return 0.0
    rank = recommended_k.index(ground_truth) + 1
    return 1.0 / math.log2(rank + 1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    item_popularity = train.groupby("item_idx").size().sort_values(ascending=False)
    ranked_items = item_popularity.index.tolist()
    train_user_items = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    results = []
    for k in [10, 20]:
        recalls = []
        ndcgs = []
        for _, row in test.iterrows():
            user = row["user_idx"]
            true_item = row["item_idx"]
            seen_items = train_user_items.get(user, set())
            recommended = [item for item in ranked_items if item not in seen_items]
            recalls.append(recall_at_k(recommended, true_item, k))
            ndcgs.append(ndcg_at_k(recommended, true_item, k))
        results.append({
            "model": "Popularity",
            "K": k,
            "Recall": sum(recalls) / len(recalls),
            "NDCG": sum(ndcgs) / len(ndcgs)
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/popularity_metrics.csv", index=False)
    print(results_df)
    print("Saved results to results/popularity_metrics.csv")

if __name__ == "__main__":
    main()