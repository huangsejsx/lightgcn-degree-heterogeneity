import os
import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
OUT_DIR = "data/processed"

def assign_group_by_quantile(series, low_q=0.2, high_q=0.8):
    low_th = series.quantile(low_q)
    high_th = series.quantile(high_q)
    def group_fn(x):
        if x <= low_th:
            return "low"
        if x >= high_th:
            return "high"
        return "medium"
    return series.apply(group_fn), low_th, high_th

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    num_users = train["user_idx"].nunique()
    num_items = train["item_idx"].nunique()
    num_edges = len(train)
    density = num_edges / (num_users * num_items)
    sparsity = 1 - density
    user_degree = train.groupby("user_idx").size().rename("degree").reset_index()
    item_degree = train.groupby("item_idx").size().rename("degree").reset_index()
    user_degree["degree_group"], user_low, user_high = assign_group_by_quantile(user_degree["degree"])
    item_degree["degree_group"], item_low, item_high = assign_group_by_quantile(item_degree["degree"])
    graph_stats = pd.DataFrame([{
        "num_users": num_users,
        "num_items": num_items,
        "num_edges": num_edges,
        "density": density,
        "sparsity": sparsity,
        "user_degree_min": user_degree["degree"].min(),
        "user_degree_mean": user_degree["degree"].mean(),
        "user_degree_median": user_degree["degree"].median(),
        "user_degree_max": user_degree["degree"].max(),
        "item_degree_min": item_degree["degree"].min(),
        "item_degree_mean": item_degree["degree"].mean(),
        "item_degree_median": item_degree["degree"].median(),
        "item_degree_max": item_degree["degree"].max(),
        "user_low_threshold": user_low,
        "user_high_threshold": user_high,
        "item_low_threshold": item_low,
        "item_high_threshold": item_high
    }])
    user_degree.to_csv(f"{OUT_DIR}/user_degrees.csv", index=False)
    item_degree.to_csv(f"{OUT_DIR}/item_degrees.csv", index=False)
    graph_stats.to_csv(f"{OUT_DIR}/graph_stats.csv", index=False)
    print(graph_stats.T)
    print("\nUser degree groups:")
    print(user_degree["degree_group"].value_counts())
    print("\nItem degree groups:")
    print(item_degree["degree_group"].value_counts())
    print("\nSaved files:")
    print("data/processed/user_degrees.csv")
    print("data/processed/item_degrees.csv")
    print("data/processed/graph_stats.csv")

if __name__ == "__main__":
    main()