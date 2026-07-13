import os
import pandas as pd

DATA_DIR = "data/processed"
RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    user_degrees = pd.read_csv(f"{DATA_DIR}/user_degrees.csv")
    item_degrees = pd.read_csv(f"{DATA_DIR}/item_degrees.csv")
    user_deg_map = dict(zip(user_degrees["user_idx"], user_degrees["degree"]))
    item_deg_map = dict(zip(item_degrees["item_idx"], item_degrees["degree"]))
    user_group_map = dict(zip(user_degrees["user_idx"], user_degrees["degree_group"]))
    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    edges = train[["user_idx", "item_idx"]].copy()
    edges["user_degree"] = edges["user_idx"].map(user_deg_map)
    edges["item_degree"] = edges["item_idx"].map(item_deg_map)
    edges["user_group"] = edges["user_idx"].map(user_group_map)
    edges["item_group"] = edges["item_idx"].map(item_group_map)
    edges["degree_sum"] = edges["user_degree"] + edges["item_degree"]
    edges["degree_diff"] = (edges["user_degree"] - edges["item_degree"]).abs()
    edges["forman_ricci"] = 4 - edges["degree_sum"]
    edges.to_csv(f"{DATA_DIR}/edge_structural_features.csv", index=False)
    summary = edges[["degree_sum", "degree_diff", "forman_ricci"]].agg(["mean", "median", "std", "min", "max"]).reset_index()
    summary.to_csv(f"{RESULT_DIR}/edge_structure_summary.csv", index=False)
    item_group_summary = edges.groupby("item_group", as_index=False)[["degree_sum", "degree_diff", "forman_ricci"]].mean()
    item_group_summary.to_csv(f"{RESULT_DIR}/edge_structure_by_item_group.csv", index=False)
    user_item_group_summary = edges.groupby(["user_group", "item_group"], as_index=False)[["degree_sum", "degree_diff", "forman_ricci"]].mean()
    user_item_group_summary.to_csv(f"{RESULT_DIR}/edge_structure_by_user_item_group.csv", index=False)
    print("Edge structure summary:")
    print(summary)
    print("\nEdge structure by item group:")
    print(item_group_summary)
    print("\nEdge structure by user-item group:")
    print(user_item_group_summary)
    print("\nSaved data/processed/edge_structural_features.csv")
    print("Saved results/edge_structure_summary.csv")
    print("Saved results/edge_structure_by_item_group.csv")
    print("Saved results/edge_structure_by_user_item_group.csv")

if __name__ == "__main__":
    main()