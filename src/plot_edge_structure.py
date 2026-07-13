import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"

def plot_forman_by_item_group():
    df = pd.read_csv(f"{RESULT_DIR}/edge_structure_by_item_group.csv")
    order = ["low", "medium", "high"]
    labels = {
        "low": "Tail items",
        "medium": "Medium-degree items",
        "high": "Head items"
    }
    df["item_group"] = pd.Categorical(df["item_group"], categories=order, ordered=True)
    df = df.sort_values("item_group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["item_group"]], df["forman_ricci"])
    plt.title("Mean Forman-Ricci Curvature by Item Group")
    plt.xlabel("Item group")
    plt.ylabel("Mean Forman-Ricci curvature")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/forman_ricci_by_item_group.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_degree_sum_by_item_group():
    df = pd.read_csv(f"{RESULT_DIR}/edge_structure_by_item_group.csv")
    order = ["low", "medium", "high"]
    labels = {
        "low": "Tail items",
        "medium": "Medium-degree items",
        "high": "Head items"
    }
    df["item_group"] = pd.Categorical(df["item_group"], categories=order, ordered=True)
    df = df.sort_values("item_group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["item_group"]], df["degree_sum"])
    plt.title("Mean Edge Degree Sum by Item Group")
    plt.xlabel("Item group")
    plt.ylabel("Mean degree sum")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/degree_sum_by_item_group.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_forman_by_item_group()
    plot_degree_sum_by_item_group()

if __name__ == "__main__":
    main()