import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"

def plot_user_groups():
    df = pd.read_csv(f"{RESULT_DIR}/lightgcn_user_group_metrics.csv")
    df = df[df["K"] == 20].copy()
    order = ["low", "medium", "high"]
    labels = {
        "low": "Low-degree users",
        "medium": "Medium-degree users",
        "high": "High-degree users"
    }
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df = df.sort_values("group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["group"]], df["Recall"])
    plt.title("LightGCN Recall@20 by User Degree Group")
    plt.xlabel("User group")
    plt.ylabel("Recall@20")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/lightgcn_user_group_recall20.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_item_groups():
    df = pd.read_csv(f"{RESULT_DIR}/lightgcn_item_group_metrics.csv")
    df = df[(df["K"] == 20) & (df["group"] != "unknown")].copy()
    order = ["low", "medium", "high"]
    labels = {
        "low": "Tail items",
        "medium": "Medium-degree items",
        "high": "Head items"
    }
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df = df.sort_values("group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["group"]], df["Recall"])
    plt.title("LightGCN Recall@20 by Item Degree Group")
    plt.xlabel("Item group")
    plt.ylabel("Recall@20")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/lightgcn_item_group_recall20.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_user_groups()
    plot_item_groups()

if __name__ == "__main__":
    main()