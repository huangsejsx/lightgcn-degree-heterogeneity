import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"

def plot_user_embedding_norm():
    df = pd.read_csv(f"{RESULT_DIR}/user_embedding_norm_by_group.csv")
    order = ["low", "medium", "high"]
    labels = {
        "low": "Low-degree users",
        "medium": "Medium-degree users",
        "high": "High-degree users"
    }
    df["degree_group"] = pd.Categorical(df["degree_group"], categories=order, ordered=True)
    df = df.sort_values("degree_group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["degree_group"]], df["mean"])
    plt.title("User Embedding Norm by Degree Group")
    plt.xlabel("User group")
    plt.ylabel("Mean embedding norm")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/user_embedding_norm_by_group.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_item_embedding_norm():
    df = pd.read_csv(f"{RESULT_DIR}/item_embedding_norm_by_group.csv")
    order = ["low", "medium", "high"]
    labels = {
        "low": "Tail items",
        "medium": "Medium-degree items",
        "high": "Head items"
    }
    df["degree_group"] = pd.Categorical(df["degree_group"], categories=order, ordered=True)
    df = df.sort_values("degree_group")
    plt.figure(figsize=(7, 5))
    plt.bar([labels[g] for g in df["degree_group"]], df["mean"])
    plt.title("Item Embedding Norm by Degree Group")
    plt.xlabel("Item group")
    plt.ylabel("Mean embedding norm")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/item_embedding_norm_by_group.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_user_embedding_norm()
    plot_item_embedding_norm()

if __name__ == "__main__":
    main()