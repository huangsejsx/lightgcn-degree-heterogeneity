from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

ORDER = ["low", "medium", "high"]
LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}


def prepare_data(path):
    df = pd.read_csv(path)
    df["degree_group"] = pd.Categorical(
        df["degree_group"],
        categories=ORDER,
        ordered=True,
    )
    return df.sort_values("degree_group")


def plot_user_groups():
    df = prepare_data(
        RESULTS_DIR / "user_embedding_group_summary.csv"
    )

    plt.figure(figsize=(6, 4))
    plt.bar(
        [LABELS[group] for group in df["degree_group"]],
        df["mean_norm"],
        yerr=df["std_across_seeds"],
        capsize=4,
    )
    plt.xlabel("User degree group")
    plt.ylabel("Mean embedding norm")
    plt.title("User Embedding Norm by Degree Group")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "user_embedding_norm_by_group.png",
        dpi=300,
    )
    plt.close()


def plot_item_groups():
    df = prepare_data(
        RESULTS_DIR / "item_embedding_group_summary.csv"
    )

    plt.figure(figsize=(6, 4))
    plt.bar(
        [LABELS[group] for group in df["degree_group"]],
        df["mean_norm"],
        yerr=df["std_across_seeds"],
        capsize=4,
    )
    plt.xlabel("Item degree group")
    plt.ylabel("Mean embedding norm")
    plt.title("Item Embedding Norm by Degree Group")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "item_embedding_norm_by_group.png",
        dpi=300,
    )
    plt.close()


def main():
    plot_user_groups()
    plot_item_groups()

    print(
        "Saved:",
        RESULTS_DIR / "user_embedding_norm_by_group.png",
    )
    print(
        "Saved:",
        RESULTS_DIR / "item_embedding_norm_by_group.png",
    )


if __name__ == "__main__":
    main()