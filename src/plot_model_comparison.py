import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    df = pd.read_csv(f"{RESULT_DIR}/all_model_metrics.csv")
    metrics = ["Recall", "NDCG"]
    for metric in metrics:
        pivot = df.pivot(index="model", columns="K", values=metric)
        pivot = pivot.loc[["Popularity", "ItemCF", "LightGCN"]]
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"{metric}@K Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.legend(title="K")
        plt.xticks(rotation=0)
        plt.tight_layout()
        out_path = f"{RESULT_DIR}/{metric.lower()}_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()