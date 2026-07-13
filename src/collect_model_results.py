import os
import pandas as pd

RESULT_DIR = "results"

def main():
    files = [
        "popularity_metrics.csv",
        "itemcf_metrics.csv",
        "lightgcn_metrics.csv"
    ]
    dfs = []
    for file in files:
        path = f"{RESULT_DIR}/{file}"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Missing file: {path}")
    all_results = pd.concat(dfs, ignore_index=True)
    all_results.to_csv(f"{RESULT_DIR}/all_model_metrics.csv", index=False)
    print(all_results)
    print("Saved results/all_model_metrics.csv")

if __name__ == "__main__":
    main()