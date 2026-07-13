import os
import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    user_emb = np.load(f"{RESULT_DIR}/lightgcn_user_embeddings.npy")
    item_emb = np.load(f"{RESULT_DIR}/lightgcn_item_embeddings.npy")
    user_degrees = pd.read_csv(f"{DATA_DIR}/user_degrees.csv")
    item_degrees = pd.read_csv(f"{DATA_DIR}/item_degrees.csv")
    user_norms = np.linalg.norm(user_emb, axis=1)
    item_norms = np.linalg.norm(item_emb, axis=1)
    user_df = pd.DataFrame({
        "user_idx": np.arange(len(user_norms)),
        "embedding_norm": user_norms
    })
    item_df = pd.DataFrame({
        "item_idx": np.arange(len(item_norms)),
        "embedding_norm": item_norms
    })
    user_df = user_df.merge(user_degrees, on="user_idx", how="left")
    item_df = item_df.merge(item_degrees, on="item_idx", how="left")
    user_df.to_csv(f"{RESULT_DIR}/user_embedding_norms.csv", index=False)
    item_df.to_csv(f"{RESULT_DIR}/item_embedding_norms.csv", index=False)
    user_summary = user_df.groupby("degree_group", as_index=False)["embedding_norm"].agg(["mean", "median", "std", "count"]).reset_index()
    item_summary = item_df.groupby("degree_group", as_index=False)["embedding_norm"].agg(["mean", "median", "std", "count"]).reset_index()
    user_summary.to_csv(f"{RESULT_DIR}/user_embedding_norm_by_group.csv", index=False)
    item_summary.to_csv(f"{RESULT_DIR}/item_embedding_norm_by_group.csv", index=False)
    user_corr = user_df[["degree", "embedding_norm"]].corr().iloc[0, 1]
    item_corr = item_df[["degree", "embedding_norm"]].corr().iloc[0, 1]
    corr_df = pd.DataFrame([
        {"node_type": "user", "degree_embedding_norm_correlation": user_corr},
        {"node_type": "item", "degree_embedding_norm_correlation": item_corr}
    ])
    corr_df.to_csv(f"{RESULT_DIR}/degree_embedding_norm_correlation.csv", index=False)
    print("User embedding norm by group:")
    print(user_summary)
    print("\nItem embedding norm by group:")
    print(item_summary)
    print("\nDegree-embedding norm correlation:")
    print(corr_df)
    print("\nSaved embedding analysis results.")

if __name__ == "__main__":
    main()