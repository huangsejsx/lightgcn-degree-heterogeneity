import os
import pandas as pd

RAW_PATH = "data/raw/ml-1m/ratings.dat"
OUT_DIR = "data/processed"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ratings = pd.read_csv(
        RAW_PATH,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    print("Raw ratings:", ratings.shape)
    interactions = ratings[ratings["rating"] >= 4].copy()
    interactions = interactions.sort_values(["user_id", "timestamp"])
    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(interactions["item_id"].unique())
    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}
    interactions["user_idx"] = interactions["user_id"].map(user_map)
    interactions["item_idx"] = interactions["item_id"].map(item_map)
    interactions = interactions[["user_idx", "item_idx", "user_id", "item_id", "rating", "timestamp"]]
    user_mapping = pd.DataFrame({
        "user_id": list(user_map.keys()),
        "user_idx": list(user_map.values())
    })
    item_mapping = pd.DataFrame({
        "item_id": list(item_map.keys()),
        "item_idx": list(item_map.values())
    })
    interactions.to_csv(f"{OUT_DIR}/interactions.csv", index=False)
    user_mapping.to_csv(f"{OUT_DIR}/user_mapping.csv", index=False)
    item_mapping.to_csv(f"{OUT_DIR}/item_mapping.csv", index=False)
    print("Positive interactions:", interactions.shape)
    print("Users:", interactions["user_idx"].nunique())
    print("Items:", interactions["item_idx"].nunique())
    print("Saved files to data/processed/")

if __name__ == "__main__":
    main()