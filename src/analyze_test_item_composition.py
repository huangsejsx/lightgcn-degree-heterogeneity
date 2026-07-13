import os
import pandas as pd

DATA_DIR = "data/processed"
RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    user_degrees = pd.read_csv(f"{DATA_DIR}/user_degrees.csv")
    item_degrees = pd.read_csv(f"{DATA_DIR}/item_degrees.csv")
    user_group_map = dict(zip(user_degrees["user_idx"], user_degrees["degree_group"]))
    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    test["user_group"] = test["user_idx"].map(user_group_map).fillna("unknown")
    test["test_item_group"] = test["item_idx"].map(item_group_map).fillna("unknown")
    summary = (
        test.groupby(["user_group", "test_item_group"])
        .size()
        .reset_index(name="count")
    )
    summary["total_in_user_group"] = summary.groupby("user_group")["count"].transform("sum")
    summary["ratio"] = summary["count"] / summary["total_in_user_group"]
    summary.to_csv(f"{RESULT_DIR}/test_item_composition_by_user_group.csv", index=False)
    print(summary)
    print("Saved results/test_item_composition_by_user_group.csv")

if __name__ == "__main__":
    main()