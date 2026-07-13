import os
import pandas as pd

IN_PATH = "data/processed/interactions.csv"
OUT_DIR = "data/processed"

def split_user_group(group):
    group = group.sort_values("timestamp")
    if len(group) < 3:
        return None, None, None
    train = group.iloc[:-2]
    val = group.iloc[-2:-1]
    test = group.iloc[-1:]
    return train, val, test

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    interactions = pd.read_csv(IN_PATH)
    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in interactions.groupby("user_idx"):
        train, val, test = split_user_group(group)
        if train is not None:
            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(test)
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
    print("Users in train:", train_df["user_idx"].nunique())
    print("Users in val:", val_df["user_idx"].nunique())
    print("Users in test:", test_df["user_idx"].nunique())
    print("Items in train:", train_df["item_idx"].nunique())
    print("Saved train/val/test to data/processed/")

if __name__ == "__main__":
    main()