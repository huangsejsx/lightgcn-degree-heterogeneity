from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_RAW_PATH = ROOT_DIR / "data" / "raw" / "amazon_beauty" / "All_Beauty.csv"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty"

DEFAULT_DOWNLOAD_URL = (
    "https://jmcauley.ucsd.edu/data/amazon_v2/"
    "categoryFilesSmall/All_Beauty.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Amazon All Beauty ratings-only data for an external "
            "dataset robustness check."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to the raw All_Beauty ratings-only CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for processed Amazon Beauty files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the raw ratings-only file if it is not already present.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_DOWNLOAD_URL,
        help="URL used when --download is supplied.",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating treated as a positive implicit interaction.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=3,
        help="Minimum positive interactions required for each user.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=1,
        help=(
            "Minimum positive interactions required for each item. Keep at 1 "
            "for the first robustness check; increase to 3 or 5 only if the "
            "graph is too sparse."
        ),
    )
    return parser.parse_args()


def maybe_download(raw_path: Path, url: str) -> None:
    if raw_path.exists():
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Amazon Beauty ratings-only data from:\n{url}")
    urllib.request.urlretrieve(url, raw_path)
    print(f"Saved raw file to {raw_path}")


def read_amazon_ratings(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            "Download the All_Beauty ratings-only file first, or rerun with --download."
        )

    preview = pd.read_csv(raw_path, nrows=5)
    columns = set(preview.columns)

    if {"user_id", "parent_asin", "rating", "timestamp"}.issubset(columns):
        ratings = pd.read_csv(
            raw_path,
            usecols=["user_id", "parent_asin", "rating", "timestamp"],
        ).rename(columns={"parent_asin": "item_id"})
        return ratings[["user_id", "item_id", "rating", "timestamp"]]

    if {"reviewerID", "asin", "overall", "unixReviewTime"}.issubset(columns):
        ratings = pd.read_csv(
            raw_path,
            usecols=["reviewerID", "asin", "overall", "unixReviewTime"],
        ).rename(
            columns={
                "reviewerID": "user_id",
                "asin": "item_id",
                "overall": "rating",
                "unixReviewTime": "timestamp",
            }
        )
        return ratings[["user_id", "item_id", "rating", "timestamp"]]

    if {"user_id", "item_id", "rating", "timestamp"}.issubset(columns):
        ratings = pd.read_csv(
            raw_path,
            usecols=["user_id", "item_id", "rating", "timestamp"],
        )
        return ratings[["user_id", "item_id", "rating", "timestamp"]]

    # UCSD Amazon 2018 ratings-only files are headerless:
    # item_id,user_id,rating,timestamp
    ratings = pd.read_csv(
        raw_path,
        names=["item_id", "user_id", "rating", "timestamp"],
    )
    return ratings[["user_id", "item_id", "rating", "timestamp"]]


def iterative_filter(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    filtered = interactions.copy()

    while True:
        before = len(filtered)

        if min_user_interactions > 1:
            user_counts = filtered["user_id"].value_counts()
            keep_users = user_counts[user_counts >= min_user_interactions].index
            filtered = filtered[filtered["user_id"].isin(keep_users)]

        if min_item_interactions > 1:
            item_counts = filtered["item_id"].value_counts()
            keep_items = item_counts[item_counts >= min_item_interactions].index
            filtered = filtered[filtered["item_id"].isin(keep_items)]

        if len(filtered) == before:
            return filtered.reset_index(drop=True)


def split_user_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group = group.sort_values("timestamp")
    train = group.iloc[:-2]
    val = group.iloc[-2:-1]
    test = group.iloc[-1:]
    return train, val, test


def assign_group_by_quantile(
    series: pd.Series,
    low_q: float = 0.2,
    high_q: float = 0.8,
) -> tuple[pd.Series, float, float]:
    low_th = float(series.quantile(low_q))
    high_th = float(series.quantile(high_q))

    def group_fn(value: float) -> str:
        if value <= low_th:
            return "tail"
        if value >= high_th:
            return "head"
        return "medium"

    return series.apply(group_fn), low_th, high_th


def gini(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan")
    if np.amin(arr) < 0:
        arr = arr - np.amin(arr)
    total = arr.sum()
    if total == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * total) - (n + 1) / n)


def build_degree_files(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> dict[str, float | int]:
    user_degrees = train.groupby("user_idx").size().rename("degree").reset_index()
    item_degrees = train.groupby("item_idx").size().rename("degree").reset_index()

    user_degrees["degree_group"], user_low, user_high = assign_group_by_quantile(
        user_degrees["degree"]
    )
    item_degrees["degree_group"], item_low, item_high = assign_group_by_quantile(
        item_degrees["degree"]
    )

    user_degrees.to_csv(out_dir / "user_degrees.csv", index=False)
    item_degrees.to_csv(out_dir / "item_degrees.csv", index=False)

    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    test_groups = test[["user_idx", "item_idx"]].copy()
    test_groups["item_group"] = test_groups["item_idx"].map(item_group_map).fillna(
        "cold_start"
    )
    test_composition = (
        test_groups["item_group"]
        .value_counts()
        .rename_axis("item_group")
        .reset_index(name="count")
    )
    test_composition["share"] = test_composition["count"] / len(test_groups)
    test_composition.to_csv(out_dir / "test_item_composition.csv", index=False)

    item_mean = item_degrees["degree"].mean()
    item_std = item_degrees["degree"].std(ddof=0)

    return {
        "user_low_degree_threshold": user_low,
        "user_high_degree_threshold": user_high,
        "item_low_degree_threshold": item_low,
        "item_high_degree_threshold": item_high,
        "train_users": int(train["user_idx"].nunique()),
        "train_items": int(train["item_idx"].nunique()),
        "item_degree_mean": float(item_mean),
        "item_degree_cv": float(item_std / item_mean) if item_mean else float("nan"),
        "item_degree_gini": gini(item_degrees["degree"]),
        "cold_start_test_items": int((test_groups["item_group"] == "cold_start").sum()),
        "cold_start_test_share": float(
            (test_groups["item_group"] == "cold_start").mean()
        ),
    }


def main() -> None:
    args = parse_args()
    raw_path = args.raw_path
    out_dir = args.out_dir

    if args.download:
        maybe_download(raw_path, args.download_url)

    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = read_amazon_ratings(raw_path)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    raw_rows = len(ratings)
    raw_users = ratings["user_id"].nunique()
    raw_items = ratings["item_id"].nunique()

    interactions = ratings[ratings["rating"] >= args.min_rating].copy()
    interactions = interactions.sort_values(["user_id", "item_id", "timestamp"])
    interactions = interactions.drop_duplicates(["user_id", "item_id"], keep="first")
    interactions = interactions.sort_values(["user_id", "timestamp"])

    positive_rows_before_filter = len(interactions)

    interactions = iterative_filter(
        interactions=interactions,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )

    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(interactions["item_id"].unique())
    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    interactions["user_idx"] = interactions["user_id"].map(user_map)
    interactions["item_idx"] = interactions["item_id"].map(item_map)
    interactions = interactions[
        ["user_idx", "item_idx", "user_id", "item_id", "rating", "timestamp"]
    ].sort_values(["user_idx", "timestamp"])

    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in interactions.groupby("user_idx"):
        train, val, test = split_user_group(group)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    interactions.to_csv(out_dir / "interactions.csv", index=False)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    pd.DataFrame(
        {"user_id": list(user_map.keys()), "user_idx": list(user_map.values())}
    ).to_csv(out_dir / "user_mapping.csv", index=False)
    pd.DataFrame(
        {"item_id": list(item_map.keys()), "item_idx": list(item_map.values())}
    ).to_csv(out_dir / "item_mapping.csv", index=False)

    degree_summary = build_degree_files(train_df, test_df, out_dir)

    summary = {
        "dataset": "Amazon All Beauty",
        "raw_rows": raw_rows,
        "raw_users": raw_users,
        "raw_items": raw_items,
        "min_rating": args.min_rating,
        "positive_rows_before_filter": positive_rows_before_filter,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "positive_interactions_after_filter": len(interactions),
        "users_after_filter": interactions["user_idx"].nunique(),
        "items_after_filter": interactions["item_idx"].nunique(),
        "train_interactions": len(train_df),
        "validation_interactions": len(val_df),
        "test_interactions": len(test_df),
        **degree_summary,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "dataset_summary.csv", index=False)

    print("\nAmazon Beauty preprocessing complete")
    print("=" * 45)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved processed files to: {out_dir}")


if __name__ == "__main__":
    main()
