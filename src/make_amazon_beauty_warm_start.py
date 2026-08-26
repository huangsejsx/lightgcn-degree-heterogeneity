from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_IN_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a warm-start evaluation view for Amazon Beauty by keeping "
            "only test interactions whose held-out item appears in the training graph."
        )
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=DEFAULT_IN_DIR,
        help="Input processed Amazon Beauty directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output processed directory with warm-start test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "interactions.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "user_mapping.csv",
        "item_mapping.csv",
        "user_degrees.csv",
        "item_degrees.csv",
    ]
    for filename in required:
        src = in_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")

    # Keep the training graph unchanged. Only the evaluation test set is filtered.
    for filename in required:
        if filename == "test.csv":
            continue
        shutil.copy2(in_dir / filename, out_dir / filename)

    train = pd.read_csv(in_dir / "train.csv")
    test = pd.read_csv(in_dir / "test.csv")
    item_degrees = pd.read_csv(in_dir / "item_degrees.csv")

    train_items = set(train["item_idx"].unique())
    warm_test = test[test["item_idx"].isin(train_items)].copy()
    cold_test = test[~test["item_idx"].isin(train_items)].copy()

    warm_test.to_csv(out_dir / "test.csv", index=False)
    cold_test.to_csv(out_dir / "cold_start_test.csv", index=False)

    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    warm_groups = warm_test[["user_idx", "item_idx"]].copy()
    warm_groups["item_group"] = warm_groups["item_idx"].map(item_group_map).fillna(
        "cold_start"
    )
    test_composition = (
        warm_groups["item_group"]
        .value_counts()
        .rename_axis("item_group")
        .reset_index(name="count")
    )
    test_composition["share"] = test_composition["count"] / len(warm_groups)
    test_composition.to_csv(out_dir / "test_item_composition.csv", index=False)

    source_summary_path = in_dir / "dataset_summary.csv"
    summary = {}
    if source_summary_path.exists():
        source_summary = pd.read_csv(source_summary_path).iloc[0].to_dict()
        summary.update({f"source_{k}": v for k, v in source_summary.items()})

    summary.update(
        {
            "dataset": "Amazon All Beauty 3-core warm-start test",
            "train_interactions": len(train),
            "validation_interactions": len(pd.read_csv(in_dir / "val.csv")),
            "original_test_interactions": len(test),
            "warm_test_interactions": len(warm_test),
            "cold_start_test_interactions_removed": len(cold_test),
            "warm_test_share_retained": len(warm_test) / len(test) if len(test) else 0.0,
            "cold_start_test_share_removed": len(cold_test) / len(test) if len(test) else 0.0,
            "warm_test_users": warm_test["user_idx"].nunique(),
            "warm_test_items": warm_test["item_idx"].nunique(),
            "warm_test_head_share": float(
                (warm_groups["item_group"] == "head").mean()
            ),
            "warm_test_medium_share": float(
                (warm_groups["item_group"] == "medium").mean()
            ),
            "warm_test_tail_share": float(
                (warm_groups["item_group"] == "tail").mean()
            ),
        }
    )

    pd.DataFrame([summary]).to_csv(out_dir / "dataset_summary.csv", index=False)

    print("\nAmazon Beauty warm-start evaluation view complete")
    print("=" * 55)
    print(f"Input directory: {in_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Train interactions: {len(train)}")
    print(f"Original test interactions: {len(test)}")
    print(f"Warm-start test interactions retained: {len(warm_test)}")
    print(f"Cold-start test interactions removed: {len(cold_test)}")
    print(
        "Warm-start retained share: "
        f"{len(warm_test) / len(test):.4f}" if len(test) else "nan"
    )
    print("\nWarm-start test item composition:")
    print(test_composition.to_string(index=False))


if __name__ == "__main__":
    main()
