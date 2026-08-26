from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "amazon_beauty_3core_warm"

KS = [10, 20]
TOP_SIM_ITEMS = 100
GROUP_ORDER = {"tail": 0, "medium": 1, "head": 2, "cold_start": 3, "unknown": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Popularity and ItemCF baselines on Amazon Beauty warm-start data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Processed Amazon Beauty warm-start data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for Amazon Beauty baseline results.",
    )
    parser.add_argument(
        "--top-sim-items",
        type=int,
        default=TOP_SIM_ITEMS,
        help="Maximum number of similar items retained per item for ItemCF.",
    )
    return parser.parse_args()


def recall_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    return 1.0 if ground_truth in recommended[:k] else 0.0


def ndcg_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    recommended_k = recommended[:k]
    if ground_truth not in recommended_k:
        return 0.0
    rank = recommended_k.index(ground_truth) + 1
    return 1.0 / math.log2(rank + 1)


def load_group_maps(data_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    user_degrees = pd.read_csv(data_dir / "user_degrees.csv")
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")
    user_group_map = dict(zip(user_degrees["user_idx"], user_degrees["degree_group"]))
    item_group_map = dict(zip(item_degrees["item_idx"], item_degrees["degree_group"]))
    return user_group_map, item_group_map


def evaluate_rankings(
    model_name: str,
    test: pd.DataFrame,
    recommend_fn,
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        user = int(row["user_idx"])
        true_item = int(row["item_idx"])
        recommended = recommend_fn(user)
        user_group = user_group_map.get(user, "unknown")
        item_group = item_group_map.get(true_item, "cold_start")

        for k in KS:
            rows.append(
                {
                    "dataset": "Amazon Beauty 3-core warm-start",
                    "model": model_name,
                    "run": "deterministic",
                    "user_idx": user,
                    "item_idx": true_item,
                    "user_group": user_group,
                    "item_group": item_group,
                    "K": k,
                    "Recall": recall_at_k(recommended, true_item, k),
                    "NDCG": ndcg_at_k(recommended, true_item, k),
                }
            )
    return pd.DataFrame(rows)


def build_popularity_recommender(train: pd.DataFrame):
    item_popularity = train.groupby("item_idx").size().sort_values(ascending=False)
    ranked_items = [int(item) for item in item_popularity.index.tolist()]
    train_user_items = (
        train.groupby("user_idx")["item_idx"].apply(lambda x: set(map(int, x))).to_dict()
    )

    def recommend(user: int) -> list[int]:
        seen_items = train_user_items.get(user, set())
        return [item for item in ranked_items if item not in seen_items][: max(KS)]

    return recommend


def build_item_similarity(train: pd.DataFrame, top_sim_items: int):
    user_items = (
        train.groupby("user_idx")["item_idx"].apply(lambda x: list(set(map(int, x)))).to_dict()
    )
    item_count = defaultdict(int)
    co_count = defaultdict(lambda: defaultdict(float))

    for items in user_items.values():
        for item in items:
            item_count[item] += 1
        if len(items) < 2:
            continue
        weight = 1.0 / math.log2(len(items) + 1)
        for i in items:
            for j in items:
                if i == j:
                    continue
                co_count[i][j] += weight

    item_sim = {}
    for i, related_items in co_count.items():
        sims = []
        for j, cij in related_items.items():
            denominator = math.sqrt(item_count[i] * item_count[j])
            if denominator == 0:
                continue
            sims.append((j, cij / denominator))
        item_sim[i] = sorted(sims, key=lambda x: x[1], reverse=True)[:top_sim_items]

    return item_sim


def build_itemcf_recommender(train: pd.DataFrame, top_sim_items: int):
    print("Building ItemCF similarity...")
    item_sim = build_item_similarity(train, top_sim_items=top_sim_items)
    print(f"ItemCF similarity built for {len(item_sim)} source items.")

    train_user_items = (
        train.groupby("user_idx")["item_idx"].apply(lambda x: set(map(int, x))).to_dict()
    )
    all_items = [int(item) for item in train["item_idx"].value_counts().index.tolist()]

    def recommend(user: int) -> list[int]:
        seen_items = train_user_items.get(user, set())
        scores = defaultdict(float)
        for item in seen_items:
            for sim_item, sim_score in item_sim.get(item, []):
                if sim_item not in seen_items:
                    scores[sim_item] += sim_score

        # Deterministic fallback so users with little similarity support still get a ranking.
        for item in all_items:
            if item not in seen_items and item not in scores:
                scores[item] = 0.0
            if len(scores) >= max(KS):
                break

        ranked_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, _ in ranked_items[: max(KS)]]

    return recommend


def summarise_overall(per_user: pd.DataFrame) -> pd.DataFrame:
    return (
        per_user.groupby(["dataset", "model", "run", "K"], as_index=False)
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
        .sort_values(["model", "K"])
    )


def summarise_grouped(per_user: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        per_user.groupby(["dataset", "model", "run", group_col, "K"], as_index=False)
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
    )
    return summary.assign(
        group_order=summary[group_col].map(GROUP_ORDER).fillna(99)
    ).sort_values(["model", "K", "group_order", group_col]).drop(columns=["group_order"])


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    user_group_map, item_group_map = load_group_maps(data_dir)

    print("\nAmazon Beauty baseline evaluation")
    print("=" * 45)
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Train interactions: {len(train)}")
    print(f"Warm-start test interactions: {len(test)}")
    print(f"Train users: {train['user_idx'].nunique()}")
    print(f"Train items: {train['item_idx'].nunique()}")

    popularity_recommend = build_popularity_recommender(train)
    itemcf_recommend = build_itemcf_recommender(train, args.top_sim_items)

    all_records = []
    all_records.append(
        evaluate_rankings(
            "Popularity", test, popularity_recommend, user_group_map, item_group_map
        )
    )
    all_records.append(
        evaluate_rankings("ItemCF", test, itemcf_recommend, user_group_map, item_group_map)
    )

    per_user = pd.concat(all_records, ignore_index=True)
    overall = summarise_overall(per_user)
    by_item_group = summarise_grouped(per_user, "item_group")
    by_user_group = summarise_grouped(per_user, "user_group")

    per_user.to_csv(results_dir / "amazon_baseline_per_user_metrics.csv", index=False)
    overall.to_csv(results_dir / "amazon_baseline_overall_metrics.csv", index=False)
    by_item_group.to_csv(results_dir / "amazon_baseline_item_group_metrics.csv", index=False)
    by_user_group.to_csv(results_dir / "amazon_baseline_user_group_metrics.csv", index=False)

    print("\nOverall metrics:")
    print(overall.to_string(index=False))
    print("\nItem-group metrics:")
    print(by_item_group.to_string(index=False))
    print(f"\nSaved baseline results to: {results_dir}")


if __name__ == "__main__":
    main()
