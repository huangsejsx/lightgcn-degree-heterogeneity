from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

from baseline_itemcf import build_item_similarity, recommend as recommend_itemcf


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

TOP_K = [10, 20]
MODEL_ORDER = {
    "Popularity": 0,
    "ItemCF": 1,
    "LightGCN": 2,
}
GROUP_TYPE_ORDER = {
    "user_degree": 0,
    "item_degree": 1,
}
GROUP_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "unknown": 3,
}


def build_user_items(df: pd.DataFrame) -> dict[int, set[int]]:
    return (
        df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def recall_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    return float(true_item in ranked_items[:k])


def ndcg_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    top_k = ranked_items[:k]

    if true_item not in top_k:
        return 0.0

    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    finite_count = int(np.isfinite(scores).sum())
    top_count = min(k, finite_count)

    if top_count == 0:
        return []

    ranked = np.argpartition(-scores, top_count - 1)[:top_count]
    ranked = ranked[np.argsort(-scores[ranked])]
    return ranked.tolist()


def append_group_records(
    records: list[dict],
    model: str,
    run: str,
    user_group: str,
    item_group: str,
    ranked_items: list[int],
    true_item: int,
) -> None:
    for k in TOP_K:
        recall = recall_at_k(ranked_items, true_item, k)
        ndcg = ndcg_at_k(ranked_items, true_item, k)

        records.append(
            {
                "model": model,
                "run": run,
                "group_type": "user_degree",
                "group": user_group,
                "K": k,
                "Recall": recall,
                "NDCG": ndcg,
            }
        )

        records.append(
            {
                "model": model,
                "run": run,
                "group_type": "item_degree",
                "group": item_group,
                "K": k,
                "Recall": recall,
                "NDCG": ndcg,
            }
        )


def evaluate_popularity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_user_items: dict[int, set[int]],
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
) -> pd.DataFrame:
    ranked_items = (
        train.groupby("item_idx")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    records = []

    for row in test.itertuples(index=False):
        user = int(row.user_idx)
        true_item = int(row.item_idx)
        seen_items = train_user_items.get(user, set())

        recommendations = [
            item
            for item in ranked_items
            if item not in seen_items
        ][: max(TOP_K)]

        append_group_records(
            records=records,
            model="Popularity",
            run="deterministic",
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
            true_item=true_item,
        )

    return summarise_test_cases(pd.DataFrame(records))


def evaluate_itemcf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_user_items: dict[int, set[int]],
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
) -> pd.DataFrame:
    print("Building ItemCF similarity matrix...")
    item_sim = build_item_similarity(train)
    all_items = train["item_idx"].value_counts().index.tolist()

    records = []

    for row in test.itertuples(index=False):
        user = int(row.user_idx)
        true_item = int(row.item_idx)

        recommendations = recommend_itemcf(
            user=user,
            train_user_items=train_user_items,
            item_sim=item_sim,
            all_items=all_items,
            topn=max(TOP_K),
        )

        append_group_records(
            records=records,
            model="ItemCF",
            run="deterministic",
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
            true_item=true_item,
        )

    return summarise_test_cases(pd.DataFrame(records))


def discover_lightgcn_seeds() -> list[int]:
    seeds = []

    for path in RESULTS_DIR.glob("lightgcn_user_embeddings_seed_*.npy"):
        match = re.search(r"seed_(\d+)\.npy$", path.name)

        if not match:
            continue

        seed = int(match.group(1))
        item_path = RESULTS_DIR / f"lightgcn_item_embeddings_seed_{seed}.npy"

        if item_path.exists():
            seeds.append(seed)

    return sorted(seeds)


def evaluate_lightgcn(
    seed: int,
    test: pd.DataFrame,
    train_user_items: dict[int, set[int]],
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
) -> pd.DataFrame:
    user_emb = np.load(
        RESULTS_DIR / f"lightgcn_user_embeddings_seed_{seed}.npy"
    )
    item_emb = np.load(
        RESULTS_DIR / f"lightgcn_item_embeddings_seed_{seed}.npy"
    )

    records = []
    max_k = max(TOP_K)

    for row in test.itertuples(index=False):
        user = int(row.user_idx)
        true_item = int(row.item_idx)

        scores = user_emb[user] @ item_emb.T
        seen_items = train_user_items.get(user, set())

        if seen_items:
            scores[list(seen_items)] = -np.inf

        recommendations = top_k_from_scores(scores, max_k)

        append_group_records(
            records=records,
            model="LightGCN",
            run=f"seed_{seed}",
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
            true_item=true_item,
        )

    return summarise_test_cases(pd.DataFrame(records))


def summarise_test_cases(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["model", "run", "group_type", "group", "K"],
            as_index=False,
        )
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n_test_cases=("Recall", "size"),
        )
    )


def summarise_runs(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["model", "group_type", "group", "K"],
            as_index=False,
        )
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_runs=("run", "nunique"),
            n_test_cases=("n_test_cases", "first"),
        )
    )

    return sort_group_metrics(summary)


def sort_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            model_order=df["model"].map(MODEL_ORDER).fillna(99),
            group_type_order=df["group_type"].map(GROUP_TYPE_ORDER).fillna(99),
            group_order=df["group"].map(GROUP_ORDER).fillna(99),
        )
        .sort_values(
            [
                "model_order",
                "group_type_order",
                "group_order",
                "K",
            ]
        )
        .drop(
            columns=[
                "model_order",
                "group_type_order",
                "group_order",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    user_degrees = pd.read_csv(DATA_DIR / "user_degrees.csv")
    item_degrees = pd.read_csv(DATA_DIR / "item_degrees.csv")

    train_user_items = build_user_items(train)
    user_group_map = dict(
        zip(user_degrees["user_idx"], user_degrees["degree_group"])
    )
    item_group_map = dict(
        zip(item_degrees["item_idx"], item_degrees["degree_group"])
    )

    all_results = [
        evaluate_popularity(
            train=train,
            test=test,
            train_user_items=train_user_items,
            user_group_map=user_group_map,
            item_group_map=item_group_map,
        ),
        evaluate_itemcf(
            train=train,
            test=test,
            train_user_items=train_user_items,
            user_group_map=user_group_map,
            item_group_map=item_group_map,
        ),
    ]

    lightgcn_seeds = discover_lightgcn_seeds()

    if not lightgcn_seeds:
        raise FileNotFoundError(
            "No LightGCN seed embedding files found in results/."
        )

    for seed in lightgcn_seeds:
        print(f"Evaluating LightGCN seed {seed}...")
        all_results.append(
            evaluate_lightgcn(
                seed=seed,
                test=test,
                train_user_items=train_user_items,
                user_group_map=user_group_map,
                item_group_map=item_group_map,
            )
        )

    by_run = sort_group_metrics(
        pd.concat(all_results, ignore_index=True)
    )
    summary = summarise_runs(by_run)

    by_run_path = RESULTS_DIR / "all_model_group_results_by_run.csv"
    summary_path = RESULTS_DIR / "all_model_group_summary.csv"
    user_summary_path = RESULTS_DIR / "all_model_user_group_summary.csv"
    item_summary_path = RESULTS_DIR / "all_model_item_group_summary.csv"

    by_run.to_csv(by_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary[summary["group_type"] == "user_degree"].to_csv(
        user_summary_path,
        index=False,
    )
    summary[summary["group_type"] == "item_degree"].to_csv(
        item_summary_path,
        index=False,
    )

    print("\nGrouped performance summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {by_run_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {user_summary_path}")
    print(f"Saved: {item_summary_path}")


if __name__ == "__main__":
    main()
