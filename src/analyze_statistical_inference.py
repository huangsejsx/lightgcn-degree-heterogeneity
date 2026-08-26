from argparse import ArgumentParser, Namespace
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
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

MODEL_ORDER = {
    "Popularity": 0,
    "ItemCF": 1,
    "LightGCN": 2,
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

    ranked_items = np.argpartition(-scores, top_count - 1)[:top_count]
    ranked_items = ranked_items[np.argsort(-scores[ranked_items])]
    return ranked_items.tolist()


def append_metric_records(
    records: list[dict],
    model: str,
    run: str,
    test_index: int,
    user: int,
    true_item: int,
    user_group: str,
    item_group: str,
    ranked_items: list[int],
) -> None:
    for k in TOP_K:
        records.append(
            {
                "model": model,
                "run": run,
                "test_index": test_index,
                "user_idx": user,
                "item_idx": true_item,
                "user_group": user_group,
                "item_group": item_group,
                "K": k,
                "Recall": recall_at_k(ranked_items, true_item, k),
                "NDCG": ndcg_at_k(ranked_items, true_item, k),
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

    for test_index, row in enumerate(test.itertuples(index=False)):
        user = int(row.user_idx)
        true_item = int(row.item_idx)
        seen_items = train_user_items.get(user, set())
        recommendations = [
            item
            for item in ranked_items
            if item not in seen_items
        ][: max(TOP_K)]

        append_metric_records(
            records=records,
            model="Popularity",
            run="deterministic",
            test_index=test_index,
            user=user,
            true_item=true_item,
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
        )

    return pd.DataFrame(records)


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

    for test_index, row in enumerate(test.itertuples(index=False)):
        user = int(row.user_idx)
        true_item = int(row.item_idx)
        recommendations = recommend_itemcf(
            user=user,
            train_user_items=train_user_items,
            item_sim=item_sim,
            all_items=all_items,
            topn=max(TOP_K),
        )

        append_metric_records(
            records=records,
            model="ItemCF",
            run="deterministic",
            test_index=test_index,
            user=user,
            true_item=true_item,
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
        )

    return pd.DataFrame(records)


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

    for test_index, row in enumerate(test.itertuples(index=False)):
        user = int(row.user_idx)
        true_item = int(row.item_idx)
        scores = user_emb[user] @ item_emb.T
        seen_items = train_user_items.get(user, set())

        if seen_items:
            scores[list(seen_items)] = -np.inf

        recommendations = top_k_from_scores(scores, max_k)

        append_metric_records(
            records=records,
            model="LightGCN",
            run=f"seed_{seed}",
            test_index=test_index,
            user=user,
            true_item=true_item,
            user_group=user_group_map.get(user, "unknown"),
            item_group=item_group_map.get(true_item, "unknown"),
            ranked_items=recommendations,
        )

    return pd.DataFrame(records)


def generate_per_test_metrics() -> pd.DataFrame:
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

    results = [
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
            "No LightGCN seed embeddings found under results/."
        )

    for seed in lightgcn_seeds:
        print(f"Evaluating LightGCN seed {seed}...")
        results.append(
            evaluate_lightgcn(
                seed=seed,
                test=test,
                train_user_items=train_user_items,
                user_group_map=user_group_map,
                item_group_map=item_group_map,
            )
        )

    per_test = pd.concat(results, ignore_index=True)
    return sort_metrics(per_test)


def sort_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            model_order=df["model"].map(MODEL_ORDER).fillna(99),
            group_order=df.get("item_group", pd.Series(index=df.index))
            .map(GROUP_ORDER)
            .fillna(99),
        )
        .sort_values(
            [
                "model_order",
                "run",
                "K",
                "test_index",
            ]
        )
        .drop(columns=["model_order", "group_order"])
        .reset_index(drop=True)
    )


def make_model_average(per_test: pd.DataFrame) -> pd.DataFrame:
    return (
        per_test.groupby(
            [
                "model",
                "test_index",
                "user_idx",
                "item_idx",
                "user_group",
                "item_group",
                "K",
            ],
            as_index=False,
        )
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n_runs=("run", "nunique"),
        )
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    n = len(values)
    estimates = np.empty(samples)

    for sample in range(samples):
        idx = rng.integers(0, n, size=n)
        estimates[sample] = float(values[idx].mean())

    return (
        float(values.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def bootstrap_diff_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float, float]:
    n = len(differences)
    estimates = np.empty(samples)

    for sample in range(samples):
        idx = rng.integers(0, n, size=n)
        estimates[sample] = float(differences[idx].mean())

    p_lower = float(np.mean(estimates <= 0))
    p_upper = float(np.mean(estimates >= 0))
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))

    return (
        float(differences.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
        p_two_sided,
    )


def overall_bootstrap_ci(
    model_average: pd.DataFrame,
    samples: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for (model, k), df in model_average.groupby(["model", "K"]):
        for metric in ["Recall", "NDCG"]:
            mean, lower, upper = bootstrap_mean_ci(
                df[metric].to_numpy(dtype=float),
                rng=rng,
                samples=samples,
            )
            rows.append(
                {
                    "model": model,
                    "K": k,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "bootstrap_samples": samples,
                    "n_test_cases": len(df),
                }
            )

    return sort_summary(pd.DataFrame(rows))


def paired_model_comparisons(
    model_average: pd.DataFrame,
    samples: int,
) -> pd.DataFrame:
    comparisons = [
        ("LightGCN", "ItemCF"),
        ("LightGCN", "Popularity"),
        ("ItemCF", "Popularity"),
    ]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)

    for k, k_df in model_average.groupby("K"):
        for metric in ["Recall", "NDCG"]:
            pivot = k_df.pivot(
                index="test_index",
                columns="model",
                values=metric,
            )

            for model_a, model_b in comparisons:
                if model_a not in pivot or model_b not in pivot:
                    continue

                differences = (
                    pivot[model_a] - pivot[model_b]
                ).to_numpy(dtype=float)
                mean, lower, upper, p_value = bootstrap_diff_ci(
                    differences=differences,
                    rng=rng,
                    samples=samples,
                )
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "difference": f"{model_a} - {model_b}",
                        "K": k,
                        "metric": metric,
                        "mean_difference": mean,
                        "ci_lower_95": lower,
                        "ci_upper_95": upper,
                        "bootstrap_p_two_sided": p_value,
                        "bootstrap_samples": samples,
                        "n_paired_test_cases": len(differences),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["metric", "K", "model_a", "model_b"]
    )


def head_tail_gap_ci(
    model_average: pd.DataFrame,
    samples: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)

    grouped = model_average[
        model_average["item_group"].isin(["low", "high"])
    ]

    for (model, k), df in grouped.groupby(["model", "K"]):
        for metric in ["Recall", "NDCG"]:
            head_values = df[df["item_group"] == "high"][
                metric
            ].to_numpy(dtype=float)
            tail_values = df[df["item_group"] == "low"][
                metric
            ].to_numpy(dtype=float)

            if len(head_values) == 0 or len(tail_values) == 0:
                continue

            estimates = np.empty(samples)

            for sample in range(samples):
                head_idx = rng.integers(
                    0,
                    len(head_values),
                    size=len(head_values),
                )
                tail_idx = rng.integers(
                    0,
                    len(tail_values),
                    size=len(tail_values),
                )
                estimates[sample] = (
                    head_values[head_idx].mean()
                    - tail_values[tail_idx].mean()
                )

            rows.append(
                {
                    "model": model,
                    "K": k,
                    "metric": metric,
                    "head_mean": float(head_values.mean()),
                    "tail_mean": float(tail_values.mean()),
                    "head_tail_gap": (
                        float(head_values.mean())
                        - float(tail_values.mean())
                    ),
                    "ci_lower_95": float(
                        np.percentile(estimates, 2.5)
                    ),
                    "ci_upper_95": float(
                        np.percentile(estimates, 97.5)
                    ),
                    "bootstrap_samples": samples,
                    "n_head_test_cases": len(head_values),
                    "n_tail_test_cases": len(tail_values),
                }
            )

    return sort_summary(pd.DataFrame(rows))


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(model_order=df["model"].map(MODEL_ORDER).fillna(99))
        .sort_values(["model_order", "K", "metric"])
        .drop(columns="model_order")
        .reset_index(drop=True)
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Compute bootstrap confidence intervals and paired "
            "bootstrap model comparisons."
        )
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--regenerate-per-test",
        action="store_true",
        help="Regenerate per-test metrics even when cached CSV exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_test_path = RESULTS_DIR / "statistical_per_test_metrics.csv"

    if per_test_path.exists() and not args.regenerate_per_test:
        print(f"Loading cached per-test metrics: {per_test_path}")
        per_test = pd.read_csv(per_test_path)
    else:
        per_test = generate_per_test_metrics()
        per_test.to_csv(per_test_path, index=False)
        print(f"Saved: {per_test_path}")

    model_average = make_model_average(per_test)
    model_average_path = (
        RESULTS_DIR / "statistical_model_average_per_test_metrics.csv"
    )
    model_average.to_csv(model_average_path, index=False)

    overall_ci = overall_bootstrap_ci(
        model_average=model_average,
        samples=args.bootstrap_samples,
    )
    paired = paired_model_comparisons(
        model_average=model_average,
        samples=args.bootstrap_samples,
    )
    head_tail = head_tail_gap_ci(
        model_average=model_average,
        samples=args.bootstrap_samples,
    )

    overall_ci_path = RESULTS_DIR / "statistical_overall_bootstrap_ci.csv"
    paired_path = RESULTS_DIR / "statistical_paired_model_comparisons.csv"
    head_tail_path = RESULTS_DIR / "statistical_head_tail_gap_ci.csv"

    overall_ci.to_csv(overall_ci_path, index=False)
    paired.to_csv(paired_path, index=False)
    head_tail.to_csv(head_tail_path, index=False)

    print("\nOverall bootstrap CIs:")
    print(overall_ci.to_string(index=False))

    print("\nPaired model comparisons:")
    print(paired.to_string(index=False))

    print("\nHead-tail gap bootstrap CIs:")
    print(head_tail.to_string(index=False))

    print(f"\nSaved: {model_average_path}")
    print(f"Saved: {overall_ci_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {head_tail_path}")


if __name__ == "__main__":
    main()
