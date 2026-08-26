from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 43, 44]
K = 20
GROUP_RENAME = {
    "low": "tail",
    "medium": "medium",
    "high": "head",
    "tail": "tail",
    "head": "head",
}
GROUP_ORDER = {
    "tail": 0,
    "medium": 1,
    "head": 2,
    "unknown": 3,
    "cold_start": 4,
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    results_dir: Path
    user_embedding_template: str
    item_embedding_template: str
    rank_training_items_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc LightGCN diagnostic: compare usual dot-product scoring with "
            "cosine scoring after unit-normalising already trained embeddings."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/Users/jinsixiong/Desktop/硕士毕业论文/dissertation-recommender"),
        help="Root directory of the dissertation-recommender repository.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/cosine_normalized_lightgcn"),
        help="Directory where diagnostic CSV and LaTeX outputs will be written.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="LightGCN optimisation seeds to evaluate.",
    )
    return parser.parse_args()


def build_user_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_idx")["item_idx"].apply(set).to_dict()


def recall_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    return float(true_item in ranked_items[:k])


def ndcg_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    top_k = ranked_items[:k]
    if true_item not in top_k:
        return 0.0
    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)


def normalise_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    finite_count = int(np.isfinite(scores).sum())
    top_count = min(k, finite_count)
    if top_count == 0:
        return []
    ranked = np.argpartition(-scores, top_count - 1)[:top_count]
    ranked = ranked[np.argsort(-scores[ranked])]
    return ranked.tolist()


def load_item_group_map(data_dir: Path) -> dict[int, str]:
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")
    item_degrees["item_group"] = (
        item_degrees["degree_group"]
        .map(GROUP_RENAME)
        .fillna(item_degrees["degree_group"])
    )
    return dict(zip(item_degrees["item_idx"], item_degrees["item_group"]))


def evaluate_seed(
    dataset: DatasetConfig,
    seed: int,
    scoring: str,
) -> pd.DataFrame:
    train = pd.read_csv(dataset.data_dir / "train.csv")
    test = pd.read_csv(dataset.data_dir / "test.csv")
    train_user_items = build_user_items(train)
    item_group_map = load_item_group_map(dataset.data_dir)

    user_embeddings = np.load(
        dataset.results_dir / dataset.user_embedding_template.format(seed=seed)
    )
    item_embeddings = np.load(
        dataset.results_dir / dataset.item_embedding_template.format(seed=seed)
    )

    if scoring == "cosine_normalized":
        user_embeddings = normalise_rows(user_embeddings)
        item_embeddings = normalise_rows(item_embeddings)
    elif scoring != "dot_product":
        raise ValueError(f"Unknown scoring mode: {scoring}")

    if dataset.rank_training_items_only:
        candidate_mask = np.zeros(item_embeddings.shape[0], dtype=bool)
        candidate_mask[train["item_idx"].unique().astype(int)] = True
    else:
        candidate_mask = np.ones(item_embeddings.shape[0], dtype=bool)

    item_embeddings_t = item_embeddings.T
    records = []

    for _, row in test.iterrows():
        user = int(row["user_idx"])
        true_item = int(row["item_idx"])

        scores = user_embeddings[user] @ item_embeddings_t
        scores = scores.copy()
        scores[~candidate_mask] = -np.inf

        seen_items = train_user_items.get(user, set())
        if seen_items:
            scores[list(seen_items)] = -np.inf

        ranked_items = top_k_from_scores(scores, K)
        item_group = item_group_map.get(true_item, "unknown")

        records.append(
            {
                "dataset": dataset.name,
                "seed": seed,
                "scoring": scoring,
                "user_idx": user,
                "item_idx": true_item,
                "item_group": item_group,
                "K": K,
                "Recall": recall_at_k(ranked_items, true_item, K),
                "NDCG": ndcg_at_k(ranked_items, true_item, K),
            }
        )

    return pd.DataFrame(records)


def summarise_overall(per_user: pd.DataFrame) -> pd.DataFrame:
    return (
        per_user.groupby(["dataset", "scoring", "seed", "K"], as_index=False)
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n=("Recall", "count"),
        )
        .sort_values(["dataset", "scoring", "seed"])
    )


def summarise_item_groups(per_user: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_user.groupby(
            ["dataset", "scoring", "seed", "item_group", "K"],
            as_index=False,
        )
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n=("Recall", "count"),
        )
    )
    grouped["group_order"] = grouped["item_group"].map(GROUP_ORDER).fillna(99)
    return grouped.sort_values(
        ["dataset", "scoring", "seed", "group_order", "item_group"]
    ).drop(columns=["group_order"])


def summarise_across_seeds(
    per_seed: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    return (
        per_seed.groupby(group_cols, as_index=False)
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_mean=("n", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(group_cols)
    )


def summarise_head_tail_gap(item_group_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, scoring, seed), df in item_group_by_seed.groupby(
        ["dataset", "scoring", "seed"]
    ):
        recall_by_group = dict(zip(df["item_group"], df["Recall"]))
        n_by_group = dict(zip(df["item_group"], df["n"]))
        if "head" not in recall_by_group or "tail" not in recall_by_group:
            continue
        rows.append(
            {
                "dataset": dataset,
                "scoring": scoring,
                "seed": seed,
                "K": K,
                "head_tail_gap": recall_by_group["head"]
                - recall_by_group["tail"],
                "head_recall": recall_by_group["head"],
                "tail_recall": recall_by_group["tail"],
                "head_n": n_by_group["head"],
                "tail_n": n_by_group["tail"],
            }
        )
    gap_by_seed = pd.DataFrame(rows)
    gap_summary = (
        gap_by_seed.groupby(["dataset", "scoring", "K"], as_index=False)
        .agg(
            head_tail_gap_mean=("head_tail_gap", "mean"),
            head_tail_gap_std=("head_tail_gap", "std"),
            head_recall_mean=("head_recall", "mean"),
            tail_recall_mean=("tail_recall", "mean"),
            head_n=("head_n", "mean"),
            tail_n=("tail_n", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "scoring"])
    )
    return gap_by_seed, gap_summary


def make_comparison_table(
    overall_summary: pd.DataFrame,
    item_group_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset, df in overall_summary.groupby("dataset"):
        recall = {
            row.scoring: row.recall_mean
            for row in df.itertuples(index=False)
        }
        rows.append(
            {
                "dataset": dataset,
                "quantity": "Overall Recall@20",
                "dot_product": recall.get("dot_product", np.nan),
                "cosine_normalized": recall.get("cosine_normalized", np.nan),
            }
        )

    for (dataset, group), df in item_group_summary.groupby(
        ["dataset", "item_group"]
    ):
        if group not in {"tail", "medium", "head"}:
            continue
        recall = {
            row.scoring: row.recall_mean
            for row in df.itertuples(index=False)
        }
        rows.append(
            {
                "dataset": dataset,
                "quantity": f"{group.title()} Recall@20",
                "dot_product": recall.get("dot_product", np.nan),
                "cosine_normalized": recall.get("cosine_normalized", np.nan),
            }
        )

    for dataset, df in gap_summary.groupby("dataset"):
        gap = {
            row.scoring: row.head_tail_gap_mean
            for row in df.itertuples(index=False)
        }
        rows.append(
            {
                "dataset": dataset,
                "quantity": "Head-tail Recall@20 gap",
                "dot_product": gap.get("dot_product", np.nan),
                "cosine_normalized": gap.get("cosine_normalized", np.nan),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["cosine_minus_dot"] = (
        comparison["cosine_normalized"] - comparison["dot_product"]
    )
    return comparison.sort_values(["dataset", "quantity"])


def format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_latex_table(
    overall_summary: pd.DataFrame,
    item_group_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    for dataset in sorted(overall_summary["dataset"].unique()):
        dataset_overall = overall_summary[
            overall_summary["dataset"] == dataset
        ]
        dataset_groups = item_group_summary[
            (item_group_summary["dataset"] == dataset)
            & (item_group_summary["item_group"].isin(["tail", "medium", "head"]))
        ]
        dataset_gaps = gap_summary[gap_summary["dataset"] == dataset]

        for scoring in ["dot_product", "cosine_normalized"]:
            overall_row = dataset_overall[
                dataset_overall["scoring"] == scoring
            ].iloc[0]
            group_rows = {
                row.item_group: row
                for row in dataset_groups[
                    dataset_groups["scoring"] == scoring
                ].itertuples(index=False)
            }
            gap_row = dataset_gaps[dataset_gaps["scoring"] == scoring].iloc[0]
            rows.append(
                [
                    dataset,
                    "Dot product"
                    if scoring == "dot_product"
                    else "Cosine-normalised",
                    format_mean_std(overall_row.recall_mean, overall_row.recall_std),
                    format_mean_std(
                        group_rows["tail"].recall_mean,
                        group_rows["tail"].recall_std,
                    ),
                    format_mean_std(
                        group_rows["medium"].recall_mean,
                        group_rows["medium"].recall_std,
                    ),
                    format_mean_std(
                        group_rows["head"].recall_mean,
                        group_rows["head"].recall_std,
                    ),
                    format_mean_std(
                        gap_row.head_tail_gap_mean,
                        gap_row.head_tail_gap_std,
                    ),
                ]
            )

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Dataset & Scoring & Overall Recall@20 & Tail Recall@20 & Medium Recall@20 & Head Recall@20 & Head--tail gap \\\\",
        "\\hline",
    ]
    for row in rows:
        latex_lines.append(" & ".join(row) + " \\\\")
    latex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "}",
            "\\caption{Post-hoc cosine-normalisation diagnostic for already trained LightGCN embeddings. Dot-product scoring uses the original learned embeddings, while cosine-normalised scoring first rescales each user and item embedding to unit norm. Values are mean $\\pm$ standard deviation across optimisation seeds 42, 43, and 44.}",
            "\\label{tab:cosine_normalized_lightgcn_diagnostic}",
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(latex_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetConfig(
            name="MovieLens-1M",
            data_dir=repo_root / "data" / "processed",
            results_dir=repo_root / "results",
            user_embedding_template="lightgcn_user_embeddings_seed_{seed}.npy",
            item_embedding_template="lightgcn_item_embeddings_seed_{seed}.npy",
            rank_training_items_only=False,
        ),
        DatasetConfig(
            name="Amazon Beauty warm-start",
            data_dir=repo_root / "data" / "processed" / "amazon_beauty_3core_warm",
            results_dir=repo_root / "results" / "amazon_beauty_3core_warm",
            user_embedding_template="amazon_lightgcn_user_embeddings_seed_{seed}.npy",
            item_embedding_template="amazon_lightgcn_item_embeddings_seed_{seed}.npy",
            rank_training_items_only=True,
        ),
    ]

    all_per_user = []
    for dataset in datasets:
        for seed in args.seeds:
            for scoring in ["dot_product", "cosine_normalized"]:
                print(
                    f"Evaluating {dataset.name} | seed={seed} | scoring={scoring}"
                )
                all_per_user.append(evaluate_seed(dataset, seed, scoring))

    per_user = pd.concat(all_per_user, ignore_index=True)
    overall_by_seed = summarise_overall(per_user)
    item_group_by_seed = summarise_item_groups(per_user)
    overall_summary = summarise_across_seeds(
        overall_by_seed,
        ["dataset", "scoring", "K"],
    )
    item_group_summary = summarise_across_seeds(
        item_group_by_seed,
        ["dataset", "scoring", "item_group", "K"],
    )
    gap_by_seed, gap_summary = summarise_head_tail_gap(item_group_by_seed)
    comparison = make_comparison_table(
        overall_summary,
        item_group_summary,
        gap_summary,
    )

    per_user.to_csv(out_dir / "cosine_diagnostic_per_user_metrics.csv", index=False)
    overall_by_seed.to_csv(
        out_dir / "cosine_diagnostic_overall_by_seed.csv",
        index=False,
    )
    item_group_by_seed.to_csv(
        out_dir / "cosine_diagnostic_item_group_by_seed.csv",
        index=False,
    )
    overall_summary.to_csv(
        out_dir / "cosine_diagnostic_overall_summary.csv",
        index=False,
    )
    item_group_summary.to_csv(
        out_dir / "cosine_diagnostic_item_group_summary.csv",
        index=False,
    )
    gap_by_seed.to_csv(out_dir / "cosine_diagnostic_head_tail_gap_by_seed.csv", index=False)
    gap_summary.to_csv(out_dir / "cosine_diagnostic_head_tail_gap_summary.csv", index=False)
    comparison.to_csv(out_dir / "cosine_diagnostic_comparison_table.csv", index=False)
    write_latex_table(
        overall_summary,
        item_group_summary,
        gap_summary,
        out_dir / "cosine_diagnostic_table.tex",
    )

    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))
    print("\nItem-group summary:")
    print(item_group_summary.to_string(index=False))
    print("\nHead-tail gap summary:")
    print(gap_summary.to_string(index=False))
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
