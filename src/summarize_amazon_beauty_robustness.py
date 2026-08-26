from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "amazon_beauty_3core_warm"

GROUP_ORDER = {"tail": 0, "medium": 1, "head": 2}
MODEL_ORDER = {"Popularity": 0, "ItemCF": 1, "LightGCN": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise Amazon Beauty external robustness check results."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    return parser.parse_args()


def format_mean_std(mean: float, std: float | None) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_head_tail_gap(
    per_user: pd.DataFrame,
    model: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    df = per_user[(per_user["model"] == model) & (per_user["K"] == 20)].copy()
    if model == "LightGCN":
        df = (
            df.groupby(["user_idx", "item_idx", "item_group", "K"], as_index=False)
            .agg(Recall=("Recall", "mean"))
        )

    tail = df[df["item_group"] == "tail"]["Recall"].to_numpy()
    head = df[df["item_group"] == "head"]["Recall"].to_numpy()

    observed = float(head.mean() - tail.mean())
    boot = []
    for _ in range(n_resamples):
        tail_sample = rng.choice(tail, size=len(tail), replace=True)
        head_sample = rng.choice(head, size=len(head), replace=True)
        boot.append(head_sample.mean() - tail_sample.mean())
    lo, hi = percentile_ci(np.asarray(boot))
    return observed, lo, hi


def bootstrap_lightgcn_itemcf_difference(
    baseline_per_user: pd.DataFrame,
    lightgcn_per_user: pd.DataFrame,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    itemcf = baseline_per_user[
        (baseline_per_user["model"] == "ItemCF") & (baseline_per_user["K"] == 20)
    ][["user_idx", "Recall"]].rename(columns={"Recall": "itemcf_recall"})
    light = (
        lightgcn_per_user[lightgcn_per_user["K"] == 20]
        .groupby("user_idx", as_index=False)
        .agg(lightgcn_recall=("Recall", "mean"))
    )
    paired = itemcf.merge(light, on="user_idx", how="inner")
    diffs = paired["lightgcn_recall"].to_numpy() - paired["itemcf_recall"].to_numpy()
    observed = float(diffs.mean())
    boot = []
    for _ in range(n_resamples):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot.append(sample.mean())
    lo, hi = percentile_ci(np.asarray(boot))
    return observed, lo, hi


def compute_degree_norm_correlations(
    data_dir: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")
    user_degrees = pd.read_csv(data_dir / "user_degrees.csv")

    rows = []
    norm_frames = []

    for item_path in sorted(results_dir.glob("amazon_lightgcn_item_embeddings_seed_*.npy")):
        seed = int(item_path.stem.split("_")[-1])
        user_path = results_dir / f"amazon_lightgcn_user_embeddings_seed_{seed}.npy"
        if not user_path.exists():
            continue

        item_emb = np.load(item_path)
        user_emb = np.load(user_path)

        item_norms = pd.DataFrame(
            {"item_idx": np.arange(len(item_emb)), "embedding_norm": np.linalg.norm(item_emb, axis=1)}
        ).merge(item_degrees, on="item_idx", how="inner")
        user_norms = pd.DataFrame(
            {"user_idx": np.arange(len(user_emb)), "embedding_norm": np.linalg.norm(user_emb, axis=1)}
        ).merge(user_degrees, on="user_idx", how="inner")

        item_corr = item_norms[["degree", "embedding_norm"]].corr().iloc[0, 1]
        user_corr = user_norms[["degree", "embedding_norm"]].corr().iloc[0, 1]
        rows.append(
            {
                "seed": seed,
                "user_degree_norm_correlation": user_corr,
                "item_degree_norm_correlation": item_corr,
            }
        )

        item_norms["seed"] = seed
        user_norms["seed"] = seed
        item_norms["node_type"] = "item"
        user_norms["node_type"] = "user"
        norm_frames.append(item_norms)
        norm_frames.append(user_norms)

    corr_by_seed = pd.DataFrame(rows).sort_values("seed")
    norm_df = pd.concat(norm_frames, ignore_index=True)
    return corr_by_seed, norm_df


def summarize_models(
    baseline_per_user: pd.DataFrame,
    lightgcn_per_user: pd.DataFrame,
    corr_by_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    baseline_overall = (
        baseline_per_user[baseline_per_user["K"] == 20]
        .groupby("model", as_index=False)
        .agg(Recall20=("Recall", "mean"), NDCG20=("NDCG", "mean"))
    )
    baseline_groups = (
        baseline_per_user[baseline_per_user["K"] == 20]
        .groupby(["model", "item_group"], as_index=False)
        .agg(group_recall=("Recall", "mean"), n=("Recall", "count"))
    )

    for _, row in baseline_overall.iterrows():
        model = row["model"]
        groups = baseline_groups[baseline_groups["model"] == model].set_index("item_group")
        tail = float(groups.loc["tail", "group_recall"])
        medium = float(groups.loc["medium", "group_recall"])
        head = float(groups.loc["head", "group_recall"])
        rows.append(
            {
                "dataset": "Amazon Beauty",
                "model": model,
                "Recall@20": f"{row['Recall20']:.4f}",
                "NDCG@20": f"{row['NDCG20']:.4f}",
                "Tail Recall@20": f"{tail:.4f}",
                "Medium Recall@20": f"{medium:.4f}",
                "Head Recall@20": f"{head:.4f}",
                "Head-tail gap": f"{head - tail:.4f}",
                "Item degree-norm corr.": "",
            }
        )

    light_overall_seed = (
        lightgcn_per_user[lightgcn_per_user["K"] == 20]
        .groupby("seed", as_index=False)
        .agg(Recall20=("Recall", "mean"), NDCG20=("NDCG", "mean"))
    )
    light_groups_seed = (
        lightgcn_per_user[lightgcn_per_user["K"] == 20]
        .groupby(["seed", "item_group"], as_index=False)
        .agg(group_recall=("Recall", "mean"), n=("Recall", "count"))
    )
    pivot = light_groups_seed.pivot(index="seed", columns="item_group", values="group_recall")

    rows.append(
        {
            "dataset": "Amazon Beauty",
            "model": "LightGCN",
            "Recall@20": format_mean_std(
                light_overall_seed["Recall20"].mean(),
                light_overall_seed["Recall20"].std(ddof=1),
            ),
            "NDCG@20": format_mean_std(
                light_overall_seed["NDCG20"].mean(),
                light_overall_seed["NDCG20"].std(ddof=1),
            ),
            "Tail Recall@20": format_mean_std(
                pivot["tail"].mean(), pivot["tail"].std(ddof=1)
            ),
            "Medium Recall@20": format_mean_std(
                pivot["medium"].mean(), pivot["medium"].std(ddof=1)
            ),
            "Head Recall@20": format_mean_std(
                pivot["head"].mean(), pivot["head"].std(ddof=1)
            ),
            "Head-tail gap": format_mean_std(
                (pivot["head"] - pivot["tail"]).mean(),
                (pivot["head"] - pivot["tail"]).std(ddof=1),
            ),
            "Item degree-norm corr.": format_mean_std(
                corr_by_seed["item_degree_norm_correlation"].mean(),
                corr_by_seed["item_degree_norm_correlation"].std(ddof=1),
            ),
        }
    )

    summary = pd.DataFrame(rows)
    return (
        summary.assign(model_order=summary["model"].map(MODEL_ORDER))
        .sort_values("model_order")
        .drop(columns=["model_order"])
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.bootstrap_seed)

    baseline_per_user = pd.read_csv(args.results_dir / "amazon_baseline_per_user_metrics.csv")
    lightgcn_per_user = pd.read_csv(
        args.results_dir / "amazon_lightgcn_per_user_metrics_all_seeds.csv"
    )

    corr_by_seed, norm_df = compute_degree_norm_correlations(args.data_dir, args.results_dir)
    summary = summarize_models(baseline_per_user, lightgcn_per_user, corr_by_seed)

    head_tail_rows = []
    combined_per_user = pd.concat([baseline_per_user, lightgcn_per_user], ignore_index=True)
    for model in ["Popularity", "ItemCF", "LightGCN"]:
        observed, lo, hi = bootstrap_head_tail_gap(
            combined_per_user, model, args.bootstrap_resamples, rng
        )
        head_tail_rows.append(
            {
                "model": model,
                "head_tail_gap": observed,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    diff, diff_lo, diff_hi = bootstrap_lightgcn_itemcf_difference(
        baseline_per_user, lightgcn_per_user, args.bootstrap_resamples, rng
    )

    corr_by_seed.to_csv(args.results_dir / "amazon_degree_norm_correlation_by_seed.csv", index=False)
    norm_df.to_csv(args.results_dir / "amazon_embedding_norms.csv", index=False)
    summary.to_csv(args.results_dir / "amazon_external_robustness_summary.csv", index=False)
    pd.DataFrame(head_tail_rows).to_csv(
        args.results_dir / "amazon_head_tail_gap_bootstrap_ci.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "comparison": "LightGCN - ItemCF",
                "Recall@20_difference": diff,
                "ci_low": diff_lo,
                "ci_high": diff_hi,
            }
        ]
    ).to_csv(args.results_dir / "amazon_lightgcn_itemcf_bootstrap_ci.csv", index=False)

    print("\nAmazon Beauty external robustness summary")
    print("=" * 50)
    print(summary.to_string(index=False))
    print("\nDegree-norm correlations by seed:")
    print(corr_by_seed.to_string(index=False))
    print("\nHead-tail gap bootstrap CIs:")
    print(pd.DataFrame(head_tail_rows).to_string(index=False))
    print("\nLightGCN - ItemCF paired bootstrap difference:")
    print(f"Recall@20 diff = {diff:.4f} [{diff_lo:.4f}, {diff_hi:.4f}]")
    print(f"\nSaved summary files to: {args.results_dir}")


if __name__ == "__main__":
    main()
