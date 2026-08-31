from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DOT = "dot_product"
COSINE = "cosine_normalized"
K = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the change in the LightGCN head-tail Recall@20 gap when "
            "switching from dot-product scoring to cosine-normalised scoring."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/cosine_normalized_lightgcn/"
            "cosine_diagnostic_per_user_metrics.csv"
        ),
        help="Per-user paired dot/cosine diagnostic CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cosine_normalized_lightgcn"),
        help="Directory where summary CSV and LaTeX files are written.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=20_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Random seed for the bootstrap.",
    )
    return parser.parse_args()


def load_paired_recall(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "dataset",
        "seed",
        "user_idx",
        "item_idx",
        "item_group",
        "K",
        "scoring",
        "Recall",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        (df["K"] == K)
        & (df["item_group"].isin(["head", "tail"]))
        & (df["scoring"].isin([DOT, COSINE]))
    ].copy()

    wide = (
        df.pivot_table(
            index=["dataset", "seed", "user_idx", "item_idx", "item_group", "K"],
            columns="scoring",
            values="Recall",
            aggfunc="mean",
        )
        .reset_index()
        .dropna(subset=[DOT, COSINE])
    )
    return wide


def gap_and_delta(rows: pd.DataFrame) -> tuple[float, float, float]:
    head = rows[rows["item_group"] == "head"]
    tail = rows[rows["item_group"] == "tail"]
    if head.empty or tail.empty:
        raise ValueError("Both head and tail groups are required.")

    dot_gap = float(head[DOT].mean() - tail[DOT].mean())
    cosine_gap = float(head[COSINE].mean() - tail[COSINE].mean())
    delta_norm = dot_gap - cosine_gap
    return dot_gap, cosine_gap, delta_norm


def seed_level_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, seed), rows in wide.groupby(["dataset", "seed"]):
        dot_gap, cosine_gap, delta_norm = gap_and_delta(rows)
        records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "K": K,
                "head_n": int((rows["item_group"] == "head").sum()),
                "tail_n": int((rows["item_group"] == "tail").sum()),
                "dot_head_tail_gap": dot_gap,
                "cosine_head_tail_gap": cosine_gap,
                "delta_norm": delta_norm,
            }
        )
    return pd.DataFrame(records).sort_values(["dataset", "seed"])


def average_over_seeds(wide: pd.DataFrame) -> pd.DataFrame:
    return (
        wide.groupby(["dataset", "user_idx", "item_idx", "item_group", "K"], as_index=False)[
            [DOT, COSINE]
        ]
        .mean()
        .sort_values(["dataset", "item_group", "user_idx", "item_idx"])
    )


def bootstrap_delta(
    rows: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float]:
    head = rows[rows["item_group"] == "head"][[DOT, COSINE]].to_numpy()
    tail = rows[rows["item_group"] == "tail"][[DOT, COSINE]].to_numpy()
    if len(head) == 0 or len(tail) == 0:
        raise ValueError("Both head and tail groups are required.")

    dot_gap, cosine_gap, observed = gap_and_delta(rows)
    boot = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        head_sample = head[rng.integers(0, len(head), len(head))]
        tail_sample = tail[rng.integers(0, len(tail), len(tail))]
        sampled_dot_gap = head_sample[:, 0].mean() - tail_sample[:, 0].mean()
        sampled_cosine_gap = head_sample[:, 1].mean() - tail_sample[:, 1].mean()
        boot[idx] = sampled_dot_gap - sampled_cosine_gap

    return boot, dot_gap, cosine_gap, observed


def summarise_bootstrap(
    seed_averaged: pd.DataFrame,
    seed_deltas: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    records = []

    for dataset, rows in seed_averaged.groupby("dataset"):
        boot, dot_gap, cosine_gap, observed = bootstrap_delta(
            rows,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        dataset_seed_deltas = seed_deltas[seed_deltas["dataset"] == dataset]
        records.append(
            {
                "dataset": dataset,
                "K": K,
                "head_n": int((rows["item_group"] == "head").sum()),
                "tail_n": int((rows["item_group"] == "tail").sum()),
                "dot_head_tail_gap": dot_gap,
                "cosine_head_tail_gap": cosine_gap,
                "delta_norm": observed,
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "bootstrap_std": float(boot.std(ddof=1)),
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": bootstrap_seed,
                "seed_delta_mean": float(dataset_seed_deltas["delta_norm"].mean()),
                "seed_delta_std": float(dataset_seed_deltas["delta_norm"].std(ddof=1)),
                "n_optimization_seeds": int(dataset_seed_deltas["seed"].nunique()),
            }
        )

    return pd.DataFrame(records).sort_values("dataset")


def write_latex(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Dataset & Head $n$ & Tail $n$ & Dot gap & Cosine gap & $\\Delta_{\\mathrm{norm}}$ [95\\% CI] \\\\",
        "\\hline",
    ]
    for row in summary.itertuples(index=False):
        ci = f"{row.delta_norm:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"
        lines.append(
            f"{row.dataset} & {row.head_n:d} & {row.tail_n:d} & "
            f"{row.dot_head_tail_gap:.4f} & {row.cosine_head_tail_gap:.4f} & "
            f"{ci} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Bootstrap confidence intervals for the reduction in the LightGCN head--tail Recall@20 gap after cosine-normalised scoring. "
            "The bootstrap resamples held-out interactions within the head and tail item groups with replacement. Dot-product and cosine-normalised outcomes are kept paired for each held-out interaction after averaging predictions across optimisation seeds 42, 43, and 44.}",
            "\\label{tab:cosine_gap_bootstrap_ci}",
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wide = load_paired_recall(args.input)
    seed_deltas = seed_level_deltas(wide)
    seed_averaged = average_over_seeds(wide)
    summary = summarise_bootstrap(
        seed_averaged,
        seed_deltas,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    seed_deltas.to_csv(args.out_dir / "cosine_gap_delta_norm_by_seed.csv", index=False)
    summary.to_csv(args.out_dir / "cosine_gap_delta_norm_bootstrap_ci.csv", index=False)
    write_latex(summary, args.out_dir / "cosine_gap_delta_norm_bootstrap_ci.tex")

    print("\nCosine gap bootstrap summary:")
    print(summary.to_string(index=False))
    print("\nSeed-level Delta_norm values:")
    print(seed_deltas.to_string(index=False))
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
