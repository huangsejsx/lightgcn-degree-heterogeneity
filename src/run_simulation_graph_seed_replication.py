from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LOCAL_ROOT_DIR = Path(__file__).resolve().parents[1]
DESKTOP_ROOT_DIR = Path("/Users/jinsixiong/Desktop/硕士毕业论文/dissertation-recommender")

if (LOCAL_ROOT_DIR / "src").exists():
    ROOT_DIR = LOCAL_ROOT_DIR
elif (DESKTOP_ROOT_DIR / "src").exists():
    ROOT_DIR = DESKTOP_ROOT_DIR
else:
    raise FileNotFoundError(
        "Could not locate dissertation-recommender. Run this script from the "
        "repository's src directory, or update DESKTOP_ROOT_DIR in the script."
    )

SRC_DIR = ROOT_DIR / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lightgcn import LightGCN, build_norm_adj, build_user_items, set_seed, train_one_epoch
from simulate_degree_heterogeneity import (
    SCENARIOS as SCENARIO_ALPHA,
    coefficient_of_variation,
    generate_synthetic_interactions,
    gini_coefficient,
)
from run_simulation_lightgcn import (
    EMBED_DIM,
    LEARNING_RATE,
    N_ITEMS,
    N_LAYERS,
    N_USERS,
    assign_item_groups,
    evaluate_grouped_recall,
    temporal_leave_one_out,
)


DEFAULT_GRAPH_SEEDS = [40, 41, 42, 43, 44]
DEFAULT_MODEL_SEED = 42
DEFAULT_EPOCHS = 100
SCENARIO_ORDER = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat synthetic LightGCN simulation results across graph-generation "
            "seeds while using one fixed LightGCN optimisation seed."
        )
    )
    parser.add_argument(
        "--graph-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_GRAPH_SEEDS,
        help="Synthetic graph-generation seeds to evaluate.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=DEFAULT_MODEL_SEED,
        help="Fixed LightGCN optimisation seed used for every graph realisation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of LightGCN training epochs per graph realisation.",
    )
    parser.add_argument(
        "--scenarios",
        choices=SCENARIO_ORDER,
        nargs="+",
        default=SCENARIO_ORDER,
        help="Simulation heterogeneity scenarios to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "simulation_graph_seed_replication",
        help="Directory where CSV summaries and plots are written.",
    )
    return parser.parse_args()


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def item_degree_statistics(item_degree: np.ndarray) -> dict[str, float | int]:
    return {
        "item_degree_cv": coefficient_of_variation(item_degree),
        "item_degree_gini": gini_coefficient(item_degree),
        "min_item_degree": int(item_degree.min()),
        "median_item_degree": float(np.median(item_degree)),
        "max_item_degree": int(item_degree.max()),
        "zero_degree_items": int((item_degree == 0).sum()),
    }


def calculate_embedding_statistics(model: LightGCN, item_degree: np.ndarray) -> dict[str, float]:
    model.eval()

    with torch.no_grad():
        _, item_embeddings = model.propagate()

    item_embeddings = item_embeddings.cpu().numpy()
    item_norms = np.linalg.norm(item_embeddings, axis=1)

    return {
        "item_degree_norm_pearson": float(np.corrcoef(item_degree, item_norms)[0, 1]),
        "item_degree_norm_spearman": rank_correlation(item_degree, item_norms),
        "mean_item_embedding_norm": float(item_norms.mean()),
        "std_item_embedding_norm": float(item_norms.std(ddof=0)),
    }


def run_one_graph(
    scenario: str,
    alpha: float,
    graph_seed: int,
    model_seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    print("\n" + "=" * 78)
    print(
        f"Scenario={scenario} | alpha={alpha} | "
        f"graph_generation_seed={graph_seed} | model_optimisation_seed={model_seed}"
    )
    print("=" * 78)

    interactions = generate_synthetic_interactions(alpha=alpha, seed=graph_seed)
    train, validation, test = temporal_leave_one_out(interactions)

    item_groups, low_threshold, high_threshold, item_degree = assign_item_groups(train)

    set_seed(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    norm_adj = build_norm_adj(
        train=train,
        num_users=N_USERS,
        num_items=N_ITEMS,
        device=device,
    )
    train_user_items = build_user_items(train)
    train_edges = train[["user_idx", "item_idx"]].to_numpy()

    model = LightGCN(
        num_users=N_USERS,
        num_items=N_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=N_ITEMS,
            device=device,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f}")

    grouped_results = evaluate_grouped_recall(
        model=model,
        test=test,
        train_user_items=train_user_items,
        item_groups=item_groups,
        device=device,
    )

    embedding_results = calculate_embedding_statistics(model=model, item_degree=item_degree)

    result = {
        "scenario": scenario,
        "alpha": alpha,
        "graph_generation_seed": graph_seed,
        "model_optimisation_seed": model_seed,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "low_item_threshold": low_threshold,
        "high_item_threshold": high_threshold,
        **item_degree_statistics(item_degree),
        **grouped_results,
        **embedding_results,
    }

    print(
        "Recall@20 tail/medium/head: "
        f"{result['tail_recall20']:.4f} / "
        f"{result['medium_recall20']:.4f} / "
        f"{result['head_recall20']:.4f}"
    )
    print(
        "Head-tail gap: "
        f"{result['head_tail_gap']:.4f} | "
        "item degree-norm Pearson: "
        f"{result['item_degree_norm_pearson']:.4f}"
    )

    return result


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "tail_recall20",
        "medium_recall20",
        "head_recall20",
        "head_tail_gap",
        "item_degree_norm_pearson",
        "item_degree_norm_spearman",
        "tail_test_count",
        "medium_test_count",
        "head_test_count",
        "item_degree_cv",
        "item_degree_gini",
        "zero_degree_items",
    ]

    summary_mean = (
        results_df.groupby(["scenario", "alpha"], sort=False)[metric_columns]
        .mean()
        .add_suffix("_mean")
    )
    summary_std = (
        results_df.groupby(["scenario", "alpha"], sort=False)[metric_columns]
        .std(ddof=1)
        .add_suffix("_std")
    )

    summary = pd.concat([summary_mean, summary_std], axis=1).reset_index()
    summary["scenario"] = pd.Categorical(
        summary["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    return summary.sort_values("scenario").reset_index(drop=True)


def plot_compact_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)

    axes[0].errorbar(
        plot_df["alpha"],
        plot_df["head_tail_gap_mean"],
        yerr=plot_df["head_tail_gap_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#1f77b4",
    )
    axes[0].set_title("Head-tail Recall@20 gap")
    axes[0].set_xlabel(r"Heterogeneity $\alpha$")
    axes[0].set_ylabel("Mean +/- SD")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        plot_df["alpha"],
        plot_df["item_degree_norm_pearson_mean"],
        yerr=plot_df["item_degree_norm_pearson_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#2ca02c",
    )
    axes[1].set_title("Item degree-norm Pearson")
    axes[1].set_xlabel(r"Heterogeneity $\alpha$")
    axes[1].set_ylabel("Mean +/- SD")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.set_xticks(plot_df["alpha"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    all_results = []

    for scenario in args.scenarios:
        alpha = SCENARIO_ALPHA[scenario]
        for graph_seed in args.graph_seeds:
            all_results.append(
                run_one_graph(
                    scenario=scenario,
                    alpha=alpha,
                    graph_seed=graph_seed,
                    model_seed=args.model_seed,
                    epochs=args.epochs,
                    device=device,
                )
            )

    results_df = pd.DataFrame(all_results)
    summary_df = summarize_results(results_df)

    raw_path = args.output_dir / "simulation_graph_seed_lightgcn_results.csv"
    summary_path = args.output_dir / "simulation_graph_seed_lightgcn_summary.csv"
    plot_path = args.output_dir / "simulation_graph_seed_replication_compact.png"

    results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_compact_summary(summary_df, plot_path)

    print("\n" + "=" * 78)
    print("Simulation graph-generation seed replication summary")
    print("=" * 78)
    display_columns = [
        "scenario",
        "alpha",
        "tail_recall20_mean",
        "medium_recall20_mean",
        "head_recall20_mean",
        "head_tail_gap_mean",
        "item_degree_norm_pearson_mean",
        "item_degree_norm_spearman_mean",
    ]
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nSaved individual results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved compact plot: {plot_path}")


if __name__ == "__main__":
    main()
