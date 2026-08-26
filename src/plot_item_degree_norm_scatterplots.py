from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DESKTOP_ROOT_DIR = Path("/Users/jinsixiong/Desktop/硕士毕业论文/dissertation-recommender")
LOCAL_ROOT_DIR = Path(__file__).resolve().parents[1]


def find_repo_root() -> Path:
    if (DESKTOP_ROOT_DIR / "results").exists():
        return DESKTOP_ROOT_DIR
    if (LOCAL_ROOT_DIR / "results").exists():
        return LOCAL_ROOT_DIR
    raise FileNotFoundError("Could not locate dissertation-recommender results directory.")


ROOT_DIR = find_repo_root()
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "degree_norm_scatterplots"
DEFAULT_SEEDS = [42, 43, 44]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot item embedding norm against log(1 + training item degree) "
            "for MovieLens and Amazon Beauty."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-bins", type=int, default=20)
    return parser.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson(x_rank, y_rank)


def load_mean_item_norms(
    degrees_path: Path,
    embedding_path_template: str,
    seeds: list[int],
    dataset: str,
) -> pd.DataFrame:
    degrees = pd.read_csv(degrees_path)
    norm_frames = []

    for seed in seeds:
        emb_path = Path(embedding_path_template.format(seed=seed))
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")

        item_emb = np.load(emb_path)
        norms = pd.DataFrame(
            {
                "item_idx": np.arange(len(item_emb)),
                f"embedding_norm_seed_{seed}": np.linalg.norm(item_emb, axis=1),
            }
        )
        norm_frames.append(norms)

    item_norms = norm_frames[0]
    for frame in norm_frames[1:]:
        item_norms = item_norms.merge(frame, on="item_idx", how="inner")

    norm_cols = [col for col in item_norms.columns if col.startswith("embedding_norm_seed_")]
    item_norms["embedding_norm_mean"] = item_norms[norm_cols].mean(axis=1)
    item_norms["embedding_norm_std"] = item_norms[norm_cols].std(axis=1, ddof=1)

    plot_df = degrees.merge(
        item_norms[["item_idx", "embedding_norm_mean", "embedding_norm_std"]],
        on="item_idx",
        how="inner",
    )
    plot_df["dataset"] = dataset
    plot_df["log1p_degree"] = np.log1p(plot_df["degree"].astype(float))
    return plot_df


def binned_mean_trend(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    n_unique = df["log1p_degree"].nunique()
    bins = max(2, min(n_bins, n_unique))

    trend = df.copy()
    trend["bin"] = pd.cut(
        trend["log1p_degree"],
        bins=bins,
    )
    return (
        trend.groupby("bin", observed=True)
        .agg(
            log1p_degree=("log1p_degree", "mean"),
            embedding_norm_mean=("embedding_norm_mean", "mean"),
            n_items=("item_idx", "count"),
        )
        .reset_index(drop=True)
    )


def add_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    trend: pd.DataFrame,
    title: str,
    color: str,
) -> None:
    ax.scatter(
        df["log1p_degree"],
        df["embedding_norm_mean"],
        s=18,
        alpha=0.35,
        color=color,
        edgecolors="none",
        label="Items",
    )
    ax.plot(
        trend["log1p_degree"],
        trend["embedding_norm_mean"],
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Binned mean",
    )

    r = pearson(df["degree"].to_numpy(dtype=float), df["embedding_norm_mean"].to_numpy(dtype=float))
    rho = spearman(df["degree"].to_numpy(dtype=float), df["embedding_norm_mean"].to_numpy(dtype=float))
    label = f"$n$={len(df)}\nPearson={r:.3f}\nSpearman={rho:.3f}"
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\log(1+d_i)$")
    ax.set_ylabel("Mean item embedding norm")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True, fontsize=8)


def save_single_plot(df: pd.DataFrame, trend: pd.DataFrame, title: str, color: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    add_panel(ax, df, trend, title, color)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    movielens = load_mean_item_norms(
        degrees_path=ROOT_DIR / "data" / "processed" / "item_degrees.csv",
        embedding_path_template=str(ROOT_DIR / "results" / "lightgcn_item_embeddings_seed_{seed}.npy"),
        seeds=args.seeds,
        dataset="MovieLens-1M",
    )
    amazon = load_mean_item_norms(
        degrees_path=ROOT_DIR
        / "data"
        / "processed"
        / "amazon_beauty_3core_warm"
        / "item_degrees.csv",
        embedding_path_template=str(
            ROOT_DIR
            / "results"
            / "amazon_beauty_3core_warm"
            / "amazon_lightgcn_item_embeddings_seed_{seed}.npy"
        ),
        seeds=args.seeds,
        dataset="Amazon Beauty",
    )

    movielens_trend = binned_mean_trend(movielens, args.n_bins)
    amazon_trend = binned_mean_trend(amazon, min(args.n_bins, 10))

    combined_data = pd.concat([movielens, amazon], ignore_index=True)
    combined_trend = pd.concat(
        [
            movielens_trend.assign(dataset="MovieLens-1M"),
            amazon_trend.assign(dataset="Amazon Beauty"),
        ],
        ignore_index=True,
    )
    combined_data.to_csv(args.output_dir / "item_degree_norm_scatter_data.csv", index=False)
    combined_trend.to_csv(args.output_dir / "item_degree_norm_binned_trends.csv", index=False)

    save_single_plot(
        movielens,
        movielens_trend,
        "MovieLens-1M items",
        "#1f77b4",
        args.output_dir / "movielens_item_degree_norm_scatter.png",
    )
    save_single_plot(
        amazon,
        amazon_trend,
        "Amazon Beauty items",
        "#d95f02",
        args.output_dir / "amazon_beauty_item_degree_norm_scatter.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    add_panel(axes[0], movielens, movielens_trend, "MovieLens-1M items", "#1f77b4")
    add_panel(axes[1], amazon, amazon_trend, "Amazon Beauty items", "#d95f02")
    fig.tight_layout()
    fig.savefig(args.output_dir / "item_degree_norm_scatterplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for dataset, df in [("MovieLens-1M", movielens), ("Amazon Beauty", amazon)]:
        summary_rows.append(
            {
                "dataset": dataset,
                "n_items": len(df),
                "pearson": pearson(df["degree"].to_numpy(float), df["embedding_norm_mean"].to_numpy(float)),
                "spearman": spearman(df["degree"].to_numpy(float), df["embedding_norm_mean"].to_numpy(float)),
                "min_degree": int(df["degree"].min()),
                "median_degree": float(df["degree"].median()),
                "max_degree": int(df["degree"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "item_degree_norm_scatter_summary.csv", index=False)

    print("Saved degree-norm scatterplots to:", args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
