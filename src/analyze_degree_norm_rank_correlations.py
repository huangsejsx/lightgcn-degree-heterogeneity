from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "degree_norm_rank_correlations"
DEFAULT_SIM_REPLICATION_DIRS = [
    LOCAL_ROOT_DIR / "outputs" / "simulation_graph_seed_replication",
    ROOT_DIR / "results" / "simulation_graph_seed_replication",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Pearson and Spearman degree-embedding norm correlations."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--simulation-replication-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing simulation_graph_seed_lightgcn_results.csv. "
            "Defaults to the newest generated output directory if available."
        ),
    )
    return parser.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson(x_rank, y_rank)


def correlation_row(
    dataset: str,
    node_type: str,
    degree: np.ndarray,
    embedding_norm: np.ndarray,
    replication_type: str,
    replication_value: int,
) -> dict[str, float | int | str]:
    return {
        "dataset": dataset,
        "node_type": node_type,
        "replication_type": replication_type,
        "replication_value": replication_value,
        "n_nodes": int(len(degree)),
        "pearson": pearson(degree, embedding_norm),
        "spearman": spearman(degree, embedding_norm),
    }


def movieLens_rows(seeds: list[int]) -> list[dict[str, float | int | str]]:
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"

    user_degrees = pd.read_csv(data_dir / "user_degrees.csv")
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")

    rows = []
    for seed in seeds:
        user_path = results_dir / f"lightgcn_user_embeddings_seed_{seed}.npy"
        item_path = results_dir / f"lightgcn_item_embeddings_seed_{seed}.npy"
        if not user_path.exists() or not item_path.exists():
            continue

        user_emb = np.load(user_path)
        item_emb = np.load(item_path)

        user_norms = pd.DataFrame(
            {
                "user_idx": np.arange(len(user_emb)),
                "embedding_norm": np.linalg.norm(user_emb, axis=1),
            }
        ).merge(user_degrees, on="user_idx", how="inner")

        item_norms = pd.DataFrame(
            {
                "item_idx": np.arange(len(item_emb)),
                "embedding_norm": np.linalg.norm(item_emb, axis=1),
            }
        ).merge(item_degrees, on="item_idx", how="inner")

        rows.append(
            correlation_row(
                dataset="MovieLens-1M",
                node_type="user",
                degree=user_norms["degree"].to_numpy(dtype=float),
                embedding_norm=user_norms["embedding_norm"].to_numpy(dtype=float),
                replication_type="optimisation_seed",
                replication_value=seed,
            )
        )
        rows.append(
            correlation_row(
                dataset="MovieLens-1M",
                node_type="item",
                degree=item_norms["degree"].to_numpy(dtype=float),
                embedding_norm=item_norms["embedding_norm"].to_numpy(dtype=float),
                replication_type="optimisation_seed",
                replication_value=seed,
            )
        )

    return rows


def amazon_rows(seeds: list[int]) -> list[dict[str, float | int | str]]:
    data_dir = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"
    results_dir = ROOT_DIR / "results" / "amazon_beauty_3core_warm"
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")

    rows = []
    for seed in seeds:
        item_path = results_dir / f"amazon_lightgcn_item_embeddings_seed_{seed}.npy"
        if not item_path.exists():
            continue

        item_emb = np.load(item_path)
        item_norms = pd.DataFrame(
            {
                "item_idx": np.arange(len(item_emb)),
                "embedding_norm": np.linalg.norm(item_emb, axis=1),
            }
        ).merge(item_degrees, on="item_idx", how="inner")

        rows.append(
            correlation_row(
                dataset="Amazon Beauty",
                node_type="item",
                degree=item_norms["degree"].to_numpy(dtype=float),
                embedding_norm=item_norms["embedding_norm"].to_numpy(dtype=float),
                replication_type="optimisation_seed",
                replication_value=seed,
            )
        )

    return rows


def locate_simulation_replication_dir(explicit_dir: Path | None) -> Path | None:
    if explicit_dir is not None:
        return explicit_dir
    for candidate in DEFAULT_SIM_REPLICATION_DIRS:
        if (candidate / "simulation_graph_seed_lightgcn_results.csv").exists():
            return candidate
    return None


def simulation_rows(replication_dir: Path | None) -> list[dict[str, float | int | str]]:
    if replication_dir is None:
        return []

    path = replication_dir / "simulation_graph_seed_lightgcn_results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": f"Simulation {row['scenario']}",
                "node_type": "item",
                "replication_type": "graph_generation_seed",
                "replication_value": int(row["graph_generation_seed"]),
                "n_nodes": 600,
                "pearson": float(row["item_degree_norm_pearson"]),
                "spearman": float(row["item_degree_norm_spearman"]),
                "alpha": float(row["alpha"]),
            }
        )
    return rows


def summarize(correlations: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "node_type"]
    if "alpha" not in correlations.columns:
        correlations["alpha"] = np.nan

    summary = (
        correlations.groupby(group_cols, dropna=False)
        .agg(
            alpha=("alpha", "first"),
            n_replications=("replication_value", "count"),
            n_nodes_mean=("n_nodes", "mean"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
        .reset_index()
    )

    dataset_order = {
        "MovieLens-1M": 0,
        "Amazon Beauty": 1,
        "Simulation low": 2,
        "Simulation medium": 3,
        "Simulation high": 4,
    }
    node_order = {"user": 0, "item": 1}
    summary["dataset_order"] = summary["dataset"].map(dataset_order).fillna(99)
    summary["node_order"] = summary["node_type"].map(node_order).fillna(99)
    summary = summary.sort_values(["dataset_order", "node_order"]).drop(
        columns=["dataset_order", "node_order"]
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_dir = locate_simulation_replication_dir(args.simulation_replication_dir)
    rows = []
    rows.extend(movieLens_rows(args.seeds))
    rows.extend(amazon_rows(args.seeds))
    rows.extend(simulation_rows(sim_dir))

    correlations = pd.DataFrame(rows)
    summary = summarize(correlations)

    raw_path = args.output_dir / "degree_norm_rank_correlations_by_replication.csv"
    summary_path = args.output_dir / "degree_norm_rank_correlation_summary.csv"
    correlations.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Degree-norm Pearson and Spearman correlations")
    print("=" * 55)
    print(summary.to_string(index=False))
    print(f"\nSaved individual correlations: {raw_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
