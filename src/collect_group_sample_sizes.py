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
DEFAULT_OUTPUT_DIR = LOCAL_ROOT_DIR / "outputs" / "group_sample_sizes"
SIM_REPLICATION_CANDIDATES = [
    LOCAL_ROOT_DIR / "outputs" / "simulation_graph_seed_replication",
    ROOT_DIR / "results" / "simulation_graph_seed_replication",
]

GROUP_LABELS = {
    "low": "Tail",
    "tail": "Tail",
    "medium": "Medium",
    "high": "Head",
    "head": "Head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect held-out test item-group sample sizes for thesis reporting."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--simulation-replication-dir", type=Path, default=None)
    return parser.parse_args()


def locate_sim_replication_dir(explicit_dir: Path | None) -> Path | None:
    if explicit_dir is not None:
        return explicit_dir
    for candidate in SIM_REPLICATION_CANDIDATES:
        if (candidate / "simulation_graph_seed_lightgcn_summary.csv").exists():
            return candidate
    return None


def movielens_counts() -> pd.DataFrame:
    path = ROOT_DIR / "results" / "all_model_item_group_summary.csv"
    df = pd.read_csv(path)
    rows = (
        df[(df["model"] == "Popularity") & (df["K"] == 20)]
        .copy()
        .assign(
            dataset="MovieLens-1M",
            evaluation="Temporal leave-one-out test set",
            alpha=np.nan,
            group=lambda x: x["group"].map(GROUP_LABELS).fillna(x["group"]),
            n=lambda x: x["n_test_cases"],
        )[["dataset", "evaluation", "alpha", "group", "n"]]
    )
    return rows[rows["group"].isin(["Tail", "Medium", "Head"])].copy()


def amazon_counts() -> pd.DataFrame:
    path = ROOT_DIR / "results" / "amazon_beauty_3core_warm" / "amazon_baseline_item_group_metrics.csv"
    df = pd.read_csv(path)
    rows = (
        df[(df["model"] == "Popularity") & (df["K"] == 20)]
        .copy()
        .assign(
            dataset="Amazon Beauty",
            evaluation="Warm-start test subset",
            alpha=np.nan,
            group=lambda x: x["item_group"].map(GROUP_LABELS).fillna(x["item_group"]),
            n=lambda x: x["n"],
        )[["dataset", "evaluation", "alpha", "group", "n"]]
    )
    return rows[rows["group"].isin(["Tail", "Medium", "Head"])].copy()


def simulation_main_counts() -> pd.DataFrame:
    path = ROOT_DIR / "results" / "simulation_test_item_composition.csv"
    df = pd.read_csv(path)
    alpha_map = {"low": 0.0, "medium": 0.7, "high": 1.3}
    rows = []
    for _, row in df.iterrows():
        scenario = row["scenario"]
        rows.extend(
            [
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Main graph realisation, graph seed 42",
                    "alpha": alpha_map[scenario],
                    "group": "Tail",
                    "n": int(row["tail_test_count"]),
                },
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Main graph realisation, graph seed 42",
                    "alpha": alpha_map[scenario],
                    "group": "Medium",
                    "n": int(row["medium_test_count"]),
                },
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Main graph realisation, graph seed 42",
                    "alpha": alpha_map[scenario],
                    "group": "Head",
                    "n": int(row["head_test_count"]),
                },
            ]
        )
    return pd.DataFrame(rows)


def simulation_replication_counts(replication_dir: Path | None) -> pd.DataFrame:
    if replication_dir is None:
        return pd.DataFrame()
    path = replication_dir / "simulation_graph_seed_lightgcn_summary.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        scenario = row["scenario"]
        rows.extend(
            [
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Mean across graph-generation seeds 40-44",
                    "alpha": float(row["alpha"]),
                    "group": "Tail",
                    "n": float(row["tail_test_count_mean"]),
                    "n_std": float(row["tail_test_count_std"]),
                },
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Mean across graph-generation seeds 40-44",
                    "alpha": float(row["alpha"]),
                    "group": "Medium",
                    "n": float(row["medium_test_count_mean"]),
                    "n_std": float(row["medium_test_count_std"]),
                },
                {
                    "dataset": f"Simulation {scenario}",
                    "evaluation": "Mean across graph-generation seeds 40-44",
                    "alpha": float(row["alpha"]),
                    "group": "Head",
                    "n": float(row["head_test_count_mean"]),
                    "n_std": float(row["head_test_count_std"]),
                },
            ]
        )
    return pd.DataFrame(rows)


def pivot_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["alpha_display"] = df["alpha"].apply(
        lambda value: "" if pd.isna(value) else f"{float(value):.1f}"
    )
    pivot = (
        df.pivot_table(
            index=["dataset", "evaluation", "alpha_display"],
            columns="group",
            values="n",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["Tail", "Medium", "Head"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot.rename(columns={"alpha_display": "alpha"})[
        ["dataset", "evaluation", "alpha", "Tail", "Medium", "Head"]
    ]
    dataset_order = {
        "MovieLens-1M": 0,
        "Amazon Beauty": 1,
        "Simulation low": 2,
        "Simulation medium": 3,
        "Simulation high": 4,
    }
    evaluation_order = {
        "Temporal leave-one-out test set": 0,
        "Warm-start test subset": 0,
        "Main graph realisation, graph seed 42": 0,
        "Mean across graph-generation seeds 40-44": 1,
    }
    pivot["dataset_order"] = pivot["dataset"].map(dataset_order).fillna(99)
    pivot["evaluation_order"] = pivot["evaluation"].map(evaluation_order).fillna(99)
    return (
        pivot.sort_values(["dataset_order", "evaluation_order"])
        .drop(columns=["dataset_order", "evaluation_order"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_replication_dir = locate_sim_replication_dir(args.simulation_replication_dir)

    frames = [
        movielens_counts(),
        amazon_counts(),
        simulation_main_counts(),
    ]
    sim_rep = simulation_replication_counts(sim_replication_dir)
    if not sim_rep.empty:
        frames.append(sim_rep)

    long_df = pd.concat(frames, ignore_index=True)
    wide_df = pivot_counts(long_df)

    long_path = args.output_dir / "item_group_sample_sizes_long.csv"
    wide_path = args.output_dir / "item_group_sample_sizes_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("Item-group held-out test sample sizes")
    print("=" * 45)
    print(wide_df.to_string(index=False))
    print(f"\nSaved long format: {long_path}")
    print(f"Saved wide format: {wide_path}")


if __name__ == "__main__":
    main()
