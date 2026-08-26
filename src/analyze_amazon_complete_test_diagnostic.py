from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DESKTOP_ROOT_DIR = Path("/Users/jinsixiong/Desktop/硕士毕业论文/dissertation-recommender")
LOCAL_ROOT_DIR = Path(__file__).resolve().parents[1]


def find_repo_root() -> Path:
    if (DESKTOP_ROOT_DIR / "data").exists():
        return DESKTOP_ROOT_DIR
    if (LOCAL_ROOT_DIR / "data").exists():
        return LOCAL_ROOT_DIR
    raise FileNotFoundError("Could not locate dissertation-recommender data directory.")


ROOT_DIR = find_repo_root()
DEFAULT_FULL_DATA_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core"
DEFAULT_WARM_DATA_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "amazon_beauty_3core_warm"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "amazon_complete_test_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report Amazon Beauty complete-test performance, including cold-start "
            "held-out items absent from the training graph."
        )
    )
    parser.add_argument("--full-data-dir", type=Path, default=DEFAULT_FULL_DATA_DIR)
    parser.add_argument("--warm-data-dir", type=Path, default=DEFAULT_WARM_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    return parser.parse_args()


def format_mean_std(mean: float, std: float | None) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_cold_start_test(full_data_dir: Path, warm_data_dir: Path) -> pd.DataFrame:
    cold_path = warm_data_dir / "cold_start_test.csv"
    if cold_path.exists():
        return pd.read_csv(cold_path)

    train = pd.read_csv(full_data_dir / "train.csv")
    test = pd.read_csv(full_data_dir / "test.csv")
    train_items = set(map(int, train["item_idx"].unique()))
    return test[~test["item_idx"].isin(train_items)].copy()


def make_cold_rows_for_baselines(
    cold_start_test: pd.DataFrame,
    baseline_per_user: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    template = baseline_per_user[baseline_per_user["K"] == k]
    models = template[["model", "run"]].drop_duplicates()

    rows = []
    for _, model_row in models.iterrows():
        for _, row in cold_start_test.iterrows():
            rows.append(
                {
                    "dataset": "Amazon Beauty 3-core complete",
                    "model": model_row["model"],
                    "run": model_row["run"],
                    "user_idx": int(row["user_idx"]),
                    "item_idx": int(row["item_idx"]),
                    "user_group": "unknown",
                    "item_group": "cold_start",
                    "K": k,
                    "Recall": 0.0,
                    "NDCG": 0.0,
                }
            )
    return pd.DataFrame(rows)


def make_cold_rows_for_lightgcn(
    cold_start_test: pd.DataFrame,
    lightgcn_per_user: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    template = lightgcn_per_user[lightgcn_per_user["K"] == k]
    runs = template[["model", "run", "seed"]].drop_duplicates()

    rows = []
    for _, run_row in runs.iterrows():
        for _, row in cold_start_test.iterrows():
            rows.append(
                {
                    "dataset": "Amazon Beauty 3-core complete",
                    "model": run_row["model"],
                    "run": run_row["run"],
                    "seed": int(run_row["seed"]),
                    "user_idx": int(row["user_idx"]),
                    "item_idx": int(row["item_idx"]),
                    "user_group": "unknown",
                    "item_group": "cold_start",
                    "K": k,
                    "Recall": 0.0,
                    "NDCG": 0.0,
                }
            )
    return pd.DataFrame(rows)


def complete_rows(
    baseline_per_user: pd.DataFrame,
    lightgcn_per_user: pd.DataFrame,
    cold_start_test: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_warm = baseline_per_user[baseline_per_user["K"] == k].copy()
    lightgcn_warm = lightgcn_per_user[lightgcn_per_user["K"] == k].copy()
    baseline_warm["dataset"] = "Amazon Beauty 3-core complete"
    lightgcn_warm["dataset"] = "Amazon Beauty 3-core complete"

    baseline_complete = pd.concat(
        [
            baseline_warm,
            make_cold_rows_for_baselines(cold_start_test, baseline_per_user, k),
        ],
        ignore_index=True,
    )
    lightgcn_complete = pd.concat(
        [
            lightgcn_warm,
            make_cold_rows_for_lightgcn(cold_start_test, lightgcn_per_user, k),
        ],
        ignore_index=True,
    )
    return baseline_complete, lightgcn_complete


def summarize_baseline(
    warm: pd.DataFrame,
    complete: pd.DataFrame,
    k: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for model in ["Popularity", "ItemCF"]:
        warm_model = warm[(warm["model"] == model) & (warm["K"] == k)]
        complete_model = complete[(complete["model"] == model) & (complete["K"] == k)]
        rows.append(
            {
                "model": model,
                "n_complete_test": int(len(complete_model)),
                "n_warm_start_test": int(len(warm_model)),
                "n_cold_start_test": int(len(complete_model) - len(warm_model)),
                "cold_start_share": float((len(complete_model) - len(warm_model)) / len(complete_model)),
                "complete_recall20_mean": float(complete_model["Recall"].mean()),
                "complete_recall20_std": np.nan,
                "complete_ndcg20_mean": float(complete_model["NDCG"].mean()),
                "complete_ndcg20_std": np.nan,
                "warm_recall20_mean": float(warm_model["Recall"].mean()),
                "warm_recall20_std": np.nan,
                "warm_ndcg20_mean": float(warm_model["NDCG"].mean()),
                "warm_ndcg20_std": np.nan,
            }
        )
    return rows


def summarize_lightgcn(
    warm: pd.DataFrame,
    complete: pd.DataFrame,
    k: int,
) -> dict[str, float | int | str]:
    warm_seed = (
        warm[(warm["model"] == "LightGCN") & (warm["K"] == k)]
        .groupby("seed", as_index=False)
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
    )
    complete_seed = (
        complete[(complete["model"] == "LightGCN") & (complete["K"] == k)]
        .groupby("seed", as_index=False)
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
    )

    n_complete = int(complete_seed["n"].iloc[0])
    n_warm = int(warm_seed["n"].iloc[0])
    return {
        "model": "LightGCN",
        "n_complete_test": n_complete,
        "n_warm_start_test": n_warm,
        "n_cold_start_test": n_complete - n_warm,
        "cold_start_share": float((n_complete - n_warm) / n_complete),
        "complete_recall20_mean": float(complete_seed["Recall"].mean()),
        "complete_recall20_std": float(complete_seed["Recall"].std(ddof=1)),
        "complete_ndcg20_mean": float(complete_seed["NDCG"].mean()),
        "complete_ndcg20_std": float(complete_seed["NDCG"].std(ddof=1)),
        "warm_recall20_mean": float(warm_seed["Recall"].mean()),
        "warm_recall20_std": float(warm_seed["Recall"].std(ddof=1)),
        "warm_ndcg20_mean": float(warm_seed["NDCG"].mean()),
        "warm_ndcg20_std": float(warm_seed["NDCG"].std(ddof=1)),
    }


def make_formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Complete n": int(row["n_complete_test"]),
                "Warm n": int(row["n_warm_start_test"]),
                "Cold n": int(row["n_cold_start_test"]),
                "Cold share": f"{row['cold_start_share']:.4f}",
                "Complete Recall@20": format_mean_std(
                    row["complete_recall20_mean"],
                    row["complete_recall20_std"],
                ),
                "Complete NDCG@20": format_mean_std(
                    row["complete_ndcg20_mean"],
                    row["complete_ndcg20_std"],
                ),
                "Warm Recall@20": format_mean_std(
                    row["warm_recall20_mean"],
                    row["warm_recall20_std"],
                ),
                "Warm NDCG@20": format_mean_std(
                    row["warm_ndcg20_mean"],
                    row["warm_ndcg20_std"],
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_test = pd.read_csv(args.full_data_dir / "test.csv")
    warm_test = pd.read_csv(args.warm_data_dir / "test.csv")
    cold_start_test = load_cold_start_test(args.full_data_dir, args.warm_data_dir)

    baseline_per_user = pd.read_csv(args.results_dir / "amazon_baseline_per_user_metrics.csv")
    lightgcn_per_user = pd.read_csv(args.results_dir / "amazon_lightgcn_per_user_metrics_all_seeds.csv")

    baseline_complete, lightgcn_complete = complete_rows(
        baseline_per_user,
        lightgcn_per_user,
        cold_start_test,
        args.k,
    )
    complete_per_user = pd.concat([baseline_complete, lightgcn_complete], ignore_index=True)

    rows = summarize_baseline(baseline_per_user, baseline_complete, args.k)
    rows.append(summarize_lightgcn(lightgcn_per_user, lightgcn_complete, args.k))
    summary = pd.DataFrame(rows)

    formatted = make_formatted_summary(summary)

    diagnostic = pd.DataFrame(
        [
            {
                "full_test_interactions": int(len(full_test)),
                "warm_start_test_interactions": int(len(warm_test)),
                "cold_start_test_interactions": int(len(cold_start_test)),
                "cold_start_test_share": float(len(cold_start_test) / len(full_test)),
            }
        ]
    )

    diagnostic.to_csv(args.output_dir / "amazon_complete_test_counts.csv", index=False)
    summary.to_csv(args.output_dir / "amazon_complete_test_diagnostic_summary.csv", index=False)
    formatted.to_csv(args.output_dir / "amazon_complete_test_diagnostic_summary_formatted.csv", index=False)
    complete_per_user.to_csv(args.output_dir / "amazon_complete_test_per_user_metrics.csv", index=False)

    print("Amazon Beauty complete test diagnostic")
    print("=" * 45)
    print(diagnostic.to_string(index=False))
    print("\nSummary:")
    print(formatted.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
