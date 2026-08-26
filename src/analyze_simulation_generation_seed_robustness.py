from pathlib import Path
import os

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/codex_mplconfig",
)

from simulate_degree_heterogeneity import (
    INTERACTIONS_PER_USER,
    N_COMMUNITIES,
    N_ITEMS,
    N_USERS,
    coefficient_of_variation,
    generate_synthetic_interactions,
    gini_coefficient,
    sequential_leave_one_out,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

SCENARIOS = {
    "low": 0.0,
    "medium": 0.7,
    "high": 1.3,
}
GENERATION_SEEDS = [40, 41, 42, 43, 44]
GROUPS = ["tail", "medium", "head"]


def assign_item_groups(
    train: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, float, float]:
    item_degree = (
        train.groupby("item_id")
        .size()
        .reindex(range(N_ITEMS), fill_value=0)
    )

    low_threshold = float(item_degree.quantile(0.20))
    high_threshold = float(item_degree.quantile(0.80))

    def get_group(degree: int) -> str:
        if degree <= low_threshold:
            return "tail"
        if degree >= high_threshold:
            return "head"
        return "medium"

    item_groups = item_degree.map(get_group)

    return (
        item_degree,
        item_groups,
        low_threshold,
        high_threshold,
    )


def analyse_single_graph(
    scenario: str,
    alpha: float,
    generation_seed: int,
) -> dict:
    interactions = generate_synthetic_interactions(
        alpha=alpha,
        seed=generation_seed,
    )
    train, validation, test = sequential_leave_one_out(interactions)
    item_degree, item_groups, low_threshold, high_threshold = (
        assign_item_groups(train)
    )

    degree_values = item_degree.to_numpy(dtype=float)
    test = test.copy()
    test["train_degree"] = test["item_id"].map(item_degree)
    test["item_group"] = test["item_id"].map(item_groups)

    group_counts = test["item_group"].value_counts()
    group_shares = test["item_group"].value_counts(normalize=True)
    cold_start_count = int((test["train_degree"] == 0).sum())

    result = {
        "scenario": scenario,
        "alpha": alpha,
        "generation_seed": generation_seed,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "n_communities": N_COMMUNITIES,
        "interactions_per_user": INTERACTIONS_PER_USER,
        "n_train_interactions": len(train),
        "n_validation_interactions": len(validation),
        "n_test_interactions": len(test),
        "mean_item_degree": float(degree_values.mean()),
        "std_item_degree": float(degree_values.std(ddof=0)),
        "item_degree_cv": coefficient_of_variation(degree_values),
        "item_degree_gini": gini_coefficient(degree_values),
        "min_item_degree": int(degree_values.min()),
        "median_item_degree": float(np.median(degree_values)),
        "max_item_degree": int(degree_values.max()),
        "p90_item_degree": float(np.percentile(degree_values, 90)),
        "zero_degree_items": int((degree_values == 0).sum()),
        "low_group_threshold": low_threshold,
        "high_group_threshold": high_threshold,
        "cold_start_test_count": cold_start_count,
        "cold_start_test_share": cold_start_count / len(test),
    }

    for group in GROUPS:
        result[f"{group}_test_count"] = int(group_counts.get(group, 0))
        result[f"{group}_test_share"] = float(
            group_shares.get(group, 0.0)
        )

    return result


def summarise_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "item_degree_cv",
        "item_degree_gini",
        "min_item_degree",
        "median_item_degree",
        "max_item_degree",
        "p90_item_degree",
        "zero_degree_items",
        "tail_test_share",
        "medium_test_share",
        "head_test_share",
        "cold_start_test_share",
    ]

    summary_mean = (
        df.groupby(["scenario", "alpha"])[metric_columns]
        .mean()
        .add_suffix("_mean")
    )
    summary_std = (
        df.groupby(["scenario", "alpha"])[metric_columns]
        .std()
        .add_suffix("_std")
    )
    summary = pd.concat([summary_mean, summary_std], axis=1).reset_index()

    scenario_order = pd.Categorical(
        summary["scenario"],
        categories=["low", "medium", "high"],
        ordered=True,
    )
    summary["scenario"] = scenario_order
    return summary.sort_values("scenario").reset_index(drop=True)


def make_monotonicity_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for generation_seed, seed_df in df.groupby("generation_seed"):
        ordered = (
            seed_df.set_index("scenario")
            .loc[["low", "medium", "high"]]
            .reset_index()
        )

        rows.append(
            {
                "generation_seed": generation_seed,
                "cv_increases_low_to_high": bool(
                    ordered["item_degree_cv"].is_monotonic_increasing
                ),
                "gini_increases_low_to_high": bool(
                    ordered["item_degree_gini"].is_monotonic_increasing
                ),
                "head_test_share_increases_low_to_high": bool(
                    ordered["head_test_share"].is_monotonic_increasing
                ),
                "tail_test_share_decreases_low_to_high": bool(
                    ordered["tail_test_share"].is_monotonic_decreasing
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for scenario, alpha in SCENARIOS.items():
        for generation_seed in GENERATION_SEEDS:
            print(
                f"Analysing scenario={scenario}, "
                f"alpha={alpha}, seed={generation_seed}"
            )
            rows.append(
                analyse_single_graph(
                    scenario=scenario,
                    alpha=alpha,
                    generation_seed=generation_seed,
                )
            )

    raw = pd.DataFrame(rows)
    summary = summarise_numeric_columns(raw)
    checks = make_monotonicity_checks(raw)

    raw_path = (
        RESULTS_DIR
        / "simulation_generation_seed_robustness_raw.csv"
    )
    summary_path = (
        RESULTS_DIR
        / "simulation_generation_seed_robustness_summary.csv"
    )
    checks_path = (
        RESULTS_DIR
        / "simulation_generation_seed_robustness_checks.csv"
    )

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    checks.to_csv(checks_path, index=False)

    print("\nGeneration-seed robustness summary:")
    selected_columns = [
        "scenario",
        "alpha",
        "item_degree_cv_mean",
        "item_degree_cv_std",
        "item_degree_gini_mean",
        "item_degree_gini_std",
        "head_test_share_mean",
        "head_test_share_std",
        "tail_test_share_mean",
        "tail_test_share_std",
        "cold_start_test_share_mean",
    ]
    print(summary[selected_columns].to_string(index=False))

    print("\nMonotonicity checks by generation seed:")
    print(checks.to_string(index=False))

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {checks_path}")


if __name__ == "__main__":
    main()
