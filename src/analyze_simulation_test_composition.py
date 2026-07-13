from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SIM_DATA_DIR = ROOT_DIR / "data" / "simulated"
RESULTS_DIR = ROOT_DIR / "results"

N_ITEMS = 600
SCENARIOS = ["low", "medium", "high"]


def temporal_leave_one_out(interactions):
    interactions = interactions.sort_values(
        ["user_id", "interaction_order"]
    ).copy()

    test = interactions.groupby("user_id").tail(1).copy()
    remaining = interactions.drop(test.index)

    validation = remaining.groupby("user_id").tail(1).copy()
    train = remaining.drop(validation.index).copy()

    return train, validation, test


def assign_item_groups(train):
    item_degree = (
        train.groupby("item_id")
        .size()
        .reindex(range(N_ITEMS), fill_value=0)
    )

    low_threshold = item_degree.quantile(0.20)
    high_threshold = item_degree.quantile(0.80)

    def get_group(degree):
        if degree <= low_threshold:
            return "tail"
        if degree >= high_threshold:
            return "head"
        return "medium"

    item_groups = item_degree.map(get_group)

    return item_degree, item_groups, low_threshold, high_threshold


def main():
    rows = []

    for scenario in SCENARIOS:
        path = (
            SIM_DATA_DIR
            / f"{scenario}_heterogeneity_interactions.csv"
        )

        interactions = pd.read_csv(path)
        train, validation, test = temporal_leave_one_out(interactions)

        item_degree, item_groups, low_threshold, high_threshold = (
            assign_item_groups(train)
        )

        test = test.copy()
        test["train_degree"] = test["item_id"].map(item_degree)
        test["item_group"] = test["item_id"].map(item_groups)

        group_counts = test["item_group"].value_counts()
        group_proportions = test["item_group"].value_counts(
            normalize=True
        )

        cold_start_count = int((test["train_degree"] == 0).sum())
        cold_start_rate = cold_start_count / len(test)

        row = {
            "scenario": scenario,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "tail_test_count": int(group_counts.get("tail", 0)),
            "medium_test_count": int(group_counts.get("medium", 0)),
            "head_test_count": int(group_counts.get("head", 0)),
            "tail_test_share": float(
                group_proportions.get("tail", 0.0)
            ),
            "medium_test_share": float(
                group_proportions.get("medium", 0.0)
            ),
            "head_test_share": float(
                group_proportions.get("head", 0.0)
            ),
            "cold_start_test_count": cold_start_count,
            "cold_start_test_rate": cold_start_rate,
        }

        rows.append(row)

        print("\n" + "=" * 60)
        print(f"Scenario: {scenario}")
        print("=" * 60)
        print(
            f"Tail / Medium / Head test counts: "
            f"{row['tail_test_count']} / "
            f"{row['medium_test_count']} / "
            f"{row['head_test_count']}"
        )
        print(
            f"Tail / Medium / Head test shares: "
            f"{row['tail_test_share']:.3f} / "
            f"{row['medium_test_share']:.3f} / "
            f"{row['head_test_share']:.3f}"
        )
        print(
            f"Cold-start test items: "
            f"{cold_start_count} "
            f"({cold_start_rate:.3%})"
        )

    summary = pd.DataFrame(rows)

    output_path = (
        RESULTS_DIR
        / "simulation_test_item_composition.csv"
    )

    summary.to_csv(output_path, index=False)

    print("\nSimulation test-item composition summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()