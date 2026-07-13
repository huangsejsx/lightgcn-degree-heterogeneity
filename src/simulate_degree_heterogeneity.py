from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "simulated"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_USERS = 1000
N_ITEMS = 600
N_COMMUNITIES = 5
INTERACTIONS_PER_USER = 20
RANDOM_SEED = 42

SCENARIOS = {
    "low": 0.0,
    "medium": 0.7,
    "high": 1.3,
}


def coefficient_of_variation(values: np.ndarray) -> float:
    mean_value = values.mean()

    if mean_value == 0:
        return 0.0

    return float(values.std(ddof=0) / mean_value)


def gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if np.all(values == 0):
        return 0.0

    values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(values)

    return float(
        (
            n
            + 1
            - 2 * np.sum(cumulative) / cumulative[-1]
        )
        / n
    )


def generate_synthetic_interactions(
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    users = np.arange(N_USERS)
    items = np.arange(N_ITEMS)

    user_communities = users % N_COMMUNITIES
    item_communities = items % N_COMMUNITIES

    rng.shuffle(user_communities)

    local_item_rank = np.zeros(N_ITEMS, dtype=int)

    for community in range(N_COMMUNITIES):
        community_items = items[
            item_communities == community
        ]

        shuffled_items = rng.permutation(
            community_items
        )

        for rank, item_id in enumerate(
            shuffled_items,
            start=1,
        ):
            local_item_rank[item_id] = rank

    popularity_weights = (
        local_item_rank.astype(float) ** (-alpha)
    )

    records = []

    for user_id in users:
        user_community = user_communities[user_id]

        community_multiplier = np.where(
            item_communities == user_community,
            4.0,
            1.0,
        )

        sampling_weights = (
            popularity_weights
            * community_multiplier
        )

        sampling_probabilities = (
            sampling_weights
            / sampling_weights.sum()
        )

        selected_items = rng.choice(
            items,
            size=INTERACTIONS_PER_USER,
            replace=False,
            p=sampling_probabilities,
        )

        selected_items = rng.permutation(
            selected_items
        )

        for interaction_order, item_id in enumerate(
            selected_items,
            start=1,
        ):
            records.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "interaction_order": interaction_order,
                    "user_community": int(
                        user_community
                    ),
                    "item_community": int(
                        item_communities[item_id]
                    ),
                }
            )

    return pd.DataFrame(records)


def sequential_leave_one_out(
    interactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    interactions = interactions.sort_values(
        ["user_id", "interaction_order"]
    ).copy()

    test = (
        interactions
        .groupby("user_id")
        .tail(1)
        .copy()
    )

    remaining = interactions.drop(test.index)

    validation = (
        remaining
        .groupby("user_id")
        .tail(1)
        .copy()
    )

    train = remaining.drop(
        validation.index
    ).copy()

    return train, validation, test


def calculate_degree_statistics(
    train: pd.DataFrame,
    scenario: str,
    alpha: float,
) -> tuple[pd.DataFrame, dict]:
    item_degree = (
        train
        .groupby("item_id")
        .size()
        .reindex(
            range(N_ITEMS),
            fill_value=0,
        )
        .rename("degree")
        .reset_index()
    )

    degree_values = (
        item_degree["degree"]
        .to_numpy()
    )

    summary = {
        "scenario": scenario,
        "alpha": alpha,
        "generation_seed": RANDOM_SEED,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "n_train_interactions": len(train),
        "mean_item_degree": float(
            degree_values.mean()
        ),
        "std_item_degree": float(
            degree_values.std(ddof=0)
        ),
        "item_degree_cv": (
            coefficient_of_variation(
                degree_values
            )
        ),
        "item_degree_gini": (
            gini_coefficient(
                degree_values
            )
        ),
        "min_item_degree": int(
            degree_values.min()
        ),
        "median_item_degree": float(
            np.median(degree_values)
        ),
        "max_item_degree": int(
            degree_values.max()
        ),
        "p90_item_degree": float(
            np.percentile(
                degree_values,
                90,
            )
        ),
        "zero_degree_items": int(
            (degree_values == 0).sum()
        ),
    }

    item_degree["scenario"] = scenario
    item_degree["alpha"] = alpha

    return item_degree, summary


def plot_degree_distributions(
    all_degree_frames: list[pd.DataFrame],
) -> None:
    plt.figure(figsize=(8, 5))

    for degree_frame in all_degree_frames:
        scenario = (
            degree_frame["scenario"]
            .iloc[0]
        )

        sorted_degrees = np.sort(
            degree_frame["degree"]
            .to_numpy()
        )[::-1]

        item_ranks = np.arange(
            1,
            len(sorted_degrees) + 1,
        )

        plt.plot(
            item_ranks,
            sorted_degrees,
            label=scenario.capitalize(),
        )

    plt.xlabel("Item rank")
    plt.ylabel("Training item degree")
    plt.title(
        "Synthetic Training Item-Degree Distributions"
    )
    plt.legend()
    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_item_degree_distributions.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(f"Saved figure: {output_path}")


def main() -> None:
    summaries = []
    all_degree_frames = []

    for scenario, alpha in SCENARIOS.items():
        interactions = (
            generate_synthetic_interactions(
                alpha=alpha,
                seed=RANDOM_SEED,
            )
        )

        interaction_path = (
            DATA_DIR
            / (
                f"{scenario}_heterogeneity_"
                "interactions.csv"
            )
        )

        interactions.to_csv(
            interaction_path,
            index=False,
        )

        train, validation, test = (
            sequential_leave_one_out(
                interactions
            )
        )

        item_degree, summary = (
            calculate_degree_statistics(
                train=train,
                scenario=scenario,
                alpha=alpha,
            )
        )

        degree_path = (
            DATA_DIR
            / (
                f"{scenario}_heterogeneity_"
                "item_degrees.csv"
            )
        )

        item_degree.to_csv(
            degree_path,
            index=False,
        )

        summaries.append(summary)
        all_degree_frames.append(
            item_degree
        )

        print(f"\nScenario: {scenario}")
        print(f"Alpha: {alpha}")
        print(
            f"Generation seed: "
            f"{RANDOM_SEED}"
        )
        print(
            f"Total interactions: "
            f"{len(interactions)}"
        )
        print(
            f"Training interactions: "
            f"{len(train)}"
        )
        print(
            f"Validation interactions: "
            f"{len(validation)}"
        )
        print(
            f"Test interactions: "
            f"{len(test)}"
        )
        print(
            "Training item-degree CV: "
            f"{summary['item_degree_cv']:.4f}"
        )
        print(
            "Training item-degree Gini: "
            f"{summary['item_degree_gini']:.4f}"
        )
        print(
            "Maximum training item degree: "
            f"{summary['max_item_degree']}"
        )
        print(
            f"Saved full interactions: "
            f"{interaction_path}"
        )
        print(
            f"Saved training degrees: "
            f"{degree_path}"
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        RESULTS_DIR
        / (
            "simulation_degree_"
            "heterogeneity_summary.csv"
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    plot_degree_distributions(
        all_degree_frames
    )

    print("\nSimulation summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()