from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

SCENARIO_ORDER = ["low", "medium", "high"]
DISPLAY_NAMES = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}


def load_data():
    heterogeneity = pd.read_csv(
        RESULTS_DIR / "simulation_degree_heterogeneity_summary.csv"
    )

    lightgcn = pd.read_csv(
        RESULTS_DIR / "simulation_lightgcn_summary.csv"
    )

    composition = pd.read_csv(
        RESULTS_DIR / "simulation_test_item_composition.csv"
    )

    for df in [heterogeneity, lightgcn, composition]:
        df["scenario"] = pd.Categorical(
            df["scenario"],
            categories=SCENARIO_ORDER,
            ordered=True,
        )
        df.sort_values("scenario", inplace=True)

    return heterogeneity, lightgcn, composition


def plot_degree_heterogeneity(heterogeneity):
    labels = [
        DISPLAY_NAMES[str(value)]
        for value in heterogeneity["scenario"]
    ]

    plt.figure(figsize=(6, 4))

    plt.bar(
        labels,
        heterogeneity["item_degree_cv"],
    )

    plt.xlabel("Simulation scenario")
    plt.ylabel("Item degree coefficient of variation")
    plt.title("Controlled Degree Heterogeneity")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_degree_cv_by_scenario.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_grouped_recall(lightgcn):
    labels = [
        DISPLAY_NAMES[str(value)]
        for value in lightgcn["scenario"]
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(7, 4.5))

    plt.bar(
        x - width,
        lightgcn["tail_recall20_mean"],
        width,
        label="Tail",
    )

    plt.bar(
        x,
        lightgcn["medium_recall20_mean"],
        width,
        label="Medium",
    )

    plt.bar(
        x + width,
        lightgcn["head_recall20_mean"],
        width,
        label="Head",
    )

    plt.xticks(x, labels)
    plt.xlabel("Simulation scenario")
    plt.ylabel("Recall@20")
    plt.title("Grouped Recall@20 under Controlled Heterogeneity")
    plt.legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_grouped_recall20.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_head_tail_gap(lightgcn):
    labels = [
        DISPLAY_NAMES[str(value)]
        for value in lightgcn["scenario"]
    ]

    means = lightgcn["head_tail_gap_mean"]
    stds = lightgcn["head_tail_gap_std"]

    plt.figure(figsize=(6, 4))

    plt.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
    )

    plt.xlabel("Simulation scenario")
    plt.ylabel("Head-tail Recall@20 gap")
    plt.title("Head-tail Performance Gap")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_head_tail_gap.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_degree_norm_correlation(lightgcn):
    labels = [
        DISPLAY_NAMES[str(value)]
        for value in lightgcn["scenario"]
    ]

    means = lightgcn["degree_norm_correlation_mean"]
    stds = lightgcn["degree_norm_correlation_std"]

    plt.figure(figsize=(6, 4))

    plt.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
    )

    plt.xlabel("Simulation scenario")
    plt.ylabel("Degree-norm correlation")
    plt.title("Item Degree and Embedding Norm Association")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_degree_norm_correlation.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_test_item_composition(composition):
    labels = [
        DISPLAY_NAMES[str(value)]
        for value in composition["scenario"]
    ]

    x = np.arange(len(labels))

    tail = composition["tail_test_share"]
    medium = composition["medium_test_share"]
    head = composition["head_test_share"]

    plt.figure(figsize=(7, 4.5))

    plt.bar(
        x,
        tail,
        label="Tail",
    )

    plt.bar(
        x,
        medium,
        bottom=tail,
        label="Medium",
    )

    plt.bar(
        x,
        head,
        bottom=tail + medium,
        label="Head",
    )

    plt.xticks(x, labels)
    plt.xlabel("Simulation scenario")
    plt.ylabel("Share of test interactions")
    plt.title("Test-item Composition by Scenario")
    plt.legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "simulation_test_item_composition.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    heterogeneity, lightgcn, composition = load_data()

    plot_degree_heterogeneity(heterogeneity)
    plot_grouped_recall(lightgcn)
    plot_head_tail_gap(lightgcn)
    plot_degree_norm_correlation(lightgcn)
    plot_test_item_composition(composition)


if __name__ == "__main__":
    main()