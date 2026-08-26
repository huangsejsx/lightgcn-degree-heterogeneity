from pathlib import Path
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/codex_mplconfig",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SIM_DATA_DIR = ROOT_DIR / "data" / "simulated"
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = RESULTS_DIR / "simulation_alpha_diagnostics"

SCENARIO_ORDER = ["low", "medium", "high"]
SCENARIOS = {
    "low": {
        "display_name": "Low",
        "alpha": 0.0,
        "alpha_label": "0.0",
        "file_label": "alpha_0_0",
    },
    "medium": {
        "display_name": "Medium",
        "alpha": 0.7,
        "alpha_label": "0.7",
        "file_label": "alpha_0_7",
    },
    "high": {
        "display_name": "High",
        "alpha": 1.3,
        "alpha_label": "1.3",
        "file_label": "alpha_1_3",
    },
}

GROUPS = ["tail", "medium", "head"]
GROUP_LABELS = {
    "tail": "Tail",
    "medium": "Medium",
    "head": "Head",
}
COLORS = {
    "tail": "#4C78A8",
    "medium": "#F58518",
    "head": "#54A24B",
}


def set_common_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    lightgcn = pd.read_csv(
        RESULTS_DIR / "simulation_lightgcn_results_by_seed.csv"
    )
    composition = pd.read_csv(
        RESULTS_DIR / "simulation_test_item_composition.csv"
    )

    return lightgcn, composition


def format_alpha_title(scenario: str) -> str:
    config = SCENARIOS[scenario]
    return (
        f"{config['display_name']} heterogeneity "
        f"(alpha = {config['alpha_label']})"
    )


def save_current_figure(
    scenario: str,
    figure_type: str,
) -> Path:
    output_path = (
        OUT_DIR
        / f"{SCENARIOS[scenario]['file_label']}_{figure_type}.png"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_degree_distribution(scenario: str) -> Path:
    degree_path = (
        SIM_DATA_DIR / f"{scenario}_heterogeneity_item_degrees.csv"
    )
    item_degrees = pd.read_csv(degree_path)
    sorted_degrees = np.sort(
        item_degrees["degree"].to_numpy()
    )[::-1]
    ranks = np.arange(1, len(sorted_degrees) + 1)

    plt.figure(figsize=(5.8, 3.8))
    plt.plot(ranks, sorted_degrees, color="#4C78A8", linewidth=2)
    plt.xlabel("Item rank")
    plt.ylabel("Training item degree")
    plt.title(f"{format_alpha_title(scenario)}: item-degree distribution")
    plt.grid(axis="y", alpha=0.25)

    return save_current_figure(scenario, "degree_distribution")


def plot_test_composition(
    scenario: str,
    composition: pd.DataFrame,
) -> Path:
    row = composition[composition["scenario"] == scenario].iloc[0]
    shares = [
        row[f"{group}_test_share"]
        for group in GROUPS
    ]
    counts = [
        int(row[f"{group}_test_count"])
        for group in GROUPS
    ]

    labels = [GROUP_LABELS[group] for group in GROUPS]
    colors = [COLORS[group] for group in GROUPS]

    plt.figure(figsize=(5.8, 3.8))
    bars = plt.bar(labels, shares, color=colors)
    plt.ylim(0, 1)
    plt.xlabel("Held-out item group")
    plt.ylabel("Share of test interactions")
    plt.title(f"{format_alpha_title(scenario)}: test-item composition")
    plt.grid(axis="y", alpha=0.25)

    for bar, share, count in zip(bars, shares, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{share:.1%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return save_current_figure(scenario, "test_composition")


def plot_grouped_recall(
    scenario: str,
    lightgcn: pd.DataFrame,
) -> Path:
    scenario_results = lightgcn[lightgcn["scenario"] == scenario]
    means = []
    stds = []

    for group in GROUPS:
        values = scenario_results[f"{group}_recall20"].to_numpy()
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)))

    labels = [GROUP_LABELS[group] for group in GROUPS]
    colors = [COLORS[group] for group in GROUPS]

    plt.figure(figsize=(5.8, 3.8))
    bars = plt.bar(
        labels,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
    )
    plt.ylim(0, max(0.1, max(means) * 1.25))
    plt.xlabel("Held-out item group")
    plt.ylabel("Recall@20")
    plt.title(f"{format_alpha_title(scenario)}: grouped Recall@20")
    plt.grid(axis="y", alpha=0.25)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.006, max(means) * 0.025),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return save_current_figure(scenario, "grouped_recall20")


def plot_head_tail_gap_by_seed(
    scenario: str,
    lightgcn: pd.DataFrame,
) -> Path:
    scenario_results = lightgcn[lightgcn["scenario"] == scenario].copy()
    scenario_results = scenario_results.sort_values("seed")
    mean_gap = scenario_results["head_tail_gap"].mean()

    labels = scenario_results["seed"].astype(str).tolist()
    values = scenario_results["head_tail_gap"].to_numpy()

    plt.figure(figsize=(5.8, 3.8))
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.axhline(
        mean_gap,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_gap:.3f}",
    )
    plt.ylim(0, max(0.1, values.max() * 1.25))
    plt.xlabel("LightGCN optimisation seed")
    plt.ylabel("Head-tail Recall@20 gap")
    plt.title(f"{format_alpha_title(scenario)}: head-tail gap by seed")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.004, values.max() * 0.025),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return save_current_figure(scenario, "head_tail_gap_by_seed")


def plot_degree_norm_correlation_by_seed(
    scenario: str,
    lightgcn: pd.DataFrame,
) -> Path:
    scenario_results = lightgcn[lightgcn["scenario"] == scenario].copy()
    scenario_results = scenario_results.sort_values("seed")
    mean_corr = scenario_results["degree_norm_correlation"].mean()

    labels = scenario_results["seed"].astype(str).tolist()
    values = scenario_results["degree_norm_correlation"].to_numpy()

    plt.figure(figsize=(5.8, 3.8))
    bars = plt.bar(labels, values, color="#54A24B")
    plt.axhline(
        mean_corr,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_corr:.3f}",
    )
    plt.ylim(0, 1)
    plt.xlabel("LightGCN optimisation seed")
    plt.ylabel("Pearson correlation")
    plt.title(
        f"{format_alpha_title(scenario)}: item degree vs. embedding norm"
    )
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return save_current_figure(
        scenario,
        "degree_norm_correlation_by_seed",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_common_style()
    lightgcn, composition = load_inputs()
    manifest_rows = []

    plotters = [
        (
            "degree_distribution",
            lambda scenario: plot_degree_distribution(scenario),
        ),
        (
            "test_composition",
            lambda scenario: plot_test_composition(scenario, composition),
        ),
        (
            "grouped_recall20",
            lambda scenario: plot_grouped_recall(scenario, lightgcn),
        ),
        (
            "head_tail_gap_by_seed",
            lambda scenario: plot_head_tail_gap_by_seed(
                scenario,
                lightgcn,
            ),
        ),
        (
            "degree_norm_correlation_by_seed",
            lambda scenario: plot_degree_norm_correlation_by_seed(
                scenario,
                lightgcn,
            ),
        ),
    ]

    for scenario in SCENARIO_ORDER:
        for figure_type, plotter in plotters:
            output_path = plotter(scenario)
            manifest_rows.append(
                {
                    "scenario": scenario,
                    "alpha": SCENARIOS[scenario]["alpha"],
                    "figure_type": figure_type,
                    "path": str(
                        output_path.relative_to(ROOT_DIR)
                    ),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "simulation_alpha_diagnostics_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
