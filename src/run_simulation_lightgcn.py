from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from lightgcn import (
    LightGCN,
    build_norm_adj,
    build_user_items,
    train_one_epoch,
    set_seed,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
SIM_DATA_DIR = ROOT_DIR / "data" / "simulated"
RESULTS_DIR = ROOT_DIR / "results"

N_USERS = 1000
N_ITEMS = 600
EMBED_DIM = 64
N_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 20
TOP_K = 20

SCENARIOS = ["low", "medium", "high"]
SEEDS = [42, 43, 44]


def temporal_leave_one_out(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = interactions.sort_values(
        ["user_id", "interaction_order"]
    ).copy()

    grouped = interactions.groupby("user_id")

    test = grouped.tail(1).copy()
    remaining = interactions.drop(test.index)

    validation = remaining.groupby("user_id").tail(1).copy()
    train = remaining.drop(validation.index).copy()

    rename_columns = {
        "user_id": "user_idx",
        "item_id": "item_idx",
    }

    train = train.rename(columns=rename_columns)
    validation = validation.rename(columns=rename_columns)
    test = test.rename(columns=rename_columns)

    return train, validation, test


def assign_item_groups(
    train: pd.DataFrame,
) -> tuple[dict[int, str], float, float, np.ndarray]:
    item_degree = (
        train.groupby("item_idx")
        .size()
        .reindex(range(N_ITEMS), fill_value=0)
        .to_numpy()
    )

    low_threshold = float(np.quantile(item_degree, 0.20))
    high_threshold = float(np.quantile(item_degree, 0.80))

    item_groups = {}

    for item_id, degree in enumerate(item_degree):
        if degree <= low_threshold:
            group = "tail"
        elif degree >= high_threshold:
            group = "head"
        else:
            group = "medium"

        item_groups[item_id] = group

    return (
        item_groups,
        low_threshold,
        high_threshold,
        item_degree,
    )


def evaluate_grouped_recall(
    model: LightGCN,
    test: pd.DataFrame,
    train_user_items: dict,
    item_groups: dict[int, str],
    device: torch.device,
) -> dict:
    model.eval()

    overall_hits = []
    group_hits = {
        "tail": [],
        "medium": [],
        "head": [],
    }

    with torch.no_grad():
        user_emb, item_emb = model.propagate()
        item_emb_t = item_emb.t()

        for _, row in test.iterrows():
            user = int(row["user_idx"])
            true_item = int(row["item_idx"])

            scores = torch.matmul(
                user_emb[user],
                item_emb_t,
            ).cpu().numpy()

            seen_items = train_user_items.get(user, set())

            if seen_items:
                scores[list(seen_items)] = -np.inf

            candidate_count = min(TOP_K, len(scores))

            ranked_items = np.argpartition(
                -scores,
                candidate_count - 1,
            )[:candidate_count]

            ranked_items = ranked_items[
                np.argsort(-scores[ranked_items])
            ]

            hit = float(true_item in ranked_items)

            overall_hits.append(hit)

            group = item_groups[true_item]
            group_hits[group].append(hit)

    results = {
        "overall_recall20": float(np.mean(overall_hits)),
    }

    for group in ["tail", "medium", "head"]:
        hits = group_hits[group]

        results[f"{group}_recall20"] = (
            float(np.mean(hits)) if hits else np.nan
        )

        results[f"{group}_test_count"] = len(hits)

    results["head_tail_gap"] = (
        results["head_recall20"]
        - results["tail_recall20"]
    )

    return results


def calculate_embedding_statistics(
    model: LightGCN,
    item_degree: np.ndarray,
) -> dict:
    model.eval()

    with torch.no_grad():
        _, item_embeddings = model.propagate()

    item_embeddings = item_embeddings.cpu().numpy()
    item_norms = np.linalg.norm(item_embeddings, axis=1)

    correlation = np.corrcoef(
        item_degree,
        item_norms,
    )[0, 1]

    return {
        "degree_norm_correlation": float(correlation),
        "mean_item_embedding_norm": float(item_norms.mean()),
        "std_item_embedding_norm": float(item_norms.std()),
    }


def calculate_curvature_statistics(
    train: pd.DataFrame,
    item_degree: np.ndarray,
) -> dict:
    user_degree = (
        train.groupby("user_idx")
        .size()
        .reindex(range(N_USERS), fill_value=0)
        .to_numpy()
    )

    curvature_magnitudes = []

    for user, item in train[
        ["user_idx", "item_idx"]
    ].itertuples(index=False):
        curvature = 4 - user_degree[user] - item_degree[item]
        curvature_magnitudes.append(abs(curvature))

    curvature_magnitudes = np.asarray(
        curvature_magnitudes,
        dtype=float,
    )

    return {
        "mean_curvature_magnitude": float(
            curvature_magnitudes.mean()
        ),
        "std_curvature_magnitude": float(
            curvature_magnitudes.std()
        ),
    }


def run_single_experiment(
    scenario: str,
    seed: int,
    device: torch.device,
) -> dict:
    print("\n" + "=" * 70)
    print(f"Scenario: {scenario} | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    interaction_path = (
        SIM_DATA_DIR
        / f"{scenario}_heterogeneity_interactions.csv"
    )

    interactions = pd.read_csv(interaction_path)

    train, validation, test = temporal_leave_one_out(
        interactions
    )

    item_groups, low_threshold, high_threshold, item_degree = (
        assign_item_groups(train)
    )

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(validation)} | "
        f"Test: {len(test)}"
    )

    print(
        f"Item thresholds: "
        f"low <= {low_threshold:.1f}, "
        f"high >= {high_threshold:.1f}"
    )

    norm_adj = build_norm_adj(
        train=train,
        num_users=N_USERS,
        num_items=N_ITEMS,
        device=device,
    )

    train_user_items = build_user_items(train)

    train_edges = train[
        ["user_idx", "item_idx"]
    ].to_numpy()

    model = LightGCN(
        num_users=N_USERS,
        num_items=N_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=N_ITEMS,
            device=device,
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"| Loss: {loss:.4f}"
            )

    grouped_results = evaluate_grouped_recall(
        model=model,
        test=test,
        train_user_items=train_user_items,
        item_groups=item_groups,
        device=device,
    )

    embedding_results = calculate_embedding_statistics(
        model=model,
        item_degree=item_degree,
    )

    curvature_results = calculate_curvature_statistics(
        train=train,
        item_degree=item_degree,
    )

    result = {
        "scenario": scenario,
        "seed": seed,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "low_item_threshold": low_threshold,
        "high_item_threshold": high_threshold,
        **grouped_results,
        **embedding_results,
        **curvature_results,
    }

    print(
        f"Overall Recall@20: "
        f"{result['overall_recall20']:.4f}"
    )

    print(
        f"Tail / Medium / Head Recall@20: "
        f"{result['tail_recall20']:.4f} / "
        f"{result['medium_recall20']:.4f} / "
        f"{result['head_recall20']:.4f}"
    )

    print(
        f"Head-tail gap: "
        f"{result['head_tail_gap']:.4f}"
    )

    print(
        f"Degree-norm correlation: "
        f"{result['degree_norm_correlation']:.4f}"
    )

    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    all_results = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            result = run_single_experiment(
                scenario=scenario,
                seed=seed,
                device=device,
            )

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    raw_output_path = (
        RESULTS_DIR
        / "simulation_lightgcn_results_by_seed.csv"
    )

    results_df.to_csv(raw_output_path, index=False)

    metric_columns = [
        "overall_recall20",
        "tail_recall20",
        "medium_recall20",
        "head_recall20",
        "head_tail_gap",
        "degree_norm_correlation",
        "mean_item_embedding_norm",
        "mean_curvature_magnitude",
    ]

    summary_mean = (
        results_df.groupby("scenario")[metric_columns]
        .mean()
        .add_suffix("_mean")
    )

    summary_std = (
        results_df.groupby("scenario")[metric_columns]
        .std()
        .add_suffix("_std")
    )

    summary_df = pd.concat(
        [summary_mean, summary_std],
        axis=1,
    ).reset_index()

    scenario_order = pd.Categorical(
        summary_df["scenario"],
        categories=["low", "medium", "high"],
        ordered=True,
    )

    summary_df["scenario"] = scenario_order
    summary_df = summary_df.sort_values("scenario")

    summary_output_path = (
        RESULTS_DIR
        / "simulation_lightgcn_summary.csv"
    )

    summary_df.to_csv(summary_output_path, index=False)

    print("\n" + "=" * 70)
    print("Simulation LightGCN summary")
    print("=" * 70)

    selected_columns = [
        "scenario",
        "overall_recall20_mean",
        "tail_recall20_mean",
        "medium_recall20_mean",
        "head_recall20_mean",
        "head_tail_gap_mean",
        "degree_norm_correlation_mean",
    ]

    print(
        summary_df[selected_columns].to_string(
            index=False
        )
    )

    print(f"\nSaved raw results: {raw_output_path}")
    print(f"Saved summary: {summary_output_path}")


if __name__ == "__main__":
    main()