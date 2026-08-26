from pathlib import Path

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

from run_simulation_lightgcn import (
    temporal_leave_one_out,
    evaluate_grouped_recall,
    assign_item_groups,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SIM_DATA_DIR = ROOT_DIR / "data" / "simulated"
RESULTS_DIR = ROOT_DIR / "results"

N_USERS = 1000
N_ITEMS = 600
EMBED_DIM = 64
N_LAYERS = 2
LEARNING_RATE = 0.001

MAX_EPOCHS = 100
CHECKPOINTS = [1, 5, 10, 20, 30, 50, 75, 100]

SCENARIOS = ["low", "medium", "high"]
SEED = 42


def run_convergence_experiment(
    scenario: str,
    device: torch.device,
) -> list[dict]:
    print("\n" + "=" * 70)
    print(f"Convergence experiment | Scenario: {scenario}")
    print("=" * 70)

    set_seed(SEED)

    interaction_path = (
        SIM_DATA_DIR
        / f"{scenario}_heterogeneity_interactions.csv"
    )

    interactions = pd.read_csv(interaction_path)

    train, validation, _ = temporal_leave_one_out(
        interactions
    )

    item_groups, _, _, _ = assign_item_groups(train)

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

    results = []

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=N_ITEMS,
            device=device,
        )

        if epoch in CHECKPOINTS:
            validation_results = evaluate_grouped_recall(
                model=model,
                test=validation,
                train_user_items=train_user_items,
                item_groups=item_groups,
                device=device,
            )

            result = {
                "scenario": scenario,
                "seed": SEED,
                "epoch": epoch,
                "training_bpr_loss": loss,
                "validation_recall20": (
                    validation_results["overall_recall20"]
                ),
                "validation_tail_recall20": (
                    validation_results["tail_recall20"]
                ),
                "validation_medium_recall20": (
                    validation_results["medium_recall20"]
                ),
                "validation_head_recall20": (
                    validation_results["head_recall20"]
                ),
                "validation_head_tail_gap": (
                    validation_results["head_tail_gap"]
                ),
            }

            results.append(result)

            print(
                f"Epoch {epoch:03d} "
                f"| Loss: {loss:.4f} "
                f"| Val Recall@20: "
                f"{result['validation_recall20']:.4f} "
                f"| Tail: "
                f"{result['validation_tail_recall20']:.4f} "
                f"| Medium: "
                f"{result['validation_medium_recall20']:.4f} "
                f"| Head: "
                f"{result['validation_head_recall20']:.4f}"
            )

    return results


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    all_results = []

    for scenario in SCENARIOS:
        scenario_results = run_convergence_experiment(
            scenario=scenario,
            device=device,
        )

        all_results.extend(scenario_results)

    results_df = pd.DataFrame(all_results)

    output_path = (
        RESULTS_DIR
        / "simulation_validation_convergence.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Simulation convergence results")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()