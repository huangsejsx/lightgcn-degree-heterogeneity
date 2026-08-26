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
    evaluate,
    set_seed,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

EMBED_DIM = 64
N_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 100
SEEDS = [42, 43, 44]


def run_single_seed(
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    norm_adj: torch.Tensor,
    train_user_items: dict,
    train_edges_original: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"MovieLens LightGCN | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_edges = train_edges_original.copy()

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_rows = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=num_items,
            device=device,
        )

        loss_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_bpr_loss": loss,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss: {loss:.6f}"
            )

    metrics = evaluate(
        model=model,
        test=test,
        train_user_items=train_user_items,
        device=device,
    )

    metrics["seed"] = seed
    metrics["epochs"] = EPOCHS

    metrics_path = (
        RESULTS_DIR
        / f"lightgcn_metrics_seed_{seed}.csv"
    )
    metrics.to_csv(metrics_path, index=False)

    loss_path = (
        RESULTS_DIR
        / f"lightgcn_loss_seed_{seed}.csv"
    )
    pd.DataFrame(loss_rows).to_csv(loss_path, index=False)

    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate()

    np.save(
        RESULTS_DIR
        / f"lightgcn_user_embeddings_seed_{seed}.npy",
        user_embeddings.cpu().numpy(),
    )

    np.save(
        RESULTS_DIR
        / f"lightgcn_item_embeddings_seed_{seed}.npy",
        item_embeddings.cpu().numpy(),
    )

    print("\nTest metrics:")
    print(metrics.to_string(index=False))
    print(f"Saved: {metrics_path}")

    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    num_users = int(
        max(
            train["user_idx"].max(),
            test["user_idx"].max(),
        )
        + 1
    )

    num_items = int(
        max(
            train["item_idx"].max(),
            test["item_idx"].max(),
        )
        + 1
    )

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Training interactions: {len(train)}")
    print(f"Test interactions: {len(test)}")
    print(f"Epochs per seed: {EPOCHS}")
    print(f"Seeds: {SEEDS}")

    print("\nBuilding normalised adjacency matrix...")

    norm_adj = build_norm_adj(
        train=train,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )

    train_user_items = build_user_items(train)

    train_edges_original = train[
        ["user_idx", "item_idx"]
    ].to_numpy()

    all_metrics = []

    for seed in SEEDS:
        seed_metrics = run_single_seed(
            seed=seed,
            train=train,
            test=test,
            norm_adj=norm_adj,
            train_user_items=train_user_items,
            train_edges_original=train_edges_original,
            num_users=num_users,
            num_items=num_items,
            device=device,
        )

        all_metrics.append(seed_metrics)

    raw_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    raw_output_path = (
        RESULTS_DIR
        / "lightgcn_movielens_results_by_seed.csv"
    )
    raw_metrics.to_csv(raw_output_path, index=False)

    summary = (
        raw_metrics.groupby(["model", "K"])
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )

    summary_output_path = (
        RESULTS_DIR
        / "lightgcn_movielens_summary.csv"
    )
    summary.to_csv(summary_output_path, index=False)

    print("\n" + "=" * 70)
    print("MovieLens multi-seed summary")
    print("=" * 70)
    print(summary.to_string(index=False))

    print(f"\nSaved raw results: {raw_output_path}")
    print(f"Saved summary: {summary_output_path}")


if __name__ == "__main__":
    main()