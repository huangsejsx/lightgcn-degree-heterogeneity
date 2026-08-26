from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

sys.path.append(str(SRC_DIR))

from lightgcn import (
    LightGCN,
    build_norm_adj,
    build_user_items,
    set_seed,
    train_one_epoch,
)

EMBED_DIM = 64
N_LAYERS = 2
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
SEED = 42
TOP_K = 20
CHECKPOINT_EPOCHS = {1, 5, 10, 20, 30, 50, 75, 100}


def evaluate_validation(
    model: LightGCN,
    validation: pd.DataFrame,
    train_user_items: dict,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    recall_values = []
    ndcg_values = []

    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate()
        item_embeddings_t = item_embeddings.t()

        for row in validation.itertuples(index=False):
            user = int(row.user_idx)
            true_item = int(row.item_idx)

            scores = torch.matmul(
                user_embeddings[user],
                item_embeddings_t,
            ).cpu().numpy()

            seen_items = train_user_items.get(user, set())

            if seen_items:
                scores[list(seen_items)] = -np.inf

            candidate_count = min(TOP_K, len(scores))

            top_items = np.argpartition(
                -scores,
                candidate_count - 1,
            )[:candidate_count]

            ranked_items = top_items[
                np.argsort(-scores[top_items])
            ]

            matching_positions = np.where(
                ranked_items == true_item
            )[0]

            if len(matching_positions) == 0:
                recall_values.append(0.0)
                ndcg_values.append(0.0)
            else:
                rank = int(matching_positions[0]) + 1
                recall_values.append(1.0)
                ndcg_values.append(
                    1.0 / np.log2(rank + 1)
                )

    return (
        float(np.mean(recall_values)),
        float(np.mean(ndcg_values)),
    )


def plot_training_loss(history: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        history["epoch"],
        history["train_bpr_loss"],
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean BPR training loss")
    plt.title("LightGCN Training Loss")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "lightgcn_training_loss_by_epoch.png",
        dpi=300,
    )
    plt.close()


def plot_validation_recall(checkpoints: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        checkpoints["epoch"],
        checkpoints["validation_recall20"],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Recall@20")
    plt.title("LightGCN Validation Recall@20 by Epoch")
    plt.xticks(checkpoints["epoch"])
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR
        / "lightgcn_validation_recall20_by_epoch.png",
        dpi=300,
    )
    plt.close()


def plot_validation_ndcg(checkpoints: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        checkpoints["epoch"],
        checkpoints["validation_ndcg20"],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation NDCG@20")
    plt.title("LightGCN Validation NDCG@20 by Epoch")
    plt.xticks(checkpoints["epoch"])
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR
        / "lightgcn_validation_ndcg20_by_epoch.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    num_users = int(
        max(
            train["user_idx"].max(),
            validation["user_idx"].max(),
            test["user_idx"].max(),
        )
        + 1
    )

    num_items = int(
        max(
            train["item_idx"].max(),
            validation["item_idx"].max(),
            test["item_idx"].max(),
        )
        + 1
    )

    print(f"Device: {device}")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Training interactions: {len(train)}")
    print(f"Validation interactions: {len(validation)}")

    print("Building normalized adjacency matrix...")

    norm_adj = build_norm_adj(
        train=train,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )

    train_user_items = build_user_items(train)

    train_edges = train[
        ["user_idx", "item_idx"]
    ].to_numpy()

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
    checkpoint_rows = []

    print("Starting convergence experiment...")

    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=num_items,
            device=device,
        )

        loss_rows.append(
            {
                "epoch": epoch,
                "train_bpr_loss": training_loss,
            }
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} "
            f"| Loss: {training_loss:.6f}"
        )

        if epoch in CHECKPOINT_EPOCHS:
            validation_recall20, validation_ndcg20 = (
                evaluate_validation(
                    model=model,
                    validation=validation,
                    train_user_items=train_user_items,
                    device=device,
                )
            )

            checkpoint_rows.append(
                {
                    "epoch": epoch,
                    "train_bpr_loss": training_loss,
                    "validation_recall20": (
                        validation_recall20
                    ),
                    "validation_ndcg20": (
                        validation_ndcg20
                    ),
                }
            )

            print(
                f"Checkpoint {epoch:03d} "
                f"| Validation Recall@20: "
                f"{validation_recall20:.6f} "
                f"| Validation NDCG@20: "
                f"{validation_ndcg20:.6f}"
            )

    loss_history = pd.DataFrame(loss_rows)
    checkpoint_history = pd.DataFrame(checkpoint_rows)

    loss_path = (
        RESULTS_DIR
        / "lightgcn_training_loss_history.csv"
    )

    checkpoint_path = (
        RESULTS_DIR
        / "lightgcn_convergence.csv"
    )

    loss_history.to_csv(loss_path, index=False)
    checkpoint_history.to_csv(
        checkpoint_path,
        index=False,
    )

    plot_training_loss(loss_history)
    plot_validation_recall(checkpoint_history)
    plot_validation_ndcg(checkpoint_history)

    best_recall_row = checkpoint_history.loc[
        checkpoint_history[
            "validation_recall20"
        ].idxmax()
    ]

    best_ndcg_row = checkpoint_history.loc[
        checkpoint_history[
            "validation_ndcg20"
        ].idxmax()
    ]

    print("\nConvergence checkpoints:")
    print(checkpoint_history.to_string(index=False))

    print(
        "\nBest validation Recall@20:"
        f" epoch={int(best_recall_row['epoch'])},"
        f" Recall@20="
        f"{best_recall_row['validation_recall20']:.6f}"
    )

    print(
        "Best validation NDCG@20:"
        f" epoch={int(best_ndcg_row['epoch'])},"
        f" NDCG@20="
        f"{best_ndcg_row['validation_ndcg20']:.6f}"
    )

    print(f"\nSaved: {loss_path}")
    print(f"Saved: {checkpoint_path}")
    print(
        "Saved: "
        "results/lightgcn_training_loss_by_epoch.png"
    )
    print(
        "Saved: "
        "results/lightgcn_validation_recall20_by_epoch.png"
    )
    print(
        "Saved: "
        "results/lightgcn_validation_ndcg20_by_epoch.png"
    )


if __name__ == "__main__":
    main()
