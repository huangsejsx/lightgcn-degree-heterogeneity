from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lightgcn import (
    LightGCN,
    build_norm_adj,
    build_user_items,
    ndcg_at_k,
    recall_at_k,
    set_seed,
    train_one_epoch,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "amazon_beauty_3core_warm"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "amazon_beauty_3core_warm"

TOP_K = [10, 20]
GROUP_ORDER = {"tail": 0, "medium": 1, "head": 2, "cold_start": 3, "unknown": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LightGCN on Amazon Beauty warm-start data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Processed Amazon Beauty warm-start data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for Amazon Beauty LightGCN results.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for training. CPU is safest for this small dataset.",
    )
    return parser.parse_args()


def load_group_maps(data_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    user_degrees = pd.read_csv(data_dir / "user_degrees.csv")
    item_degrees = pd.read_csv(data_dir / "item_degrees.csv")
    return (
        dict(zip(user_degrees["user_idx"], user_degrees["degree_group"])),
        dict(zip(item_degrees["item_idx"], item_degrees["degree_group"])),
    )


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    finite_count = int(np.isfinite(scores).sum())
    top_count = min(k, finite_count)
    if top_count == 0:
        return []
    ranked = np.argpartition(-scores, top_count - 1)[:top_count]
    ranked = ranked[np.argsort(-scores[ranked])]
    return ranked.tolist()


def evaluate_lightgcn(
    model: LightGCN,
    test: pd.DataFrame,
    train_user_items: dict[int, set[int]],
    candidate_items: set[int],
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    max_k = max(TOP_K)
    records = []
    candidate_mask = None

    with torch.no_grad():
        user_emb, item_emb = model.propagate()
        item_emb_t = item_emb.t()

        all_item_ids = np.arange(item_emb.shape[0])
        candidate_mask = np.ones(item_emb.shape[0], dtype=bool)
        candidate_mask[:] = False
        candidate_mask[list(candidate_items)] = True

        for _, row in test.iterrows():
            user = int(row["user_idx"])
            true_item = int(row["item_idx"])
            scores = torch.matmul(user_emb[user], item_emb_t).cpu().numpy()

            # Match the baseline setting: rank only items observed in the training graph.
            scores[~candidate_mask] = -np.inf

            seen_items = train_user_items.get(user, set())
            if seen_items:
                scores[list(seen_items)] = -np.inf

            ranked_items = top_k_from_scores(scores, max_k)
            user_group = user_group_map.get(user, "unknown")
            item_group = item_group_map.get(true_item, "cold_start")

            for k in TOP_K:
                records.append(
                    {
                        "dataset": "Amazon Beauty 3-core warm-start",
                        "model": "LightGCN",
                        "run": f"seed_{seed}",
                        "seed": seed,
                        "user_idx": user,
                        "item_idx": true_item,
                        "user_group": user_group,
                        "item_group": item_group,
                        "K": k,
                        "Recall": recall_at_k(ranked_items, true_item, k),
                        "NDCG": ndcg_at_k(ranked_items, true_item, k),
                    }
                )

    return pd.DataFrame(records)


def summarise_overall(per_user: pd.DataFrame) -> pd.DataFrame:
    return (
        per_user.groupby(["dataset", "model", "run", "seed", "K"], as_index=False)
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
        .sort_values(["seed", "K"])
    )


def summarise_grouped(per_user: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        per_user.groupby(
            ["dataset", "model", "run", "seed", group_col, "K"], as_index=False
        )
        .agg(Recall=("Recall", "mean"), NDCG=("NDCG", "mean"), n=("Recall", "count"))
    )
    return summary.assign(
        group_order=summary[group_col].map(GROUP_ORDER).fillna(99)
    ).sort_values(["seed", "K", "group_order", group_col]).drop(columns=["group_order"])


def summarise_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        per_seed.groupby(["dataset", "model", "K"], as_index=False)
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["K"])
    )


def run_single_seed(
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    norm_adj: torch.Tensor,
    train_user_items: dict[int, set[int]],
    candidate_items: set[int],
    train_edges_original: np.ndarray,
    num_users: int,
    num_items: int,
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"Amazon Beauty LightGCN | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_edges = train_edges_original.copy()

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_rows = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_edges=train_edges,
            train_user_items=train_user_items,
            num_items=num_items,
            device=device,
        )
        loss_rows.append({"seed": seed, "epoch": epoch, "train_bpr_loss": loss})

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Loss: {loss:.6f}")

    per_user = evaluate_lightgcn(
        model=model,
        test=test,
        train_user_items=train_user_items,
        candidate_items=candidate_items,
        user_group_map=user_group_map,
        item_group_map=item_group_map,
        seed=seed,
        device=device,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(
        args.results_dir / f"amazon_lightgcn_per_user_metrics_seed_{seed}.csv",
        index=False,
    )
    pd.DataFrame(loss_rows).to_csv(
        args.results_dir / f"amazon_lightgcn_loss_seed_{seed}.csv",
        index=False,
    )

    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate()

    np.save(
        args.results_dir / f"amazon_lightgcn_user_embeddings_seed_{seed}.npy",
        user_embeddings.cpu().numpy(),
    )
    np.save(
        args.results_dir / f"amazon_lightgcn_item_embeddings_seed_{seed}.npy",
        item_embeddings.cpu().numpy(),
    )

    overall = summarise_overall(per_user)
    print("\nSeed test metrics:")
    print(overall.to_string(index=False))
    return per_user


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    user_group_map, item_group_map = load_group_maps(args.data_dir)

    num_users = int(max(train["user_idx"].max(), test["user_idx"].max()) + 1)
    num_items = int(max(train["item_idx"].max(), test["item_idx"].max()) + 1)
    candidate_items = set(map(int, train["item_idx"].unique()))

    print("\nAmazon Beauty LightGCN evaluation")
    print("=" * 45)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Users: {num_users}")
    print(f"Items in index space: {num_items}")
    print(f"Training catalog items: {len(candidate_items)}")
    print(f"Training interactions: {len(train)}")
    print(f"Warm-start test interactions: {len(test)}")
    print(f"Epochs per seed: {args.epochs}")
    print(f"Seeds: {args.seeds}")

    norm_adj = build_norm_adj(
        train=train,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )
    train_user_items = build_user_items(train)
    train_edges_original = train[["user_idx", "item_idx"]].to_numpy()

    all_per_user = []
    for seed in args.seeds:
        all_per_user.append(
            run_single_seed(
                seed=seed,
                train=train,
                test=test,
                norm_adj=norm_adj,
                train_user_items=train_user_items,
                candidate_items=candidate_items,
                train_edges_original=train_edges_original,
                num_users=num_users,
                num_items=num_items,
                user_group_map=user_group_map,
                item_group_map=item_group_map,
                args=args,
                device=device,
            )
        )

    per_user_all = pd.concat(all_per_user, ignore_index=True)
    overall_by_seed = summarise_overall(per_user_all)
    item_group_by_seed = summarise_grouped(per_user_all, "item_group")
    user_group_by_seed = summarise_grouped(per_user_all, "user_group")
    overall_summary = summarise_across_seeds(overall_by_seed)

    per_user_all.to_csv(
        args.results_dir / "amazon_lightgcn_per_user_metrics_all_seeds.csv",
        index=False,
    )
    overall_by_seed.to_csv(
        args.results_dir / "amazon_lightgcn_overall_metrics_by_seed.csv",
        index=False,
    )
    item_group_by_seed.to_csv(
        args.results_dir / "amazon_lightgcn_item_group_metrics_by_seed.csv",
        index=False,
    )
    user_group_by_seed.to_csv(
        args.results_dir / "amazon_lightgcn_user_group_metrics_by_seed.csv",
        index=False,
    )
    overall_summary.to_csv(
        args.results_dir / "amazon_lightgcn_overall_summary.csv",
        index=False,
    )

    print("\nAmazon Beauty LightGCN overall summary:")
    print(overall_summary.to_string(index=False))
    print(f"\nSaved LightGCN results to: {args.results_dir}")


if __name__ == "__main__":
    main()
