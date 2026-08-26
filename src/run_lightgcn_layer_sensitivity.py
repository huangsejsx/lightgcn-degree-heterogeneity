from argparse import ArgumentParser, Namespace
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import torch

from lightgcn import (
    LightGCN,
    build_norm_adj,
    build_user_items,
    set_seed,
    train_one_epoch,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = RESULTS_DIR / "layer_sensitivity"

EMBED_DIM = 64
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100
DEFAULT_LAYERS = [0, 1, 2, 3, 4]
DEFAULT_SEEDS = [42, 43, 44]
TOP_K = [10, 20]

GROUP_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "unknown": 3,
}


def recall_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    return float(true_item in ranked_items[:k])


def ndcg_at_k(ranked_items: list[int], true_item: int, k: int) -> float:
    top_k = ranked_items[:k]

    if true_item not in top_k:
        return 0.0

    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    finite_count = int(np.isfinite(scores).sum())
    top_count = min(k, finite_count)

    if top_count == 0:
        return []

    ranked_items = np.argpartition(-scores, top_count - 1)[:top_count]
    ranked_items = ranked_items[np.argsort(-scores[ranked_items])]
    return ranked_items.tolist()


def evaluate_embeddings(
    n_layers: int,
    seed: int,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test: pd.DataFrame,
    train_user_items: dict[int, set[int]],
    user_group_map: dict[int, str],
    item_group_map: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_records = []
    user_group_records = []
    item_group_records = []
    max_k = max(TOP_K)

    for row in test.itertuples(index=False):
        user = int(row.user_idx)
        true_item = int(row.item_idx)

        scores = user_emb[user] @ item_emb.T
        seen_items = train_user_items.get(user, set())

        if seen_items:
            scores[list(seen_items)] = -np.inf

        ranked_items = top_k_from_scores(scores, max_k)
        user_group = user_group_map.get(user, "unknown")
        item_group = item_group_map.get(true_item, "unknown")

        for k in TOP_K:
            recall = recall_at_k(ranked_items, true_item, k)
            ndcg = ndcg_at_k(ranked_items, true_item, k)

            base_record = {
                "n_layers": n_layers,
                "seed": seed,
                "K": k,
                "Recall": recall,
                "NDCG": ndcg,
            }

            overall_records.append(base_record)
            user_group_records.append(
                {
                    **base_record,
                    "group": user_group,
                }
            )
            item_group_records.append(
                {
                    **base_record,
                    "group": item_group,
                }
            )

    overall = (
        pd.DataFrame(overall_records)
        .groupby(["n_layers", "seed", "K"], as_index=False)
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n_test_cases=("Recall", "size"),
        )
    )

    user_group = summarise_groups(
        pd.DataFrame(user_group_records)
    )
    item_group = summarise_groups(
        pd.DataFrame(item_group_records)
    )

    return overall, user_group, item_group


def summarise_groups(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["n_layers", "seed", "group", "K"],
            as_index=False,
        )
        .agg(
            Recall=("Recall", "mean"),
            NDCG=("NDCG", "mean"),
            n_test_cases=("Recall", "size"),
        )
    )


def analyse_embedding_correlations(
    n_layers: int,
    seed: int,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_degrees: pd.DataFrame,
    item_degrees: pd.DataFrame,
) -> pd.DataFrame:
    user_df = pd.DataFrame(
        {
            "user_idx": np.arange(user_emb.shape[0]),
            "embedding_norm": np.linalg.norm(user_emb, axis=1),
        }
    ).merge(user_degrees, on="user_idx", how="left")

    item_df = pd.DataFrame(
        {
            "item_idx": np.arange(item_emb.shape[0]),
            "embedding_norm": np.linalg.norm(item_emb, axis=1),
        }
    ).merge(item_degrees, on="item_idx", how="left")

    return pd.DataFrame(
        [
            {
                "n_layers": n_layers,
                "seed": seed,
                "node_type": "user",
                "degree_norm_correlation": user_df[
                    ["degree", "embedding_norm"]
                ].corr().iloc[0, 1],
            },
            {
                "n_layers": n_layers,
                "seed": seed,
                "node_type": "item",
                "degree_norm_correlation": item_df[
                    ["degree", "embedding_norm"]
                ].corr().iloc[0, 1],
            },
        ]
    )


def train_model(
    n_layers: int,
    seed: int,
    epochs: int,
    train_user_items: dict[int, set[int]],
    train_edges_original: np.ndarray,
    norm_adj: torch.Tensor,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    print("\n" + "=" * 70)
    print(f"Training LightGCN | layers={n_layers} | seed={seed}")
    print("=" * 70)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
        n_layers=n_layers,
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_edges = train_edges_original.copy()
    loss_rows = []

    for epoch in range(1, epochs + 1):
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
                "n_layers": n_layers,
                "seed": seed,
                "epoch": epoch,
                "train_bpr_loss": loss,
            }
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"layers={n_layers} seed={seed} "
                f"epoch={epoch:03d}/{epochs} loss={loss:.6f}"
            )

    with torch.no_grad():
        user_emb, item_emb = model.propagate()

    return (
        user_emb.cpu().numpy(),
        item_emb.cpu().numpy(),
        pd.DataFrame(loss_rows),
    )


def embedding_paths(n_layers: int, seed: int) -> tuple[Path, Path]:
    user_path = (
        OUT_DIR
        / f"lightgcn_user_embeddings_layers_{n_layers}_seed_{seed}.npy"
    )
    item_path = (
        OUT_DIR
        / f"lightgcn_item_embeddings_layers_{n_layers}_seed_{seed}.npy"
    )
    return user_path, item_path


def legacy_layer_two_paths(seed: int) -> tuple[Path, Path]:
    return (
        RESULTS_DIR / f"lightgcn_user_embeddings_seed_{seed}.npy",
        RESULTS_DIR / f"lightgcn_item_embeddings_seed_{seed}.npy",
    )


def load_cached_embeddings(
    n_layers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    user_path, item_path = embedding_paths(n_layers, seed)

    if user_path.exists() and item_path.exists():
        print(
            f"Using cached layer-sensitivity embeddings "
            f"for layers={n_layers}, seed={seed}."
        )
        return np.load(user_path), np.load(item_path)

    if n_layers == 2:
        legacy_user_path, legacy_item_path = legacy_layer_two_paths(seed)

        if legacy_user_path.exists() and legacy_item_path.exists():
            print(
                f"Using existing legacy K=2 embeddings "
                f"for seed={seed}."
            )
            return (
                np.load(legacy_user_path),
                np.load(legacy_item_path),
            )

    return None


def save_embeddings(
    n_layers: int,
    seed: int,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
) -> None:
    user_path, item_path = embedding_paths(n_layers, seed)
    np.save(user_path, user_emb)
    np.save(item_path, item_emb)


def summarise_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["n_layers", "K"], as_index=False)
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_seeds=("seed", "nunique"),
            n_test_cases=("n_test_cases", "first"),
        )
        .sort_values(["n_layers", "K"])
    )


def summarise_groups_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            group_order=df["group"].map(GROUP_ORDER).fillna(99)
        )
        .groupby(["n_layers", "group", "K"], as_index=False)
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_seeds=("seed", "nunique"),
            n_test_cases=("n_test_cases", "first"),
            group_order=("group_order", "first"),
        )
        .sort_values(["n_layers", "group_order", "K"])
        .drop(columns="group_order")
    )


def summarise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["n_layers", "node_type"], as_index=False)
        .agg(
            correlation_mean=("degree_norm_correlation", "mean"),
            correlation_std=("degree_norm_correlation", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["n_layers", "node_type"])
    )


def make_head_tail_gap(item_group_by_seed: pd.DataFrame) -> pd.DataFrame:
    recall20 = item_group_by_seed[item_group_by_seed["K"] == 20]
    wide = recall20.pivot_table(
        index=["n_layers", "seed"],
        columns="group",
        values="Recall",
    ).reset_index()

    if "high" not in wide or "low" not in wide:
        return pd.DataFrame()

    wide["head_tail_recall20_gap"] = wide["high"] - wide["low"]

    return (
        wide.groupby("n_layers", as_index=False)
        .agg(
            head_tail_gap_mean=(
                "head_tail_recall20_gap",
                "mean",
            ),
            head_tail_gap_std=(
                "head_tail_recall20_gap",
                "std",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("n_layers")
    )


def write_outputs(
    all_overall: list[pd.DataFrame],
    all_user_groups: list[pd.DataFrame],
    all_item_groups: list[pd.DataFrame],
    all_correlations: list[pd.DataFrame],
    all_loss_rows: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not all_overall:
        empty = pd.DataFrame()
        return empty, empty

    overall_by_seed = pd.concat(all_overall, ignore_index=True)
    user_group_by_seed = pd.concat(all_user_groups, ignore_index=True)
    item_group_by_seed = pd.concat(all_item_groups, ignore_index=True)
    correlation_by_seed = pd.concat(all_correlations, ignore_index=True)

    overall_summary = summarise_across_seeds(overall_by_seed)
    user_group_summary = summarise_groups_across_seeds(
        user_group_by_seed
    )
    item_group_summary = summarise_groups_across_seeds(
        item_group_by_seed
    )
    correlation_summary = summarise_correlations(correlation_by_seed)
    head_tail_gap = make_head_tail_gap(item_group_by_seed)

    overall_by_seed.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_results_by_seed.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_summary.csv",
        index=False,
    )
    user_group_by_seed.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_user_group_by_seed.csv",
        index=False,
    )
    user_group_summary.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_user_group_summary.csv",
        index=False,
    )
    item_group_by_seed.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_item_group_by_seed.csv",
        index=False,
    )
    item_group_summary.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_item_group_summary.csv",
        index=False,
    )
    correlation_by_seed.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_correlation_by_seed.csv",
        index=False,
    )
    correlation_summary.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_correlation_summary.csv",
        index=False,
    )
    head_tail_gap.to_csv(
        OUT_DIR / "lightgcn_layer_sensitivity_head_tail_gap.csv",
        index=False,
    )

    if all_loss_rows:
        pd.concat(all_loss_rows, ignore_index=True).to_csv(
            OUT_DIR / "lightgcn_layer_sensitivity_loss_history.csv",
            index=False,
        )

    return overall_summary, head_tail_gap


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Run LightGCN propagation-depth sensitivity on MovieLens."
        )
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Propagation layer counts to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Optimisation seeds to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs for newly trained runs.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain even when cached embeddings are available.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate cached embeddings and skip missing runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    user_degrees = pd.read_csv(DATA_DIR / "user_degrees.csv")
    item_degrees = pd.read_csv(DATA_DIR / "item_degrees.csv")

    num_users = int(
        max(train["user_idx"].max(), test["user_idx"].max()) + 1
    )
    num_items = int(
        max(train["item_idx"].max(), test["item_idx"].max()) + 1
    )

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Training interactions: {len(train)}")
    print(f"Test interactions: {len(test)}")
    print(f"Layers: {args.layers}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs for new runs: {args.epochs}")

    print("\nBuilding normalised adjacency matrix...")
    norm_adj = build_norm_adj(
        train=train,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )

    train_user_items = build_user_items(train)
    train_edges_original = train[["user_idx", "item_idx"]].to_numpy()
    user_group_map = dict(
        zip(user_degrees["user_idx"], user_degrees["degree_group"])
    )
    item_group_map = dict(
        zip(item_degrees["item_idx"], item_degrees["degree_group"])
    )

    all_overall = []
    all_user_groups = []
    all_item_groups = []
    all_correlations = []
    all_loss_rows = []

    for n_layers in args.layers:
        for seed in args.seeds:
            cached = None

            if not args.retrain:
                cached = load_cached_embeddings(n_layers, seed)

            if cached is None:
                if args.evaluate_only:
                    print(
                        f"Skipping missing cached run: "
                        f"layers={n_layers}, seed={seed}"
                    )
                    continue

                user_emb, item_emb, loss_rows = train_model(
                    n_layers=n_layers,
                    seed=seed,
                    epochs=args.epochs,
                    train_user_items=train_user_items,
                    train_edges_original=train_edges_original,
                    norm_adj=norm_adj,
                    num_users=num_users,
                    num_items=num_items,
                    device=device,
                )
                save_embeddings(
                    n_layers=n_layers,
                    seed=seed,
                    user_emb=user_emb,
                    item_emb=item_emb,
                )
                all_loss_rows.append(loss_rows)
            else:
                user_emb, item_emb = cached

            overall, user_group, item_group = evaluate_embeddings(
                n_layers=n_layers,
                seed=seed,
                user_emb=user_emb,
                item_emb=item_emb,
                test=test,
                train_user_items=train_user_items,
                user_group_map=user_group_map,
                item_group_map=item_group_map,
            )

            correlations = analyse_embedding_correlations(
                n_layers=n_layers,
                seed=seed,
                user_emb=user_emb,
                item_emb=item_emb,
                user_degrees=user_degrees,
                item_degrees=item_degrees,
            )

            all_overall.append(overall)
            all_user_groups.append(user_group)
            all_item_groups.append(item_group)
            all_correlations.append(correlations)

            print("\nOverall metrics:")
            print(overall.to_string(index=False))
            write_outputs(
                all_overall=all_overall,
                all_user_groups=all_user_groups,
                all_item_groups=all_item_groups,
                all_correlations=all_correlations,
                all_loss_rows=all_loss_rows,
            )

    overall_summary, head_tail_gap = write_outputs(
        all_overall=all_overall,
        all_user_groups=all_user_groups,
        all_item_groups=all_item_groups,
        all_correlations=all_correlations,
        all_loss_rows=all_loss_rows,
    )

    print("\n" + "=" * 70)
    print("Propagation-depth summary")
    print("=" * 70)
    if overall_summary.empty:
        print("No completed layer-sensitivity runs were available.")
    else:
        print(overall_summary.to_string(index=False))

    print("\nHead-tail Recall@20 gap:")
    if head_tail_gap.empty:
        print("No head-tail gap summary is available.")
    else:
        print(head_tail_gap.to_string(index=False))

    print(f"\nSaved outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
