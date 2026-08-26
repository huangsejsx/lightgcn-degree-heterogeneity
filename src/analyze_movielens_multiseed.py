from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

SEEDS = [42, 43, 44]
TOP_K = [10, 20]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


def build_user_items(df):
    return (
        df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def recall_at_k(ranked_items, true_item, k):
    return float(true_item in ranked_items[:k])


def ndcg_at_k(ranked_items, true_item, k):
    top_k = ranked_items[:k]

    if true_item not in top_k:
        return 0.0

    rank = top_k.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)


def evaluate_groups(
    seed,
    user_emb,
    item_emb,
    test,
    train_user_items,
    user_group_map,
    item_group_map,
):
    user_records = []
    item_records = []

    max_k = max(TOP_K)

    for _, row in test.iterrows():
        user = int(row["user_idx"])
        true_item = int(row["item_idx"])

        scores = user_emb[user] @ item_emb.T

        seen_items = train_user_items.get(user, set())

        if seen_items:
            scores[list(seen_items)] = -np.inf

        ranked_items = np.argpartition(
            -scores,
            max_k - 1,
        )[:max_k]

        ranked_items = ranked_items[
            np.argsort(-scores[ranked_items])
        ].tolist()

        user_group = user_group_map.get(
            user,
            "unknown",
        )

        item_group = item_group_map.get(
            true_item,
            "unknown",
        )

        for k in TOP_K:
            recall = recall_at_k(
                ranked_items,
                true_item,
                k,
            )

            ndcg = ndcg_at_k(
                ranked_items,
                true_item,
                k,
            )

            user_records.append(
                {
                    "seed": seed,
                    "group": user_group,
                    "K": k,
                    "Recall": recall,
                    "NDCG": ndcg,
                }
            )

            item_records.append(
                {
                    "seed": seed,
                    "group": item_group,
                    "K": k,
                    "Recall": recall,
                    "NDCG": ndcg,
                }
            )

    user_df = pd.DataFrame(user_records)

    item_df = pd.DataFrame(item_records)

    user_summary = (
        user_df.groupby(
            ["seed", "group", "K"],
            as_index=False,
        )[["Recall", "NDCG"]]
        .mean()
    )

    item_summary = (
        item_df.groupby(
            ["seed", "group", "K"],
            as_index=False,
        )[["Recall", "NDCG"]]
        .mean()
    )

    return user_summary, item_summary


def analyse_embeddings(
    seed,
    user_emb,
    item_emb,
    user_degrees,
    item_degrees,
):
    user_norms = np.linalg.norm(
        user_emb,
        axis=1,
    )

    item_norms = np.linalg.norm(
        item_emb,
        axis=1,
    )

    user_df = pd.DataFrame(
        {
            "user_idx": np.arange(len(user_norms)),
            "embedding_norm": user_norms,
        }
    )

    item_df = pd.DataFrame(
        {
            "item_idx": np.arange(len(item_norms)),
            "embedding_norm": item_norms,
        }
    )

    user_df = user_df.merge(
        user_degrees,
        on="user_idx",
        how="left",
    )

    item_df = item_df.merge(
        item_degrees,
        on="item_idx",
        how="left",
    )

    user_df["seed"] = seed
    item_df["seed"] = seed

    user_group = (
        user_df.groupby(
            ["seed", "degree_group"],
            as_index=False,
        )
        .agg(
            mean_embedding_norm=(
                "embedding_norm",
                "mean",
            ),
            median_embedding_norm=(
                "embedding_norm",
                "median",
            ),
            std_embedding_norm=(
                "embedding_norm",
                "std",
            ),
            count=(
                "embedding_norm",
                "count",
            ),
        )
    )

    item_group = (
        item_df.groupby(
            ["seed", "degree_group"],
            as_index=False,
        )
        .agg(
            mean_embedding_norm=(
                "embedding_norm",
                "mean",
            ),
            median_embedding_norm=(
                "embedding_norm",
                "median",
            ),
            std_embedding_norm=(
                "embedding_norm",
                "std",
            ),
            count=(
                "embedding_norm",
                "count",
            ),
        )
    )

    user_corr = user_df[
        ["degree", "embedding_norm"]
    ].corr().iloc[0, 1]

    item_corr = item_df[
        ["degree", "embedding_norm"]
    ].corr().iloc[0, 1]

    corr_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "node_type": "user",
                "correlation": user_corr,
            },
            {
                "seed": seed,
                "node_type": "item",
                "correlation": item_corr,
            },
        ]
    )

    return (
        user_group,
        item_group,
        corr_df,
        user_df,
        item_df,
    )


def bootstrap_correlation(
    df,
    node_type,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    df = df[
        ["degree", "embedding_norm"]
    ].dropna().copy()

    degree = df["degree"].to_numpy()
    norm = df["embedding_norm"].to_numpy()

    n = len(df)
    correlations = []

    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        sample_degree = degree[idx]
        sample_norm = norm[idx]

        corr = np.corrcoef(
            sample_degree,
            sample_norm,
        )[0, 1]

        correlations.append(corr)

    correlations = np.asarray(correlations)

    return {
        "node_type": node_type,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "ci_lower_95": np.percentile(
            correlations,
            2.5,
        ),
        "ci_upper_95": np.percentile(
            correlations,
            97.5,
        ),
    }


def summarise_group_metrics(df):
    return (
        df.groupby(["group", "K"])
        .agg(
            recall_mean=("Recall", "mean"),
            recall_std=("Recall", "std"),
            ndcg_mean=("NDCG", "mean"),
            ndcg_std=("NDCG", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )


def summarise_embedding_groups(df):
    return (
        df.groupby("degree_group")
        .agg(
            mean_norm=(
                "mean_embedding_norm",
                "mean",
            ),
            std_across_seeds=(
                "mean_embedding_norm",
                "std",
            ),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    user_degrees = pd.read_csv(
        DATA_DIR / "user_degrees.csv"
    )

    item_degrees = pd.read_csv(
        DATA_DIR / "item_degrees.csv"
    )

    train_user_items = build_user_items(train)

    user_group_map = dict(
        zip(
            user_degrees["user_idx"],
            user_degrees["degree_group"],
        )
    )

    item_group_map = dict(
        zip(
            item_degrees["item_idx"],
            item_degrees["degree_group"],
        )
    )

    all_user_metrics = []
    all_item_metrics = []

    all_user_groups = []
    all_item_groups = []

    all_correlations = []

    all_user_nodes = []
    all_item_nodes = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"Analysing seed {seed}")
        print("=" * 70)

        user_emb = np.load(
            RESULTS_DIR
            / f"lightgcn_user_embeddings_seed_{seed}.npy"
        )

        item_emb = np.load(
            RESULTS_DIR
            / f"lightgcn_item_embeddings_seed_{seed}.npy"
        )

        (
            user_metrics,
            item_metrics,
        ) = evaluate_groups(
            seed=seed,
            user_emb=user_emb,
            item_emb=item_emb,
            test=test,
            train_user_items=train_user_items,
            user_group_map=user_group_map,
            item_group_map=item_group_map,
        )

        (
            user_group,
            item_group,
            corr_df,
            user_nodes,
            item_nodes,
        ) = analyse_embeddings(
            seed=seed,
            user_emb=user_emb,
            item_emb=item_emb,
            user_degrees=user_degrees,
            item_degrees=item_degrees,
        )

        all_user_metrics.append(user_metrics)
        all_item_metrics.append(item_metrics)

        all_user_groups.append(user_group)
        all_item_groups.append(item_group)

        all_correlations.append(corr_df)

        all_user_nodes.append(user_nodes)
        all_item_nodes.append(item_nodes)

    user_metrics = pd.concat(
        all_user_metrics,
        ignore_index=True,
    )

    item_metrics = pd.concat(
        all_item_metrics,
        ignore_index=True,
    )

    user_groups = pd.concat(
        all_user_groups,
        ignore_index=True,
    )

    item_groups = pd.concat(
        all_item_groups,
        ignore_index=True,
    )

    correlations = pd.concat(
        all_correlations,
        ignore_index=True,
    )

    user_nodes = pd.concat(
        all_user_nodes,
        ignore_index=True,
    )

    item_nodes = pd.concat(
        all_item_nodes,
        ignore_index=True,
    )

    user_metrics.to_csv(
        RESULTS_DIR
        / "lightgcn_user_group_results_by_seed.csv",
        index=False,
    )

    item_metrics.to_csv(
        RESULTS_DIR
        / "lightgcn_item_group_results_by_seed.csv",
        index=False,
    )

    user_summary = summarise_group_metrics(
        user_metrics
    )

    item_summary = summarise_group_metrics(
        item_metrics
    )

    user_summary.to_csv(
        RESULTS_DIR
        / "lightgcn_user_group_summary.csv",
        index=False,
    )

    item_summary.to_csv(
        RESULTS_DIR
        / "lightgcn_item_group_summary.csv",
        index=False,
    )

    user_groups.to_csv(
        RESULTS_DIR
        / "user_embedding_group_results_by_seed.csv",
        index=False,
    )

    item_groups.to_csv(
        RESULTS_DIR
        / "item_embedding_group_results_by_seed.csv",
        index=False,
    )

    user_embedding_summary = (
        summarise_embedding_groups(user_groups)
    )

    item_embedding_summary = (
        summarise_embedding_groups(item_groups)
    )

    user_embedding_summary.to_csv(
        RESULTS_DIR
        / "user_embedding_group_summary.csv",
        index=False,
    )

    item_embedding_summary.to_csv(
        RESULTS_DIR
        / "item_embedding_group_summary.csv",
        index=False,
    )

    correlations.to_csv(
        RESULTS_DIR
        / "degree_norm_correlation_by_seed.csv",
        index=False,
    )

    correlation_summary = (
        correlations.groupby("node_type")
        .agg(
            correlation_mean=(
                "correlation",
                "mean",
            ),
            correlation_std=(
                "correlation",
                "std",
            ),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )

    correlation_summary.to_csv(
        RESULTS_DIR
        / "degree_norm_correlation_summary.csv",
        index=False,
    )

    mean_user_nodes = (
        user_nodes.groupby("user_idx")
        .agg(
            degree=("degree", "first"),
            embedding_norm=(
                "embedding_norm",
                "mean",
            ),
        )
        .reset_index()
    )

    mean_item_nodes = (
        item_nodes.groupby("item_idx")
        .agg(
            degree=("degree", "first"),
            embedding_norm=(
                "embedding_norm",
                "mean",
            ),
        )
        .reset_index()
    )

    bootstrap_results = pd.DataFrame(
        [
            bootstrap_correlation(
                mean_user_nodes,
                "user",
            ),
            bootstrap_correlation(
                mean_item_nodes,
                "item",
            ),
        ]
    )

    bootstrap_results.to_csv(
        RESULTS_DIR
        / "degree_norm_bootstrap_ci.csv",
        index=False,
    )

    print("\nUser grouped metrics:")
    print(user_summary.to_string(index=False))

    print("\nItem grouped metrics:")
    print(item_summary.to_string(index=False))

    print("\nUser embedding norm summary:")
    print(
        user_embedding_summary.to_string(
            index=False
        )
    )

    print("\nItem embedding norm summary:")
    print(
        item_embedding_summary.to_string(
            index=False
        )
    )

    print("\nDegree-norm correlation:")
    print(
        correlation_summary.to_string(
            index=False
        )
    )

    print("\nBootstrap 95% CI:")
    print(
        bootstrap_results.to_string(
            index=False
        )
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
