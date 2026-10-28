"""
analyze_popularity_segments.py

Analyze which test users benefit from popularity reranking.

Uses the existing tuned BERT4Rec checkpoint and validation-selected
popularity weight. Does not train or retune any model.

Run with PYTHONPATH=. python scripts/analyze_popularity_segments.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import BERT4Rec
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from scripts.bootstrap_bert4rec_reranking import (
    METRICS,
    per_user_metrics,
)
from scripts.run_bert4rec_newman_reranking import (
    load_global_scores,
    predict_logits,
    prepare_rows,
)

ROOT = Path(".")
RESULTS = ROOT / "results/bert4rec_reranking"
RESULT_JSON = RESULTS / "tuned_results.json"
USER_OUTPUT = RESULTS / "popularity_per_user_diagnostics.csv"
SEGMENT_OUTPUT = RESULTS / "popularity_segment_summary.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def summarize_segment(df, group_column):
    """Aggregate paired baseline and reranked metrics by segment."""
    records = []

    for segment, group in df.groupby(group_column, observed=True):
        record = {
            "grouping": group_column,
            "segment": str(segment),
            "n_users": int(len(group)),
        }

        for metric in METRICS:
            base_col = f"baseline_{metric}"
            rerank_col = f"popularity_{metric}"
            delta_col = f"delta_{metric}"

            record[f"baseline_{metric}"] = group[base_col].mean()
            record[f"popularity_{metric}"] = group[rerank_col].mean()
            record[f"delta_{metric}"] = group[delta_col].mean()

            # Percentage of users whose individual metric improved.
            record[f"users_improved_{metric}_pct"] = (
                100.0 * (group[delta_col] > 0).mean()
            )
            record[f"users_unchanged_{metric}_pct"] = (
                100.0 * (group[delta_col] == 0).mean()
            )

        record["median_target_train_count"] = (
            group["target_train_count"].median()
        )
        record["median_history_length"] = group["history_length"].median()
        records.append(record)

    return pd.DataFrame(records)


def main():
    if not RESULT_JSON.exists():
        raise FileNotFoundError(f"Missing {RESULT_JSON}")

    saved = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    if saved.get("model") != "tuned":
        raise ValueError("Expected tuned BERT4Rec results.")

    weight = float(saved["selected_weights"]["popularity"])
    LOGGER.info("Using existing validation-selected popularity lambda=%s", weight)

    config = load_config(ROOT / "configs/bert4rec_tuned.yaml")
    model_cfg = config["model"]
    data_cfg = config["data"]

    vocab_size = int(model_cfg["vocab_size"])
    pad_id = int(model_cfg["pad_token_id"])
    mask_id = int(model_cfg["mask_token_id"])
    max_seq_len = int(model_cfg["max_seq_len"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading tuned checkpoint on %s", device)

    model = BERT4Rec(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        feed_forward_dim=model_cfg["feed_forward_dim"],
        dropout=model_cfg["dropout"],
        pad_token_id=pad_id,
    ).to(device)

    checkpoint_path = (
        ROOT / "checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_df = pd.read_parquet(data_cfg["test_path"])
    rows = prepare_rows(
        test_df, "test_input", "test_item", max_seq_len, pad_id, mask_id
    )
    if len(rows) != len(test_df):
        raise ValueError("Some test rows were lost during preparation.")

    LOGGER.info("Scoring %d test users", len(rows))
    logits = predict_logits(
        model, rows, device, vocab_size, pad_id, mask_id, batch_size=64
    )

    baseline = per_user_metrics(
        logits, rows, pad_token_id=pad_id, mask_token_id=mask_id
    )

    baseline_scores_path = (
        ROOT / "data/processed/ml-1m/movie_network/"
        "global_ranking_baselines.csv"
    )
    popularity_scores = load_global_scores(
        baseline_scores_path, vocab_size, score_column="popularity_score"
    )

    reranked = per_user_metrics(
        logits,
        rows,
        global_scores=popularity_scores,
        weight=weight,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
    )

    if not np.array_equal(
        baseline["user_id"].to_numpy(), reranked["user_id"].to_numpy()
    ):
        raise ValueError("Baseline and reranked user ordering differs.")

    # Verify this script reproduces the already saved aggregate metrics.
    for metric in METRICS:
        for method, frame in [
            ("bert4rec", baseline),
            ("popularity", reranked),
        ]:
            observed = float(frame[metric].mean())
            expected = float(saved["test_metrics"][method][metric])
            if not np.isclose(observed, expected, atol=1e-6, rtol=0):
                raise ValueError(
                    f"{method} {metric} mismatch: {observed:.8f} "
                    f"vs saved {expected:.8f}"
                )

    diagnostics = pd.DataFrame({
        "user_id": [int(row["user_id"]) for row in rows],
        "target_movie_id": [int(row["target"]) for row in rows],
        "history_length": [len(row["history"]) for row in rows],
    })

    for metric in METRICS:
        diagnostics[f"baseline_{metric}"] = baseline[metric].to_numpy()
        diagnostics[f"popularity_{metric}"] = reranked[metric].to_numpy()
        diagnostics[f"delta_{metric}"] = (
            diagnostics[f"popularity_{metric}"]
            - diagnostics[f"baseline_{metric}"]
        )

    # Popularity is measured exclusively from training interactions.
    train_path = ROOT / "data/interim/ml-1m/network_train_ratings.parquet"
    train = pd.read_parquet(train_path, columns=["mapped_movie_id"])
    counts = train["mapped_movie_id"].value_counts()

    diagnostics["target_train_count"] = (
        diagnostics["target_movie_id"].map(counts).fillna(0).astype(int)
    )

    # Balanced target-popularity quartiles; Q1 is least popular.
    ranks = diagnostics["target_train_count"].rank(method="first")
    diagnostics["target_popularity_quartile"] = pd.qcut(
        ranks,
        q=4,
        labels=["Q1_least_popular", "Q2", "Q3", "Q4_most_popular"],
    )

    diagnostics["history_group"] = pd.cut(
        diagnostics["history_length"],
        bins=[0, 5, 10, 20, 50, np.inf],
        labels=["1-4", "5-9", "10-19", "20-49", "50+"],
        right=False,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(USER_OUTPUT, index=False)

    summaries = pd.concat(
        [
            summarize_segment(diagnostics, "history_group"),
            summarize_segment(diagnostics, "target_popularity_quartile"),
        ],
        ignore_index=True,
    )
    summaries.to_csv(SEGMENT_OUTPUT, index=False)

    print("\nAggregate reproduction checks passed.")
    print(f"Per-user diagnostics: {USER_OUTPUT}")
    print(f"Segment summary:      {SEGMENT_OUTPUT}")
    print(f"Users: {len(diagnostics)}")
    print(f"Popularity lambda: {weight}")

    columns = [
        "grouping", "segment", "n_users",
        "delta_HR@10", "delta_NDCG@10",
        "users_improved_HR@10_pct",
        "users_improved_NDCG@10_pct",
    ]
    print("\nSegment summary (reranked minus baseline):")
    print(summaries[columns].to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
