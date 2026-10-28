"""
bootstrap_bert4rec_reranking.py

Paired per-user bootstrap for tuned BERT4Rec reranking.

Run with PYTHONPATH=. python scripts/bootstrap_bert4rec_reranking.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import BERT4Rec
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from scripts.run_bert4rec_newman_reranking import (
    load_global_scores,
    predict_logits,
    prepare_rows,
)

LOGGER = logging.getLogger(__name__)

ROOT = Path(".")
RESULTS_DIR = ROOT / "results/bert4rec_reranking"
BASELINE_RESULTS = RESULTS_DIR / "tuned_results.json"
OUTPUT_PATH = RESULTS_DIR / "paired_user_bootstrap.json"
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
METRICS = ("HR@10", "NDCG@10")


def per_user_metrics(
    base_scores: np.ndarray,
    rows: list[dict],
    global_scores: np.ndarray | None = None,
    weight: float = 0.0,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
) -> pd.DataFrame:
    """Return one row per user using the existing evaluation rules."""
    scores = base_scores.astype(np.float64, copy=True)

    if global_scores is not None and weight != 0.0:
        scores += weight * global_scores[np.newaxis, :]

    scores[:, pad_token_id] = -np.inf
    scores[:, mask_token_id] = -np.inf

    records = []

    for index, row in enumerate(rows):
        row_scores = scores[index].copy()

        for item in row["history"]:
            item = int(item)
            if 0 <= item < len(row_scores):
                row_scores[item] = -np.inf

        ranked = np.argsort(-row_scores, kind="stable")
        ranked = [
            int(item) for item in ranked
            if np.isfinite(row_scores[item])
            and item not in (pad_token_id, mask_token_id)
        ][:10]

        target = row["target"]
        try:
            rank = ranked.index(target) + 1
        except ValueError:
            rank = -1

        records.append({
            "user_id": int(row["user_id"]),
            "HR@10": float(1 <= rank <= 10),
            "NDCG@10": (
                float(1.0 / np.log2(rank + 1))
                if 1 <= rank <= 10 else 0.0
            ),
        })

    result = pd.DataFrame(records)

    if result["user_id"].duplicated().any():
        raise ValueError("Expected exactly one test row per user.")

    return result


def bootstrap_difference(
    baseline: np.ndarray,
    reranked: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Paired user bootstrap CI for reranked minus baseline."""
    if baseline.shape != reranked.shape:
        raise ValueError("Baseline and reranked arrays must be aligned.")

    differences = reranked - baseline
    n_users = len(differences)

    if n_users == 0:
        raise ValueError("Cannot bootstrap an empty test set.")

    observed = float(differences.mean())
    boot_means = np.empty(N_BOOTSTRAP, dtype=np.float64)

    # Batch replicates to avoid allocating a huge index matrix.
    batch_size = 250
    for start in range(0, N_BOOTSTRAP, batch_size):
        size = min(batch_size, N_BOOTSTRAP - start)
        indices = rng.integers(0, n_users, size=(size, n_users))
        boot_means[start:start + size] = differences[indices].mean(axis=1)

    lower, upper = np.quantile(boot_means, [0.025, 0.975])

    return {
        "baseline_mean": float(baseline.mean()),
        "reranked_mean": float(reranked.mean()),
        "difference_reranked_minus_baseline": observed,
        "relative_change_percent": (
            float(100.0 * observed / baseline.mean())
            if baseline.mean() != 0 else None
        ),
        "ci_95_percentile": [float(lower), float(upper)],
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "bootstrap_probability_difference_positive": float(
            np.mean(boot_means > 0)
        ),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result_path = BASELINE_RESULTS
    if not result_path.exists():
        raise FileNotFoundError(f"Missing existing results: {result_path}")

    saved = json.loads(result_path.read_text(encoding="utf-8"))
    if saved.get("model") != "tuned":
        raise ValueError(f"Expected tuned results, found {saved.get('model')}")

    selected_weights = saved["selected_weights"]

    config_path = ROOT / "configs/bert4rec_tuned.yaml"
    checkpoint_path = (
        ROOT / "checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt"
    )

    config = load_config(config_path)
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

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_df = pd.read_parquet(data_cfg["test_path"])
    rows = prepare_rows(
        test_df,
        "test_input",
        "test_item",
        max_seq_len,
        pad_id,
        mask_id,
    )

    if len(rows) != len(test_df):
        raise ValueError("Some test rows were lost during preparation.")

    LOGGER.info("Scoring %d test users", len(rows))
    logits = predict_logits(
        model, rows, device, vocab_size, pad_id, mask_id, batch_size=64
    )

    baseline_users = per_user_metrics(
        logits, rows, pad_token_id=pad_id, mask_token_id=mask_id
    )
    expected_users = baseline_users["user_id"].to_numpy()

    # Confirm that the new per-user evaluator reproduces saved aggregate
    # baseline metrics before interpreting any bootstrap results.
    for metric in METRICS:
        calculated = float(baseline_users[metric].mean())
        expected = float(saved["test_metrics"]["bert4rec"][metric])
        if not np.isclose(calculated, expected, atol=1e-6, rtol=0):
            raise ValueError(
                f"{metric} mismatch: recomputed {calculated:.8f}, "
                f"saved {expected:.8f}. Stop and investigate."
            )

    baseline_by_metric = {
        metric: baseline_users[metric].to_numpy(dtype=np.float64)
        for metric in METRICS
    }

    baseline_scores_path = (
        ROOT / "data/processed/ml-1m/movie_network/"
        "global_ranking_baselines.csv"
    )
    newman_path = ROOT / "data/processed/ml-1m/movie_network/"
    newman_path = newman_path / "movie_scores_direct.csv"

    methods = {
        "newman": load_global_scores(newman_path, vocab_size),
        "mean_rating": load_global_scores(
            baseline_scores_path, vocab_size,
            score_column="mean_rating_score",
        ),
        "shrunk_mean_rating": load_global_scores(
            baseline_scores_path, vocab_size,
            score_column="shrunk_mean_score",
        ),
        "popularity": load_global_scores(
            baseline_scores_path, vocab_size,
            score_column="popularity_score",
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = {}

    for method, global_scores in methods.items():
        weight = float(selected_weights[method])
        LOGGER.info("Evaluating %s at validation-selected lambda=%s",
                    method, weight)

        reranked_users = per_user_metrics(
            logits,
            rows,
            global_scores=global_scores,
            weight=weight,
            pad_token_id=pad_id,
            mask_token_id=mask_id,
        )

        if not np.array_equal(
            expected_users, reranked_users["user_id"].to_numpy()
        ):
            raise ValueError(f"User alignment mismatch for {method}.")

        # Check aggregate reproduction against the saved test results.
        for metric in METRICS:
            calculated = float(reranked_users[metric].mean())
            expected = float(saved["test_metrics"][method][metric])
            if not np.isclose(calculated, expected, atol=1e-6, rtol=0):
                raise ValueError(
                    f"{method} {metric} mismatch: recomputed {calculated:.8f}, "
                    f"saved {expected:.8f}. Stop and investigate."
                )

        comparisons[method] = {
            "validation_selected_lambda": weight,
            "metrics": {},
        }

        for metric in METRICS:
            comparisons[method]["metrics"][metric] = bootstrap_difference(
                baseline_by_metric[metric],
                reranked_users[metric].to_numpy(dtype=np.float64),
                rng,
            )

    output = {
        "model": "tuned",
        "checkpoint": str(checkpoint_path),
        "test_data": str(data_cfg["test_path"]),
        "baseline_results_source": str(result_path),
        "n_users": len(rows),
        "bootstrap_replicates": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "confidence_interval": "95% percentile paired-user bootstrap",
        "difference_direction": "reranked minus BERT4Rec baseline",
        "weight_selection": "Existing validation-selected weights; not retuned",
        "aggregate_reproduction_checks": "passed",
        "comparisons": comparisons,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nPaired bootstrap results saved to: {OUTPUT_PATH}")
    print(f"Users: {len(rows)} | Replicates: {N_BOOTSTRAP}")
    print("Differences are reranked minus baseline.")
    for method, details in comparisons.items():
        print(f"\n{method} (lambda={details['validation_selected_lambda']})")
        for metric, result in details["metrics"].items():
            lo, hi = result["ci_95_percentile"]
            print(
                f"  {metric}: delta={result['difference_reranked_minus_baseline']:+.6f}, "
                f"95% CI [{lo:+.6f}, {hi:+.6f}], "
                f"excludes zero={result['ci_excludes_zero']}"
            )


if __name__ == "__main__":
    main()
