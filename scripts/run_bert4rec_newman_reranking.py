"""
run_bert4rec_newman_reranking.py

Validation-tuned BERT4Rec reranking with Newman global scores and popularity.

Existing checkpoints and baseline evaluation results are never overwritten.

Run with 'PYTHONPATH=. python scripts/run_bert4rec_newman_reranking.py'
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import BERT4Rec
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config

LOGGER = logging.getLogger(__name__)

K_VALUES = (5, 10)
LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def standardize(values: np.ndarray) -> np.ndarray:
    """Z-score a vector, rejecting invalid inputs."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores contain non-finite values.")
    std = values.std()
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def load_global_scores(
    path: Path,
    vocab_size: int,
    score_column: str | None = None,
) -> np.ndarray:
    """Load Newman scores and standardize them across real catalog items."""
    df = pd.read_csv(path)

    if "mapped_movie_id" not in df.columns:
        raise ValueError(
            f"{path} must contain a 'mapped_movie_id' column; "
            f"found {list(df.columns)}"
        )

    if score_column is None:
        if "score" in df.columns:
            score_column = "score"
        elif "strength" in df.columns:
            # Strengths are positive; log transforms reduce extreme skew.
            score_column = "strength"
            df[score_column] = np.log(df[score_column].clip(lower=1e-12))
        else:
            raise ValueError(
                f"No 'score' or 'strength' column found in {path}."
            )

    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in {path}.")

    item_ids = df["mapped_movie_id"].astype(int).to_numpy()
    values = pd.to_numeric(df[score_column], errors="raise").to_numpy(
        dtype=np.float64
    )

    if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("Newman file contains duplicate mapped movie IDs.")

    real_ids = np.arange(2, vocab_size)
    lookup = dict(zip(item_ids, values))
    missing = [int(i) for i in real_ids if int(i) not in lookup]

    if missing:
        raise ValueError(
            f"Newman scores are missing {len(missing)} catalog items; "
            f"first missing IDs: {missing[:10]}"
        )

    result = np.zeros(vocab_size, dtype=np.float64)
    result[real_ids] = standardize(
        np.array([lookup[int(i)] for i in real_ids])
    )
    return result


def load_popularity_scores(
    train_sequences_path: Path,
    vocab_size: int,
) -> np.ndarray:
    """Count training interactions from the existing training sequences."""
    df = pd.read_parquet(train_sequences_path)

    if "train_sequence" not in df.columns:
        raise ValueError(
            f"{train_sequences_path} must contain 'train_sequence'; "
            f"found {list(df.columns)}"
        )

    counts = np.zeros(vocab_size, dtype=np.float64)

    for sequence in df["train_sequence"]:
        sequence = sequence.tolist() if hasattr(sequence, "tolist") else sequence
        for item in sequence:
            item = int(item)
            if 2 <= item < vocab_size:
                counts[item] += 1

    result = np.zeros(vocab_size, dtype=np.float64)
    result[2:] = standardize(np.log1p(counts[2:]))
    return result


def prepare_rows(
    df: pd.DataFrame,
    input_column: str,
    target_column: str,
    max_seq_len: int,
    pad_token_id: int,
    mask_token_id: int,
) -> list[dict]:
    required = {"user_id", input_column, target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {sorted(missing)}")

    rows = []

    for _, row in df.iterrows():
        sequence = row[input_column]
        sequence = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        sequence = [int(item) for item in sequence]
        target = int(row[target_column])

        if target < 2:
            raise ValueError(f"Invalid target item ID {target}.")

        eval_sequence = (sequence + [mask_token_id])[-max_seq_len:]
        if mask_token_id not in eval_sequence:
            raise ValueError(
                f"User {row['user_id']} has no MASK token in evaluation input."
            )

        mask_position = (
            len(eval_sequence) - 1 - eval_sequence[::-1].index(mask_token_id)
        )
        padded = (
            [pad_token_id] * (max_seq_len - len(eval_sequence))
            + eval_sequence
        )

        rows.append({
            "user_id": int(row["user_id"]),
            "target": target,
            "history": sequence,
            "input_ids": padded,
            "mask_position": mask_position + max_seq_len - len(eval_sequence),
        })

    return rows


def predict_logits(
    model: BERT4Rec,
    rows: list[dict],
    device: torch.device,
    vocab_size: int,
    pad_token_id: int,
    mask_token_id: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Return one full vocabulary score vector per user."""
    all_scores = np.empty((len(rows), vocab_size), dtype=np.float32)
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs = torch.tensor(
                [r["input_ids"] for r in batch],
                dtype=torch.long,
                device=device,
            )
            positions = torch.tensor(
                [r["mask_position"] for r in batch],
                dtype=torch.long,
                device=device,
            )
            logits = model(inputs)
            indices = torch.arange(len(batch), device=device)
            scores = logits[indices, positions].float()
            scores[:, pad_token_id] = -float("inf")
            scores[:, mask_token_id] = -float("inf")
            all_scores[start:start + len(batch)] = scores.cpu().numpy()

    return all_scores


def calculate_metrics(
    base_scores: np.ndarray,
    rows: list[dict],
    global_scores: np.ndarray | None = None,
    weight: float = 0.0,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
) -> dict[str, float]:
    """Evaluate full-catalog recommendations using the shared filtering rules."""
    scores = base_scores.astype(np.float64, copy=True)

    if global_scores is not None and weight != 0.0:
        scores += weight * global_scores[np.newaxis, :]

    scores[:, pad_token_id] = -np.inf
    scores[:, mask_token_id] = -np.inf

    metric_sums = {
        "HR@5": 0.0,
        "HR@10": 0.0,
        "NDCG@5": 0.0,
        "NDCG@10": 0.0,
    }

    for index, row in enumerate(rows):
        row_scores = scores[index].copy()

        # Match evaluate.py: previously seen items are removed from candidates.
        for item in row["history"]:
            item = int(item)
            if 0 <= item < len(row_scores):
                row_scores[item] = -np.inf

        # Full ranking first, followed by special-token/seen-item filtering.
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

        for k in K_VALUES:
            metric_sums[f"HR@{k}"] += float(1 <= rank <= k)
            metric_sums[f"NDCG@{k}"] += (
                1.0 / np.log2(rank + 1) if 1 <= rank <= k else 0.0
            )

    n = len(rows)
    return {key: value / n for key, value in metric_sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "tuned"),
        default=("baseline", "tuned"),
    )
    parser.add_argument(
        "--newman-scores",
        default="data/processed/ml-1m/movie_network/movie_scores_direct.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/bert4rec_reranking",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(".")
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    newman_path = root / args.newman_scores
    if not newman_path.exists():
        raise FileNotFoundError(
            f"Newman score file not found: {newman_path}"
        )

    for variant in args.models:
        config_path = root / f"configs/bert4rec_{variant}.yaml"
        checkpoint_path = (
            root / f"checkpoints/bert4rec_{variant}/"
            f"bert4rec_{variant}_best.pt"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config = load_config(config_path)
        model_cfg = config["model"]
        data_cfg = config["data"]
        vocab_size = int(model_cfg["vocab_size"])
        pad_id = int(model_cfg["pad_token_id"])
        mask_id = int(model_cfg["mask_token_id"])
        max_seq_len = int(model_cfg["max_seq_len"])

        configured_device = config.get("device")
        if configured_device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using CPU.")
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        LOGGER.info("Loading %s checkpoint on %s", variant, device)
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

        validation_df = pd.read_parquet(data_cfg["validation_path"])
        test_df = pd.read_parquet(data_cfg["test_path"])

        validation_rows = prepare_rows(
            validation_df, "train_sequence", "validation_item",
            max_seq_len, pad_id, mask_id,
        )
        test_rows = prepare_rows(
            test_df, "test_input", "test_item",
            max_seq_len, pad_id, mask_id,
        )

        LOGGER.info("Scoring validation users")
        validation_logits = predict_logits(
            model, validation_rows, device, vocab_size, pad_id, mask_id,
            args.batch_size,
        )
        LOGGER.info("Scoring test users")
        test_logits = predict_logits(
            model, test_rows, device, vocab_size, pad_id, mask_id,
            args.batch_size,
        )

        newman = load_global_scores(newman_path, vocab_size)

        # All additional global baselines are built from training ratings only.
        baseline_scores_path = (
            root / "data/processed/ml-1m/movie_network/"
            "global_ranking_baselines.csv"
        )

        mean_rating = load_global_scores(
            baseline_scores_path,
            vocab_size,
            score_column="mean_rating_score",
        )
        shrunk_mean_rating = load_global_scores(
            baseline_scores_path,
            vocab_size,
            score_column="shrunk_mean_score",
        )
        popularity = load_global_scores(
            baseline_scores_path,
            vocab_size,
            score_column="popularity_score",
        )

        methods = {
            "newman": newman,
            "mean_rating": mean_rating,
            "shrunk_mean_rating": shrunk_mean_rating,
            "popularity": popularity,
        }
        validation_grid = []
        selected = {}

        # Tune each reranker independently using validation NDCG@10.
        baseline_validation = calculate_metrics(
            validation_logits, validation_rows, pad_token_id=pad_id,
            mask_token_id=mask_id,
        )

        for method_name, global_scores in methods.items():
            for weight in LAMBDA_GRID:
                metrics = calculate_metrics(
                    validation_logits,
                    validation_rows,
                    global_scores=global_scores,
                    weight=weight,
                    pad_token_id=pad_id,
                    mask_token_id=mask_id,
                )
                validation_grid.append({
                    "model": variant,
                    "method": method_name,
                    "lambda": weight,
                    **metrics,
                })

            method_grid = [
                item for item in validation_grid
                if item["method"] == method_name
            ]
            best = max(
                method_grid,
                key=lambda item: (item["NDCG@10"], -item["lambda"]),
            )
            selected[method_name] = best["lambda"]

        grid_path = output_dir / f"{variant}_validation_grid.csv"
        pd.DataFrame(validation_grid).to_csv(grid_path, index=False)

        test_results = {
            "model": variant,
            "checkpoint": str(checkpoint_path),
            "validation_metrics_baseline": baseline_validation,
            "selected_weights": selected,
            "test_metrics": {
                "bert4rec": calculate_metrics(
                    test_logits, test_rows, pad_token_id=pad_id,
                    mask_token_id=mask_id,
                )
            },
            "validation_selection_metric": "NDCG@10",
            "lambda_grid": list(LAMBDA_GRID),
            "newman_score_file": str(newman_path),
        }

        for method_name, global_scores in methods.items():
            weight = selected[method_name]
            test_results["test_metrics"][method_name] = calculate_metrics(
                test_logits,
                test_rows,
                global_scores=global_scores,
                weight=weight,
                pad_token_id=pad_id,
                mask_token_id=mask_id,
            )

        result_path = output_dir / f"{variant}_results.json"
        result_path.write_text(
            json.dumps(test_results, indent=2),
            encoding="utf-8",
        )

        print(f"\n{variant.upper()} — validation-selected reranking")
        print(f"Selected Newman lambda:     {selected['newman']}")
        print(f"Selected popularity lambda:  {selected['popularity']}")
        print("\nTest metrics:")
        for method_name, metrics in test_results["test_metrics"].items():
            print(f"\n{method_name}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<8} {value:.4f}")

        print(f"\nSaved validation grid: {grid_path}")
        print(f"Saved test results:    {result_path}")


if __name__ == "__main__":
    main()
