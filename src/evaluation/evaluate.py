
"""
Evaluation routines for sequential recommendation models.

BERT4Rec uses the candidate-filtering protocol and
canonical ranking metrics:
    HR@5, HR@10, NDCG@5, NDCG@10
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_ranking_metrics


def filter_ranked_recommendations(
    ranked_candidates: torch.Tensor,
    histories: Sequence[Sequence[int]] | torch.Tensor,
    *,
    k: int,
    pad_token_id: int = 0,
    mask_token_id: int | None = 1,
    history_lengths: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Remove previously seen items and special tokens from ranked candidates.

    ranked_candidates must contain the complete candidate ranking, sorted
    from highest to lowest score, rather than only the original top-k.

    Returns [batch_size, k]. Missing recommendations are represented by -1.
    """
    if ranked_candidates.ndim != 2:
        raise ValueError("ranked_candidates must be a 2D tensor.")
    if k <= 0:
        raise ValueError("k must be positive.")

    batch_size = ranked_candidates.shape[0]

    if isinstance(histories, torch.Tensor):
        if histories.ndim != 2 or histories.shape[0] != batch_size:
            raise ValueError("Tensor histories must have shape [batch, seq_len].")
        history_rows = histories.detach().cpu().tolist()
    else:
        history_rows = list(histories)

    if len(history_rows) != batch_size:
        raise ValueError("Number of histories must match ranking batch size.")

    if history_lengths is not None:
        if isinstance(history_lengths, torch.Tensor):
            lengths = history_lengths.detach().cpu().tolist()
        else:
            lengths = list(history_lengths)

        if len(lengths) != batch_size:
            raise ValueError("history_lengths must match ranking batch size.")
    else:
        lengths = None

    filtered_rows: list[list[int]] = []

    for row_idx, history in enumerate(history_rows):
        if lengths is not None:
            valid_length = max(0, min(int(lengths[row_idx]), len(history)))
            history = history[:valid_length]

        seen = {
            int(item)
            for item in history
            if int(item) != pad_token_id
            and (mask_token_id is None or int(item) != mask_token_id)
        }

        selected: list[int] = []

        for candidate in ranked_candidates[row_idx].detach().cpu().tolist():
            candidate = int(candidate)

            if candidate == pad_token_id:
                continue
            if mask_token_id is not None and candidate == mask_token_id:
                continue
            if candidate in seen:
                continue

            selected.append(candidate)

            if len(selected) == k:
                break

        selected.extend([-1] * (k - len(selected)))
        filtered_rows.append(selected)

    return torch.tensor(
        filtered_rows,
        dtype=torch.long,
        device=ranked_candidates.device,
    )


def evaluate_bert4rec(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    max_seq_len: int,
    vocab_size: int,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
    target_column: str = "validation_target",
    input_column: str = "train_sequence",
    input_requires_mask: bool = True,
    limit_users: int | None = None,
    exclude_seen: bool = True,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Evaluate BERT4Rec with the shared seen-item filtering protocol.

    Returns HR@5, HR@10, NDCG@5, NDCG@10 and a small set of debug examples.
    """
    required_columns = {"user_id", input_column, target_column}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Evaluation DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if vocab_size <= 2 or max_seq_len <= 0:
        raise ValueError(
            "vocab_size must be > 2 and max_seq_len must be positive."
        )
    if not (0 <= pad_token_id < vocab_size):
        raise ValueError("pad_token_id must be within the model vocabulary.")
    if not (0 <= mask_token_id < vocab_size):
        raise ValueError("mask_token_id must be within the model vocabulary.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if limit_users is not None and limit_users <= 0:
        raise ValueError("limit_users must be positive.")

    eval_users = df.iloc[:limit_users] if limit_users is not None else df

    # Each entry:
    # user_id, history length, target, original history, padded input,
    # prediction mask position.
    prepared: list[tuple[Any, int, int, list[int], list[int], int]] = []

    for _, row in eval_users.iterrows():
        sequence = row[input_column]
        sequence = (
            sequence.tolist()
            if hasattr(sequence, "tolist")
            else list(sequence)
        )
        sequence = [int(item) for item in sequence]
        target = int(row[target_column])

        if target < 0 or target >= vocab_size:
            raise ValueError(
                f"Target item {target} for user {row['user_id']} "
                f"is outside vocabulary [0, {vocab_size - 1}]."
            )

        eval_seq = (
            sequence + [mask_token_id]
            if input_requires_mask
            else sequence.copy()
        )
        eval_seq = eval_seq[-max_seq_len:]

        if mask_token_id not in eval_seq:
            raise ValueError(
                f"User {row['user_id']} evaluation sequence does not "
                f"contain MASK token ({mask_token_id})."
            )

        mask_position = (
            len(eval_seq) - 1 - eval_seq[::-1].index(mask_token_id)
        )
        pad_len = max_seq_len - len(eval_seq)
        padded = [pad_token_id] * pad_len + eval_seq
        mask_position += pad_len

        prepared.append(
            (
                row["user_id"],
                len(sequence),
                target,
                sequence,
                padded,
                mask_position,
            )
        )

    if not prepared:
        raise ValueError("No users were available for BERT4Rec evaluation.")

    was_training = model.training
    model.eval()

    metric_rows: list[dict[str, float]] = []
    debug_outputs: list[dict[str, Any]] = []

    try:
        with torch.inference_mode():
            for offset in range(0, len(prepared), batch_size):
                batch_rows = prepared[offset:offset + batch_size]

                input_ids = torch.tensor(
                    [row[4] for row in batch_rows],
                    dtype=torch.long,
                    device=device,
                )
                mask_positions = torch.tensor(
                    [row[5] for row in batch_rows],
                    dtype=torch.long,
                    device=device,
                )

                logits = model(input_ids)
                row_indices = torch.arange(len(batch_rows), device=device)
                scores = logits[row_indices, mask_positions].float()

                # Exclude special tokens from recommendation candidates.
                scores[:, pad_token_id] = -float("inf")
                scores[:, mask_token_id] = -float("inf")

                # Rank all candidates before applying the shared filter.
                ranked_candidates = torch.argsort(
                    scores,
                    dim=1,
                    descending=True,
                )

                histories = (
                    [row[3] for row in batch_rows]
                    if exclude_seen
                    else [[] for _ in batch_rows]
                )

                top_k = filter_ranked_recommendations(
                    ranked_candidates,
                    histories,
                    k=10,
                    pad_token_id=pad_token_id,
                    mask_token_id=mask_token_id,
                ).cpu()

                targets = torch.tensor(
                    [row[2] for row in batch_rows],
                    dtype=torch.long,
                )

                for i, row in enumerate(batch_rows):
                    recommendations = top_k[i]
                    target = targets[i]

                    hits5 = recommendations[:5].eq(target)
                    hits10 = recommendations[:10].eq(target)

                    hit_positions = torch.nonzero(
                        recommendations.eq(target),
                        as_tuple=False,
                    )
                    rank = (
                        int(hit_positions[0, 0].item()) + 1
                        if len(hit_positions)
                        else -1
                    )

                    user_metrics = {
                        "HR@5": float(bool(hits5.any())),
                        "HR@10": float(bool(hits10.any())),
                        "NDCG@5": (
                            1.0 / np.log2(rank + 1)
                            if 1 <= rank <= 5
                            else 0.0
                        ),
                        "NDCG@10": (
                            1.0 / np.log2(rank + 1)
                            if 1 <= rank <= 10
                            else 0.0
                        ),
                    }
                    metric_rows.append(user_metrics)

                    if len(debug_outputs) < 5:
                        debug_outputs.append(
                            {
                                "user_id": row[0],
                                "history_len": row[1],
                                "target": row[2],
                                "top_10": recommendations.tolist(),
                                "rank": rank,
                                "hit_5": bool(user_metrics["HR@5"]),
                                "hit_10": bool(user_metrics["HR@10"]),
                            }
                        )
    finally:
        if was_training:
            model.train()

    return {
        "HR@5": float(np.mean([m["HR@5"] for m in metric_rows])),
        "HR@10": float(np.mean([m["HR@10"] for m in metric_rows])),
        "NDCG@5": float(np.mean([m["NDCG@5"] for m in metric_rows])),
        "NDCG@10": float(np.mean([m["NDCG@10"] for m in metric_rows])),
        "debug_outputs": debug_outputs,
    }
