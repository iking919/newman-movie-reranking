"""Newman's typed pairwise-comparison ranking (MAP estimate).

Implements the model in:
M. E. J. Newman, "Ranking with Multiple Types of Pairwise Comparisons."

Model
-----
Each movie i has a strength pi_i > 0 with prior P(pi) ~ 1/(1+pi)^2.
Each comparison type k has a valence q_k in (0, 1) (flat prior).
For a type-k comparison between i and j, i is "dominant" with
probability pi_i / (pi_i + pi_j), and the dominant movie is the
observed winner with probability q_k. Marginally:

    P(i wins) = (pi_i q_k + pi_j (1 - q_k)) / (pi_i + pi_j)

Estimation
----------
Both methods maximize the SAME posterior over (log pi, q):

  --method em      Newman's EM with the MAP strength update (Eq. 22).
                   Literal paper procedure; converges slowly here.
  --method direct  L-BFGS on the marginal log-posterior. Same objective
                   and prior, different optimizer. Provides a gradient-
                   based convergence criterion.

Input:
    nodes.csv: mapped_movie_id
    edges.csv: source, target, comparison_type, weight
source = observed winner (higher-rated movie), target = observed loser.

Output: movie_scores.csv with strength, score (= log strength), rank.


Run with:

PYTHONPATH=. python -m src.network.newman_ranking \
  --method direct \
  --init-scores data/processed/ml-1m/movie_network/movie_scores_em.csv \
  --max-iter 2000 \
  --output-name movie_scores_direct.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

NUM_TYPES = 2


# --------------------------------------------------------------------------
# Validation and preparation
# --------------------------------------------------------------------------
def prepare_data(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    required_nodes = {"mapped_movie_id"}
    required_edges = {"source", "target", "comparison_type", "weight"}

    if not required_nodes.issubset(nodes.columns):
        raise ValueError("nodes must contain mapped_movie_id.")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"edges must contain {sorted(required_edges)}.")

    node_ids = np.sort(nodes["mapped_movie_id"].to_numpy(dtype=np.int64))
    if len(node_ids) == 0 or len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Node IDs must be nonempty and unique.")
    if not np.array_equal(
        node_ids, np.arange(node_ids[0], node_ids[0] + len(node_ids))
    ):
        raise ValueError("Expected contiguous mapped movie IDs.")

    if edges.empty:
        raise ValueError("The comparison network contains no edges.")
    if edges[list(required_edges)].isna().any().any():
        raise ValueError("Network edges contain missing values.")
    if not edges["comparison_type"].isin([1, 2]).all():
        raise ValueError("Expected comparison types 1 and 2.")

    weights = edges["weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Edge weights must be finite and positive.")

    n = len(node_ids)
    source = edges["source"].to_numpy(dtype=np.int64) - node_ids[0]
    target = edges["target"].to_numpy(dtype=np.int64) - node_ids[0]

    if (
        (source < 0).any() or (source >= n).any()
        or (target < 0).any() or (target >= n).any()
    ):
        raise ValueError("An edge refers to an unknown movie ID.")
    if np.any(source == target):
        raise ValueError("Self-comparisons are not allowed.")

    type_index = edges["comparison_type"].to_numpy(dtype=np.int64) - 1
    type_counts = np.bincount(type_index, weights=weights, minlength=NUM_TYPES)
    if np.any(type_counts == 0):
        raise ValueError("Both comparison types need positive total weight.")

    return {
        "n": n,
        "node_ids": node_ids,
        "source": source,
        "target": target,
        "type_index": type_index,
        "weights": weights,
        "type_counts": type_counts,
    }


# --------------------------------------------------------------------------
# Shared objective: negative log-posterior in (x = log pi, theta = logit q)
# --------------------------------------------------------------------------
def neg_log_posterior(params: np.ndarray, data: dict) -> tuple[float, np.ndarray]:
    n = data["n"]
    source, target = data["source"], data["target"]
    type_index, W = data["type_index"], data["weights"]

    x, theta = params[:n], params[n:]
    q = expit(theta)
    log_q = -np.logaddexp(0.0, -theta)
    log_1mq = -np.logaddexp(0.0, theta)

    xs, xt = x[source], x[target]
    lq, l1q = log_q[type_index], log_1mq[type_index]
    d = xs - xt

    # Marginal log-likelihood of each observed outcome.
    ell = np.logaddexp(xs + lq, xt + l1q) - np.logaddexp(xs, xt)

    # Prior P(pi) ~ (1+pi)^-2  ->  -log prior = 2 log(1+e^x) - x
    value = -np.dot(W, ell) + np.sum(2.0 * np.logaddexp(0.0, x) - x)

    r = expit(d + lq - l1q)  # posterior P(winner is dominant | outcome)
    p = expit(d)             # prior P(winner is dominant)
    g_edge = W * (r - p)

    grad_x = -(
        np.bincount(source, weights=g_edge, minlength=n)
        - np.bincount(target, weights=g_edge, minlength=n)
    )
    grad_x += 2.0 * expit(x) - 1.0
    grad_theta = -np.bincount(
        type_index, weights=W * (r - q[type_index]), minlength=NUM_TYPES
    )
    return float(value), np.concatenate([grad_x, grad_theta])


# --------------------------------------------------------------------------
# Method 1: Newman's EM with the Eq. 22 MAP strength update
# --------------------------------------------------------------------------
def run_em(
    data: dict,
    max_iter: int,
    inner_max_iter: int,
    tol: float,
    inner_tol: float,
    seed: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = data["n"]
    source, target = data["source"], data["target"]
    type_index, W = data["type_index"], data["weights"]
    type_counts = data["type_counts"]

    # A_ij: total weight per unordered pair (both directions, both types).
    low, high = np.minimum(source, target), np.maximum(source, target)
    pair_totals = np.bincount(low * n + high, weights=W, minlength=n * n)
    pair_keys = np.flatnonzero(pair_totals)
    pair_i, pair_j = pair_keys // n, pair_keys % n
    pair_weight = pair_totals[pair_keys]
    del pair_totals, pair_keys, low, high

    rng = np.random.default_rng(seed)
    strength = (
        np.exp(x0) if x0 is not None
        else np.exp(rng.normal(0.0, 0.1, size=n))
    )
    q = rng.uniform(0.65, 0.85, size=NUM_TYPES)

    converged = inner_converged = False
    final_strength_change = final_q_change = final_inner_change = float("inf")
    inner_used = 0

    for iteration in range(1, max_iter + 1):
        old_strength, old_q = strength.copy(), q.copy()

        # E-step: posterior that the observed winner is dominant.
        num = strength[source] * q[type_index]
        den = num + strength[target] * (1.0 - q[type_index])
        posterior = num / den

        # M-step for valences.
        q = np.clip(
            np.bincount(type_index, weights=W * posterior, minlength=NUM_TYPES)
            / type_counts,
            1e-10, 1.0 - 1e-10,
        )

        # Expected dominance counts, fixed during the strength update.
        expected_dominance = (
            np.bincount(source, weights=W * posterior, minlength=n)
            + np.bincount(target, weights=W * (1.0 - posterior), minlength=n)
        )
        numerator = 1.0 + expected_dominance

        # Eq. 22 fixed-point iteration.
        inner_converged = False
        for inner in range(1, inner_max_iter + 1):
            contrib = pair_weight / (strength[pair_i] + strength[pair_j])
            interaction = (
                np.bincount(pair_i, weights=contrib, minlength=n)
                + np.bincount(pair_j, weights=contrib, minlength=n)
            )
            new_strength = numerator / (2.0 / (strength + 1.0) + interaction)
            final_inner_change = float(
                np.max(np.abs(np.log(new_strength) - np.log(strength)))
            )
            strength = new_strength
            if final_inner_change < inner_tol:
                inner_converged = True
                break
        inner_used = inner

        final_strength_change = float(
            np.max(np.abs(np.log(strength) - np.log(old_strength)))
        )
        final_q_change = float(np.max(np.abs(q - old_q)))

        if iteration == 1 or iteration % 10 == 0:
            print(
                f"EM {iteration:4d}/{max_iter}: "
                f"strength_change={final_strength_change:.3e}, "
                f"q_change={final_q_change:.3e}, "
                f"q1={q[0]:.6f}, q2={q[1]:.6f}"
            )

        if max(final_strength_change, final_q_change) < tol:
            converged = True
            break

    info = {
        "converged": converged,
        "iterations": iteration,
        "final_strength_change": final_strength_change,
        "final_q_change": final_q_change,
        "inner_converged_last_step": inner_converged,
        "inner_iterations_last_step": inner_used,
        "final_inner_change": final_inner_change,
    }
    return np.log(strength), q, info


# --------------------------------------------------------------------------
# Method 2: direct maximization of the same posterior
# --------------------------------------------------------------------------
def run_direct(
    data: dict,
    max_iter: int,
    gtol: float,
    seed: int,
    x0: np.ndarray | None = None,
    q0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = data["n"]
    source, target, W = data["source"], data["target"], data["weights"]

    if x0 is None:
        # Start in the "winners are strong" basin; the model has a mirror
        # optimum (x, q) -> (-x, 1-q), resolved after fitting.
        wins = np.bincount(source, weights=W, minlength=n)
        losses = np.bincount(target, weights=W, minlength=n)
        z = np.log((wins + 1.0) / (losses + 1.0))
        z = (z - z.mean()) / z.std()
        x0 = 0.5 * z + np.random.default_rng(seed).normal(0.0, 0.01, n)

    theta0 = (
        logit(np.clip(q0, 1e-6, 1 - 1e-6)) if q0 is not None
        else np.full(NUM_TYPES, logit(0.75))
    )

    iteration_count = {"k": 0}

    def callback(_):
        iteration_count["k"] += 1
        if iteration_count["k"] % 100 == 0:
            print(f"L-BFGS iteration {iteration_count['k']}")

    result = minimize(
        neg_log_posterior,
        np.concatenate([x0, theta0]),
        args=(data,),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": max_iter,
            "maxfun": 2 * max_iter,
            "maxcor": 30,
            "ftol": 1e-15,
            "gtol": gtol,
            "maxls": 50,
        },
    )

    info = {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "optimizer_message": str(result.message),
    }
    return result.x[:n], expit(result.x[n:]), info


# --------------------------------------------------------------------------
# Finalization
# --------------------------------------------------------------------------
def finalize(
    data: dict, x: np.ndarray, q: np.ndarray, info: dict, method: str
) -> tuple[pd.DataFrame, dict]:
    n, W = data["n"], data["weights"]

    # Resolve the (x, q) -> (-x, 1-q) mirror symmetry first.
    # Use the same orientation for saved scores and diagnostics.
    if np.average(q, weights=data["type_counts"]) < 0.5:
        x, q = -x, 1.0 - q

    # Report posterior value and gradient for the final orientation.
    theta = logit(np.clip(q, 1e-10, 1 - 1e-10))
    nlp, grad = neg_log_posterior(np.concatenate([x, theta]), data)

    # Resolve the (x, q) -> (-x, 1-q) mirror symmetry: choose the
    # orientation in which observed winners are more likely dominant.
    if np.average(q, weights=data["type_counts"]) < 0.5:
        x, q = -x, 1.0 - q

    scores = pd.DataFrame(
        {
            "mapped_movie_id": data["node_ids"],
            "strength": np.exp(x),
            "score": x,
        }
    )
    scores["rank"] = (
        scores["score"].rank(method="min", ascending=False).astype(int)
    )
    scores = scores.sort_values(["rank", "mapped_movie_id"]).reset_index(
        drop=True
    )

    diagnostics = {
        "method": method,
        **info,
        "neg_log_posterior": nlp,
        "max_abs_gradient": float(np.max(np.abs(grad))),
        "comparison_type_1_q": float(q[0]),
        "comparison_type_2_q": float(q[1]),
        "num_movies": int(n),
        "num_typed_edges": int(len(W)),
        "total_comparison_weight": int(W.sum()),
        "type_1_comparison_weight": int(data["type_counts"][0]),
        "type_2_comparison_weight": int(data["type_counts"][1]),
        "min_strength": float(scores["strength"].min()),
        "median_strength": float(scores["strength"].median()),
        "max_strength": float(scores["strength"].max()),
    }
    return scores, diagnostics


def fit_newman_ranking(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    method: str = "direct",
    max_iter: int = 5000,
    inner_max_iter: int = 200,
    tol: float = 1e-6,
    inner_tol: float = 1e-7,
    gtol: float = 1e-2,
    seed: int = 42,
    init_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Estimate movie strengths and comparison-type valences (MAP)."""
    data = prepare_data(nodes, edges)

    x0 = None
    if init_scores is not None:
        lookup = init_scores.set_index("mapped_movie_id")["score"]
        x0 = lookup.reindex(data["node_ids"]).to_numpy(dtype=np.float64)
        if np.isnan(x0).any():
            raise ValueError("init_scores does not cover every movie.")

    if method == "direct":
        x, q, info = run_direct(data, max_iter, gtol, seed, x0=x0)
    elif method == "em":
        x, q, info = run_em(
            data, max_iter, inner_max_iter, tol, inner_tol, seed, x0=x0
        )
    else:
        raise ValueError("method must be 'direct' or 'em'.")

    return finalize(data, x, q, info, method)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Newman's typed pairwise ranking (MAP estimate)."
    )
    parser.add_argument("--input-dir", default="data/processed/ml-1m/movie_network")
    parser.add_argument("--method", choices=["direct", "em"], default="direct")
    parser.add_argument("--max-iter", type=int, default=5000,
                        help="L-BFGS iterations (direct) or EM iterations (em).")
    parser.add_argument("--inner-max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--inner-tol", type=float, default=1e-7)
    parser.add_argument("--gtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-scores", default=None,
                        help="Optional movie_scores.csv to warm-start from.")
    parser.add_argument("--output-name", default=None,
                        help="Defaults to movie_scores.csv (direct) or "
                             "movie_scores_em.csv (em).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    nodes = pd.read_csv(input_dir / "nodes.csv", dtype={"mapped_movie_id": "int32"})
    edges = pd.read_csv(
        input_dir / "edges.csv",
        dtype={"source": "int32", "target": "int32",
               "comparison_type": "int8", "weight": "int32"},
    )
    init_scores = pd.read_csv(args.init_scores) if args.init_scores else None

    print(f"Fitting Newman's ranking (method={args.method})...")
    print(f"Movies: {len(nodes):,}  Typed edges: {len(edges):,}")

    scores, diagnostics = fit_newman_ranking(
        nodes, edges,
        method=args.method,
        max_iter=args.max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        inner_tol=args.inner_tol,
        gtol=args.gtol,
        seed=args.seed,
        init_scores=init_scores,
    )

    output_name = args.output_name or (
        "movie_scores.csv" if args.method == "direct" else "movie_scores_em.csv"
    )
    output_path = input_dir / output_name
    scores.to_csv(output_path, index=False)

    print("\nRanking diagnostics")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print(f"\nSaved movie scores to: {output_path}")
    print("\nTop 10 movies by inferred global strength:")
    print(scores.head(10).to_string(index=False))

    if not diagnostics["converged"]:
        print("\nWARNING: Optimizer did not report convergence. "
              "Check max_abs_gradient before treating this as final.")


if __name__ == "__main__":
    main()