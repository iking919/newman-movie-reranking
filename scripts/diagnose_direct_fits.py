from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from src.network.newman_ranking import neg_log_posterior, prepare_data

ROOT = Path("data/processed/ml-1m/movie_network")

FITS = {
    "canonical": (
        "movie_scores_direct.csv",
        [0.8512651038075252, 0.9934124987290911],
    ),
    "seed1": (
        "movie_scores_direct_seed1.csv",
        [0.8574451618539487, 0.9999999998334357],
    ),
    "seed7": (
        "movie_scores_direct_seed7.csv",
        [0.8574451588064305, 0.9999999998906588],
    ),
}

nodes = pd.read_csv(ROOT / "nodes.csv")
edges = pd.read_csv(ROOT / "edges.csv")
data = prepare_data(nodes, edges)
n = data["n"]


def evaluate(scores, q):
    scores = scores.set_index("mapped_movie_id").reindex(data["node_ids"])
    if scores["score"].isna().any():
        raise ValueError("A score file is missing movie IDs.")

    x = scores["score"].to_numpy(dtype=np.float64)
    theta = logit(np.clip(q, 1e-10, 1 - 1e-10))
    value, grad = neg_log_posterior(np.concatenate([x, theta]), data)

    return value, np.max(np.abs(grad)), grad[n:]


print("=== Saved-fit objective check ===")
loaded = {}

for name, (filename, q_values) in FITS.items():
    scores = pd.read_csv(ROOT / filename)
    q = np.array(q_values)
    value, max_grad, grad_q = evaluate(scores, q)
    loaded[name] = (scores, q)

    print(
        f"{name}: objective={value:.6f}, "
        f"max_abs_gradient={max_grad:.6g}, "
        f"gradient_theta1={grad_q[0]:.6g}, "
        f"gradient_theta2={grad_q[1]:.6g}"
    )


print("\n=== q2 profile (scores and q1 held fixed) ===")
q2_values = [0.99, 0.9934124987, 0.999, 0.9999, 0.999999, 0.999999999]

for name, (scores, q) in loaded.items():
    print(f"\n{name}:")
    for q2 in q2_values:
        value, max_grad, grad_q = evaluate(scores, np.array([q[0], q2]))
        print(
            f"  q2={q2:.10f}  objective={value:.6f}  "
            f"gradient_theta2={grad_q[1]:.6g}"
        )
