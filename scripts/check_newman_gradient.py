"""
check_newman_gradient.py

Unit test for the gradient of the newman_ranking objective.

Run with `PYTHONPATH=. python scripts/check_newman_gradient.py`.
"""
import numpy as np
import pandas as pd

from src.network.newman_ranking import prepare_data, neg_log_posterior


def main():
    nodes = pd.DataFrame({
        "mapped_movie_id": [10, 11, 12, 13]
    })

    # source is the observed winner; target is the observed loser.
    edges = pd.DataFrame({
        "source": [10, 10, 11, 12, 13, 11],
        "target": [11, 12, 12, 13, 10, 13],
        "comparison_type": [1, 2, 1, 2, 1, 2],
        "weight": [3, 2, 5, 4, 2, 3],
    })

    data = prepare_data(nodes, edges)

    # Four log-strengths followed by two valence logits.
    params = np.array([
        0.4, -0.3, 0.8, -0.6,
        0.7, 1.1,
    ], dtype=np.float64)

    value, analytic_grad = neg_log_posterior(params, data)

    # Central finite differences.
    eps = 1e-6
    numerical_grad = np.zeros_like(params)

    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[i] += eps
        minus[i] -= eps

        f_plus, _ = neg_log_posterior(plus, data)
        f_minus, _ = neg_log_posterior(minus, data)

        numerical_grad[i] = (f_plus - f_minus) / (2 * eps)

    abs_error = np.abs(analytic_grad - numerical_grad)
    max_abs_error = abs_error.max()
    max_rel_error = np.max(
        abs_error / np.maximum(
            1e-8,
            np.maximum(np.abs(analytic_grad), np.abs(numerical_grad))
        )
    )

    print(f"Objective: {value:.10f}")
    print(f"Maximum absolute gradient error: {max_abs_error:.3e}")
    print(f"Maximum relative gradient error: {max_rel_error:.3e}")
    print()
    print("Parameter | Analytic gradient | Numerical gradient")
    for i, (a, n) in enumerate(zip(analytic_grad, numerical_grad)):
        print(f"{i:9d} | {a:17.10f} | {n:19.10f}")

    if max_abs_error < 1e-5:
        print("\nPASS: analytic gradient matches finite differences.")
    else:
        raise AssertionError(
            "Gradient check failed. Inspect the objective and derivatives."
        )


if __name__ == "__main__":
    main()
