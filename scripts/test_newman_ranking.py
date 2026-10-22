"""
test_newman_ranking.py

Unit tests for the newman_ranking module.

Run with `PYTHONPATH=. python scripts/test_newman_ranking.py`.
"""

import pandas as pd

from src.network.newman_ranking import fit_newman_ranking

nodes = pd.DataFrame({
    "mapped_movie_id": [2, 3, 4, 5],
})

# Movie 2 consistently wins; movie 5 consistently loses.
# Include both comparison types.
edges = pd.DataFrame([
    {"source": 2, "target": 3, "comparison_type": 1, "weight": 20},
    {"source": 2, "target": 4, "comparison_type": 2, "weight": 20},
    {"source": 2, "target": 5, "comparison_type": 2, "weight": 20},
    {"source": 3, "target": 4, "comparison_type": 1, "weight": 10},
    {"source": 3, "target": 5, "comparison_type": 2, "weight": 15},
    {"source": 4, "target": 5, "comparison_type": 1, "weight": 20},
])

scores, diagnostics = fit_newman_ranking(
    nodes,
    edges,
    seed=42,
    max_iter=500,
    inner_max_iter=500,
)

print(scores.to_string(index=False))
print(diagnostics)

assert scores.iloc[0]["mapped_movie_id"] == 2
assert scores.iloc[-1]["mapped_movie_id"] == 5
assert scores["score"].notna().all()
assert scores["strength"].gt(0).all()
