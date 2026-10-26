"""
build_global_ranking_baselines.py

Compute global ranking baselines for movies in the training set, including:
- Mean rating
- Shrunk mean rating (with explicit shrinkage strength m=20)
- Popularity (log of rating count)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ratings_path = Path("data/interim/ml-1m/network_train_ratings.parquet")
newman_path = Path(
    "data/processed/ml-1m/movie_network/movie_scores_direct.csv"
)
output_path = Path(
    "data/processed/ml-1m/movie_network/global_ranking_baselines.csv"
)

ratings = pd.read_parquet(ratings_path)
newman = pd.read_csv(newman_path)

required = {"mapped_movie_id", "rating"}
missing = required - set(ratings.columns)
if missing:
    raise ValueError(f"Missing rating columns: {sorted(missing)}")

stats = ratings.groupby("mapped_movie_id")["rating"].agg(
    rating_count="count",
    mean_rating="mean",
).reset_index()

global_mean = ratings["rating"].mean()
m = 20  # Explicit shrinkage strength: equivalent to 20 prior ratings.

stats["mean_rating_score"] = stats["mean_rating"]
stats["shrunk_mean_score"] = (
    stats["rating_count"] * stats["mean_rating"] + m * global_mean
) / (stats["rating_count"] + m)

stats["popularity_score"] = np.log1p(stats["rating_count"])

# Standardize each score across movies for a fair initial comparison.
score_columns = [
    "mean_rating_score",
    "shrunk_mean_score",
    "popularity_score",
]

for column in score_columns:
    std = stats[column].std(ddof=0)
    if std == 0:
        raise ValueError(f"Cannot standardize constant score: {column}")
    stats[column] = (stats[column] - stats[column].mean()) / std

newman_required = {"mapped_movie_id", "score"}
if not newman_required.issubset(newman.columns):
    raise ValueError(
        f"Newman CSV needs {sorted(newman_required)}; "
        f"found {newman.columns.tolist()}"
    )

newman_scores = newman[["mapped_movie_id", "score"]].rename(
    columns={"score": "newman_score"}
)
newman_scores["newman_score"] = (
    newman_scores["newman_score"]
    - newman_scores["newman_score"].mean()
) / newman_scores["newman_score"].std(ddof=0)

result = stats.merge(
    newman_scores, on="mapped_movie_id", how="left", validate="one_to_one"
)

if result["newman_score"].isna().any():
    raise ValueError("Some movies are missing Newman scores.")

output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)

print(f"Training-set global mean rating: {global_mean:.4f}")
print(f"Shrinkage strength m: {m}")
print(f"Movies: {len(result)}")
print(f"Saved: {output_path}")
print("\nCorrelation with Newman score:")
print(
    result[
        [
            "newman_score",
            "mean_rating_score",
            "shrunk_mean_score",
            "popularity_score",
        ]
    ].corr()["newman_score"].to_string()
)
print("\nTop 10 by shrunk mean rating:")
print(
    result.sort_values("shrunk_mean_score", ascending=False)[
        [
            "mapped_movie_id",
            "rating_count",
            "mean_rating",
            "newman_score",
            "shrunk_mean_score",
        ]
    ].head(10).to_string(index=False)
)
