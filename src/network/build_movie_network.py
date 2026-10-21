
"""Construct a typed movie-comparison network from training ratings."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


def build_movie_network(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build directed, typed comparison edges.

    Type 1: ratings differ by exactly one star.
    Type 2: ratings differ by at least two stars.

    Each comparison is directed from the higher-rated movie
    to the lower-rated movie. Edge weights count distinct
    users' comparisons of that type.
    """
    required = {"user_id", "mapped_movie_id", "rating"}
    missing = required - set(ratings.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if ratings[list(required)].isna().any().any():
        raise ValueError("Ratings contain missing required values.")

    if ratings.duplicated(["user_id", "mapped_movie_id"]).any():
        raise ValueError(
            "Duplicate user/movie ratings found; resolve them first."
        )

    edge_counts: Counter = Counter()

    # Each user's training ratings provide pairwise comparisons.
    for _, group in ratings.groupby("user_id", sort=False):
        items = list(
            group[["mapped_movie_id", "rating"]].itertuples(
                index=False, name=None
            )
        )

        for (movie_a, rating_a), (movie_b, rating_b) in combinations(
            items, 2
        ):
            difference = abs(float(rating_a) - float(rating_b))

            # Equal ratings do not express a strict preference.
            if difference == 0:
                continue

            # Orient the edge from winner to lower-rated movie.
            if rating_a > rating_b:
                winner, loser = int(movie_a), int(movie_b)
            else:
                winner, loser = int(movie_b), int(movie_a)

            comparison_type = 1 if difference == 1 else 2

            edge_counts[(winner, loser, comparison_type)] += 1

    edges = pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "comparison_type": comparison_type,
                "weight": weight,
            }
            for (source, target, comparison_type), weight
            in edge_counts.items()
        ],
        columns=[
            "source",
            "target",
            "comparison_type",
            "weight",
        ],
    )

    # Include every movie observed in training, even if it has
    # no unequal-rating comparisons.
    nodes = pd.DataFrame(
        {
            "mapped_movie_id": sorted(
                ratings["mapped_movie_id"].astype(int).unique()
            )
        }
    )

    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/interim/ml-1m/network_train_ratings.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/ml-1m/movie_network",
    )
    args = parser.parse_args()

    ratings = pd.read_parquet(args.input)
    nodes, edges = build_movie_network(ratings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)

    print("Movie network construction complete.")
    print(f"Training ratings: {len(ratings):,}")
    print(f"Movies (nodes): {len(nodes):,}")
    print(f"Typed directed edges: {len(edges):,}")

    for comparison_type in (1, 2):
        typed_edges = edges[
            edges["comparison_type"] == comparison_type
        ]
        print(
            f"Type {comparison_type} edges: "
            f"{len(typed_edges):,}; "
            f"total comparison weight: "
            f"{typed_edges['weight'].sum():,}"
        )

    print(f"Saved network artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
