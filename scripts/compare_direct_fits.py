"""
compare_direct_fits.py

Compare multiple direct fit results for the same movie network.
Run with `PYTHONPATH=. python scripts/compare_direct_fits.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("data/processed/ml-1m/movie_network")

# The first fit is the canonical reference; subsequent fits are compared against it.
FILES = [
    "movie_scores_direct.csv",
    "movie_scores_direct_seed1.csv",
    "movie_scores_direct_seed7.csv",
]

fits = {}

# Load and validate each available fit before doing any comparisons.
for filename in FILES:
    path = BASE / filename

    if not path.exists():
        print(f"\nMISSING FILE: {path}")
        continue

    df = pd.read_csv(path)

    # These columns define the movie identity and the values used below.
    required = {"mapped_movie_id", "strength", "score", "rank"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")

    if df["mapped_movie_id"].duplicated().any():
        raise ValueError(f"Duplicate movie IDs in {filename}")

    df = df.set_index("mapped_movie_id").sort_index()

    # Strength should be the exponentiated score; flag mismatches without
    # stopping the comparison so the remaining results are still available.
    if not np.allclose(
        df["strength"].to_numpy(),
        np.exp(df["score"].to_numpy()),
        rtol=1e-5,
        atol=1e-10,
    ):
        print(f"WARNING: strength != exp(score) in {filename}")

    fits[filename] = df

    print(f"\n{'=' * 70}")
    print(filename)
    print(f"Movies: {len(df)}")
    print(f"Minimum strength: {df['strength'].min():.10g}")
    print(f"Median strength:  {df['strength'].median():.10g}")
    print(f"Maximum strength: {df['strength'].max():.10g}")
    print(f"Minimum score:    {df['score'].min():.10g}")
    print(f"Maximum score:    {df['score'].max():.10g}")

    print("\nTop 10:")
    print(
        df.sort_values("rank")
        .head(10)[["rank", "strength", "score"]]
        .to_string()
    )

if FILES[0] not in fits:
    raise FileNotFoundError(
        f"Canonical fit missing: {BASE / FILES[0]}"
    )

reference_name = FILES[0]
reference = fits[reference_name]

summary_rows = []
comparison_frames = []

# Compare every available non-reference fit to the same canonical baseline.
for filename in FILES[1:]:
    if filename not in fits:
        continue

    other = fits[filename]

    if set(reference.index) != set(other.index):
        print(f"\nERROR: Movie ID sets differ for {filename}")
        print("Only in reference:", len(set(reference.index) - set(other.index)))
        print("Only in other:", len(set(other.index) - set(reference.index)))
        continue

    # Align rows by movie ID so score and rank differences compare the same movies.
    other = other.reindex(reference.index)

    # Spearman correlation measures whether the score ordering is consistent.
    rho = spearmanr(
        reference["score"],
        other["score"],
    ).statistic

    rank_diff = (
        reference["rank"] - other["rank"]
    ).abs()

    # Count shared movie IDs among the top-k movies in each fit.
    top_results = {}

    for k in (10, 25, 50, 100):
        ref_top = set(reference.nsmallest(k, "rank").index)
        other_top = set(other.nsmallest(k, "rank").index)

        top_results[f"top_{k}_overlap"] = len(ref_top & other_top)

    # Store per-movie values for the detailed output and largest-change report.
    comparison = pd.DataFrame(index=reference.index)

    comparison["reference_rank"] = reference["rank"]
    comparison["other_rank"] = other["rank"]
    comparison["absolute_rank_difference"] = rank_diff
    comparison["reference_score"] = reference["score"]
    comparison["other_score"] = other["score"]
    comparison["score_difference"] = (
        other["score"] - reference["score"]
    )
    comparison["reference_strength"] = reference["strength"]
    comparison["other_strength"] = other["strength"]

    comparison["absolute_score_difference"] = (
        comparison["score_difference"].abs()
    )

    # Since log(strength) == score, this difference is identical to |score delta|.
    comparison["absolute_log_strength_difference"] = (
        comparison["absolute_score_difference"]
    )

    comparison_frames.append(
        comparison.assign(comparison=filename)
        .reset_index()
    )

    # Aggregate fit-to-fit metrics for one row in the summary CSV.
    row = {
        "comparison": filename,
        "spearman_score": rho,
        "max_absolute_rank_difference": rank_diff.max(),
        "median_absolute_rank_difference": rank_diff.median(),
        "mean_absolute_rank_difference": rank_diff.mean(),
        "max_absolute_score_difference": (
            comparison["absolute_score_difference"].max()
        ),
        "median_absolute_score_difference": (
            comparison["absolute_score_difference"].median()
        ),
        "max_strength_reference": reference["strength"].max(),
        "max_strength_other": other["strength"].max(),
        **top_results,
    }

    summary_rows.append(row)

    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {reference_name} vs {filename}")
    print(f"Spearman score correlation: {rho:.12f}")
    print(f"Maximum absolute rank difference: {rank_diff.max()}")
    print(f"Median absolute rank difference: {rank_diff.median()}")
    print(f"Mean absolute rank difference: {rank_diff.mean():.6f}")
    print(
        "Maximum absolute score difference: "
        f"{comparison['absolute_score_difference'].max():.8f}"
    )
    print(
        "Median absolute score difference: "
        f"{comparison['absolute_score_difference'].median():.8f}"
    )

    for k in (10, 25, 50, 100):
        print(f"Top-{k} overlap: {top_results[f'top_{k}_overlap']}/{k}")

    print("\nMovies with largest absolute score changes:")
    print(
        comparison.nlargest(10, "absolute_score_difference")[
            [
                "reference_rank",
                "other_rank",
                "reference_score",
                "other_score",
                "score_difference",
            ]
        ].to_string()
    )

if summary_rows:
    # Write aggregate metrics only when at least one comparison was possible.
    summary_path = BASE / "direct_fit_comparison_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")

if comparison_frames:
    # Preserve the movie-level comparisons for downstream inspection.
    details_path = BASE / "direct_fit_comparison_details.csv"
    pd.concat(comparison_frames, ignore_index=True).to_csv(
        details_path, index=False
    )
    print(f"Saved per-movie comparison details: {details_path}")
