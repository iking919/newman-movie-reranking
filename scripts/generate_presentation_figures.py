"""
Generate all presentation-only figures and tables for the Newman Movie Reranking project.

Run from the repository root:
    PYTHONPATH=. python scripts/generate_presentation_figures.py

Reads existing experiment/network outputs only. Does not train models, rerun
experiments, or modify source results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results/bert4rec_reranking"
NETWORK_DIR = ROOT / "data/processed/ml-1m/movie_network"
FIGURE_DIR = ROOT / "presentation/figures"
TABLE_DIR = ROOT / "presentation/tables"

BOOTSTRAP_PATH = RESULTS_DIR / "paired_user_bootstrap.json"
SEGMENT_PATH = RESULTS_DIR / "popularity_segment_summary.csv"
TUNED_RESULTS_PATH = RESULTS_DIR / "reliability_experiment/tuned_results.json"
VALIDATION_GRID_PATH = RESULTS_DIR / "reliability_experiment/tuned_validation_grid.csv"
EDGES_PATH = NETWORK_DIR / "edges.csv"
SCORES_PATH = NETWORK_DIR / "movie_scores_direct.csv"

METHOD_ORDER = [
    "newman", "mean_rating", "shrunk_mean_rating", "popularity",
    "newman_shrink_k10", "newman_shrink_k25",
    "newman_shrink_k50", "newman_shrink_k100",
]
METHOD_NAMES = {
    "newman": "Newman",
    "mean_rating": "Mean rating",
    "shrunk_mean_rating": "Shrunk mean rating",
    "popularity": "Popularity",
    "newman_shrink_k10": "Newman shrink (k=10)",
    "newman_shrink_k25": "Newman shrink (k=25)",
    "newman_shrink_k50": "Newman shrink (k=50)",
    "newman_shrink_k100": "Newman shrink (k=100)",
}
QUARTILE_ORDER = ["Q1_least_popular", "Q2", "Q3", "Q4_most_popular"]
QUARTILE_NAMES = {
    "Q1_least_popular": "Q1 — least popular",
    "Q2": "Q2",
    "Q3": "Q3",
    "Q4_most_popular": "Q4 — most popular",
}


def require_files():
    required = [
        BOOTSTRAP_PATH, SEGMENT_PATH, TUNED_RESULTS_PATH,
        VALIDATION_GRID_PATH, EDGES_PATH, SCORES_PATH,
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required project outputs:\n  - " + "\n  - ".join(missing)
        )


def load_inputs():
    require_files()
    with BOOTSTRAP_PATH.open(encoding="utf-8") as f:
        bootstrap = json.load(f)
    with TUNED_RESULTS_PATH.open(encoding="utf-8") as f:
        tuned_results = json.load(f)
    segments = pd.read_csv(SEGMENT_PATH)
    validation = pd.read_csv(VALIDATION_GRID_PATH)
    # Aggregate the 8M-row edge file in chunks so the presentation script
    # does not need to hold the full network edge table in memory.
    edge_summary = {}
    for chunk in pd.read_csv(
        EDGES_PATH, usecols=["comparison_type", "weight"], chunksize=500_000
    ):
        for comparison_type, group in chunk.groupby("comparison_type"):
            current = edge_summary.setdefault(int(comparison_type), {"count": 0, "sum": 0})
            current["count"] += int(len(group))
            current["sum"] += int(group["weight"].sum())

    scores = pd.read_csv(SCORES_PATH)

    required_segment_columns = {
        "grouping", "segment", "n_users", "delta_HR@10", "delta_NDCG@10"
    }
    missing = required_segment_columns.difference(segments.columns)
    if missing:
        raise ValueError(f"Popularity segment CSV missing columns: {sorted(missing)}")

    if set(METHOD_ORDER).difference(bootstrap["comparisons"]):
        raise ValueError("Bootstrap JSON is missing one or more expected methods.")
    if not {"method", "lambda", "NDCG@10"}.issubset(validation.columns):
        raise ValueError("Validation grid must contain method, lambda, and NDCG@10.")
    if not {1, 2}.issubset(set(edge_summary)):
        raise ValueError("Network edge file must contain comparison types 1 and 2.")
    if not {"strength", "score"}.issubset(scores.columns):
        raise ValueError("Newman score file is missing strength and/or score.")

    return bootstrap, tuned_results, segments, validation, edge_summary, scores


def save_figure(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {path.relative_to(ROOT)}")


def generate_bootstrap_figure(bootstrap):
    """Paired-user bootstrap effects and 95% percentile confidence intervals."""
    comparisons = bootstrap["comparisons"]
    metrics = [("HR@10", "HR@10"), ("NDCG@10", "NDCG@10")]
    fig, axes = plt.subplots(ncols=2, figsize=(15, 8), sharey=True)
    y = np.arange(len(METHOD_ORDER))

    for ax, (metric_key, title) in zip(axes, metrics):
        effects, low_err, high_err = [], [], []
        for method in METHOD_ORDER:
            metric = comparisons[method]["metrics"][metric_key]
            effect = metric["difference_reranked_minus_baseline"]
            low, high = metric["ci_95_percentile"]
            effects.append(effect * 100)
            low_err.append((effect - low) * 100)
            high_err.append((high - effect) * 100)

        ax.errorbar(
            effects, y, xerr=[low_err, high_err], fmt="o",
            capsize=4, markersize=6, linewidth=1.5,
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Change from BERT4Rec baseline (percentage points)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([
            f"{METHOD_NAMES[m]} (λ={comparisons[m]['validation_selected_lambda']:g})"
            for m in METHOD_ORDER
        ])
        ax.invert_yaxis()

    fig.suptitle("Reranking Performance: Paired-User Bootstrap",
                 fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.015,
        f"95% percentile paired-user bootstrap CIs; {bootstrap['n_users']:,} users; "
        f"{bootstrap['bootstrap_replicates']:,} replicates. Weights selected on "
        "validation NDCG@10. Intervals are not adjusted for method/lambda selection.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    save_figure(fig, "paired_bootstrap_comparison.png")


def generate_popularity_figure(segments):
    """Descriptive popularity-reranking effects across held-out target quartiles."""
    q = segments[segments["grouping"] == "target_popularity_quartile"].copy()
    q["segment"] = pd.Categorical(q["segment"], categories=QUARTILE_ORDER, ordered=True)
    q = q.sort_values("segment")
    if len(q) != 4:
        raise ValueError(f"Expected four target-popularity quartiles; found {len(q)}.")

    labels = [QUARTILE_NAMES[str(s)] for s in q["segment"]]
    metrics = [("delta_HR@10", "Change in HR@10"),
               ("delta_NDCG@10", "Change in NDCG@10")]
    fig, axes = plt.subplots(ncols=2, figsize=(13, 6))
    y = np.arange(4)
    hatches = ["///", "\\\\\\", "...", "xxx"]

    for ax, (column, title) in zip(axes, metrics):
        values = q[column].to_numpy(dtype=float) * 100
        bars = ax.barh(y, values, height=0.62, edgecolor="black", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Change from BERT4Rec baseline (percentage points)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        max_abs = max(abs(values.min()), abs(values.max()))
        margin = max_abs * 0.32 if max_abs else 0.01
        ax.set_xlim(values.min() - margin, values.max() + margin)
        for bar, value in zip(bars, values):
            offset = max_abs * 0.035 if max_abs else 0.001
            ax.text(value + offset if value >= 0 else value - offset,
                    bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                    va="center", ha="left" if value >= 0 else "right", fontsize=10)

    fig.suptitle("Popularity Reranking by Held-Out Target Popularity",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.015,
        "Descriptive subgroup analysis; 1,510 users per quartile. Quartiles are based "
        "on training interaction counts of each user's held-out target item. "
        "No subgroup confidence intervals are shown.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_figure(fig, "popularity_quartile_analysis.png")


def generate_main_metric_comparison(tuned_results):
    """Compare baseline, Newman reranking, and popularity reranking on all four test metrics."""
    test = tuned_results["test_metrics"]
    methods = [
        ("bert4rec", "BERT4Rec"),
        ("newman", f"Newman (λ={tuned_results['selected_weights']['newman']:g})"),
        ("popularity", f"Popularity (λ={tuned_results['selected_weights']['popularity']:g})"),
    ]
    metrics = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    x = np.arange(len(methods))
    hatches = ["///", "\\\\\\", "..."]

    for ax, metric in zip(axes, metrics):
        values = [test[key][metric] for key, _ in methods]
        bars = ax.bar(x, values, edgecolor="black", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in methods], rotation=12, ha="right")
        ax.set_ylabel("Test score")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        upper = max(values) * 1.17
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + upper*0.015,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Test-Set Recommendation Performance",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015,
             "Reranking weights selected on validation NDCG@10; values shown are test metrics.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, "main_metric_comparison.png")


def generate_network_construction_diagram():
    """Conceptual schematic; illustrates construction logic, not actual movie identities."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, linestyle="-"):
        # Different line styles distinguish the two comparison types without
        # relying on color or placing hatch marks behind the text.
        patch = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8,
                              edgecolor="black", linestyle=linestyle)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.70, title, ha="center", va="center",
                fontsize=10.5 if len(title) > 22 else 12, fontweight="bold")
        ax.text(x + w/2, y + h*0.34, body, ha="center", va="center",
                fontsize=9, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linewidth=1.6))

    box(0.25, 2.0, 2.2, 2.0, "User histories",
        "Ordered movie\nsequences\n(training data)")
    box(3.05, 3.55, 2.55, 1.65, "Type 1 comparison",
        "Consecutive interactions\n(gap = 1)", "--")
    box(3.05, 0.8, 2.55, 1.65, "Type 2 comparison",
        "Non-consecutive interactions\n(gap ≥ 2)", ":")
    box(6.35, 2.0, 2.4, 2.0, "Weighted edges",
        "Ordered movie pair (i, j)\nEdge weight = count\nof observed comparisons")
    box(9.55, 2.0, 3.1, 2.0, "Pairwise-comparison\nnetwork",
        "3,416 movie nodes\n8,026,782 typed edge rows\n33,471,393 total weight")

    arrow(2.45, 3.0, 3.05, 4.25)
    arrow(2.45, 3.0, 3.05, 1.6)
    arrow(5.6, 4.35, 6.35, 3.35)
    arrow(5.6, 1.6, 6.35, 2.65)
    arrow(8.75, 3.0, 9.55, 3.0)

    ax.text(6.5, 0.35,
            "Ties omitted • comparisons constructed from training interactions only",
            ha="center", va="center", fontsize=10, style="italic")
    fig.suptitle("From Sequential Movie Interactions to Pairwise Evidence",
                 fontsize=16, fontweight="bold", y=0.97)
    save_figure(fig, "network_construction_diagram.png")


def generate_strength_distribution(scores):
    """Distribution of fitted Newman movie strengths, shown on a log10 scale."""
    strengths = scores["strength"].to_numpy(dtype=float)
    strengths = strengths[np.isfinite(strengths) & (strengths > 0)]
    if len(strengths) == 0:
        raise ValueError("No positive finite Newman strengths found.")

    log_strengths = np.log10(strengths)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_strengths, bins=45, edgecolor="black", linewidth=0.6)
    ax.set_title("Distribution of Estimated Newman Movie Strengths",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("log10(movie strength)")
    ax.set_ylabel("Number of movies")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    median = np.median(log_strengths)
    ax.axvline(median, linestyle="--", linewidth=1.5,
               label=f"Median strength = {10**median:.3f}")
    ax.legend()
    ax.text(0.99, 0.96, f"N = {len(strengths):,} movies",
            transform=ax.transAxes, ha="right", va="top")
    fig.text(0.5, 0.01,
             "Strengths are fitted global ranking parameters (πᵢ); horizontal axis is logarithmic.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "newman_strength_distribution.png")


def generate_comparison_type_breakdown(edge_summary):
    """Show comparison evidence by type using edge-row counts and total weights."""
    labels = ["Type 1\nConsecutive (gap=1)", "Type 2\nNon-consecutive (gap≥2)"]
    counts = [edge_summary[1]["count"], edge_summary[2]["count"]]
    weights = [edge_summary[1]["sum"], edge_summary[2]["sum"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    x = np.arange(2)
    hatches = ["///", "..."]

    for ax, values, title, ylabel, formatter in [
        (axes[0], counts, "Number of typed edge rows", "Edge rows", lambda v: f"{v/1e6:.2f}M"),
        (axes[1], weights, "Total comparison weight", "Aggregate weight", lambda v: f"{v/1e6:.2f}M"),
    ]:
        bars = ax.bar(x, values, edgecolor="black", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(0, max(values) * 1.18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*0.02,
                    formatter(value), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Pairwise Evidence by Comparison Type",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015,
             "Type 1: consecutive training interactions. Type 2: non-consecutive training interactions. "
             "Ties are omitted.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    save_figure(fig, "comparison_type_breakdown.png")


def generate_validation_lambda_curves(validation):
    """Validation NDCG@10 versus lambda for the four primary signal methods."""
    methods = ["newman", "mean_rating", "shrunk_mean_rating", "popularity"]
    labels = {
        "newman": "Newman",
        "mean_rating": "Mean rating",
        "shrunk_mean_rating": "Shrunk mean rating",
        "popularity": "Popularity",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]

    for method, marker in zip(methods, markers):
        subset = validation[validation["method"] == method].sort_values("lambda")
        if subset.empty:
            raise ValueError(f"Validation grid contains no rows for method '{method}'.")
        ax.plot(subset["lambda"], subset["NDCG@10"], marker=marker,
                linewidth=1.7, markersize=5, label=labels[method])

    ax.set_title("Validation NDCG@10 Across Reranking Weights",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Reranking weight (λ)")
    ax.set_ylabel("Validation NDCG@10")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(title="Global signal")
    ax.text(0.99, 0.02,
            "Weights selected by maximum validation NDCG@10; test set not used for selection.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "validation_lambda_curves.png")


def generate_results_tables(bootstrap, tuned_results):
    """Create a compact test-results table and a detailed CSV."""
    comparisons = bootstrap["comparisons"]
    test = tuned_results["test_metrics"]
    rows = []
    for method in METHOD_ORDER:
        result = comparisons[method]
        hr = result["metrics"]["HR@10"]
        ndcg = result["metrics"]["NDCG@10"]
        rows.append({
            "method": METHOD_NAMES[method],
            "lambda": result["validation_selected_lambda"],
            "test_HR@5": test[method]["HR@5"],
            "test_HR@10": test[method]["HR@10"],
            "delta_HR@10": hr["difference_reranked_minus_baseline"],
            "test_NDCG@5": test[method]["NDCG@5"],
            "test_NDCG@10": test[method]["NDCG@10"],
            "delta_NDCG@10": ndcg["difference_reranked_minus_baseline"],
            "HR@10_CI_lower": hr["ci_95_percentile"][0],
            "HR@10_CI_upper": hr["ci_95_percentile"][1],
            "NDCG@10_CI_lower": ndcg["ci_95_percentile"][0],
            "NDCG@10_CI_upper": ndcg["ci_95_percentile"][1],
        })

    csv_path = TABLE_DIR / "main_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created: {csv_path.relative_to(ROOT)}")

    md_path = TABLE_DIR / "main_results.md"
    baseline = test["bert4rec"]
    lines = [
        "# Main Test Results", "",
        "All reranking weights were selected using validation NDCG@10. "
        "Results below are test-set metrics.", "",
        "| Method | λ | HR@5 | HR@10 | NDCG@5 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| BERT4Rec baseline | — | {baseline['HR@5']:.6f} | "
        f"{baseline['HR@10']:.6f} | {baseline['NDCG@5']:.6f} | "
        f"{baseline['NDCG@10']:.6f} |",
    ]
    for method, row in zip(METHOD_ORDER, rows):
        lines.append(
            f"| {row['method']} | {row['lambda']:g} | "
            f"{row['test_HR@5']:.6f} | {row['test_HR@10']:.6f} | "
            f"{row['test_NDCG@5']:.6f} | {row['test_NDCG@10']:.6f} |"
        )
    lines.extend([
        "", f"Bootstrap: {bootstrap['n_users']:,} users, "
        f"{bootstrap['bootstrap_replicates']:,} replicates, "
        f"{bootstrap['confidence_interval']}.",
        "",
        "Confidence intervals are paired-user percentile intervals and are not "
        "adjusted for the full method/lambda selection process. "
        "Popularity-quartile subgroup results are descriptive.", "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {md_path.relative_to(ROOT)}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading saved experiment and network outputs...")
    bootstrap, tuned_results, segments, validation, edge_summary, scores = load_inputs()

    print(f"Bootstrap: {bootstrap['n_users']:,} users, "
          f"{bootstrap['bootstrap_replicates']:,} replicates")
    total_edge_rows = sum(v["count"] for v in edge_summary.values())
    print(f"Network: {len(scores):,} movies, {total_edge_rows:,} typed edge rows")
    print()

    generate_bootstrap_figure(bootstrap)
    generate_popularity_figure(segments)
    generate_main_metric_comparison(tuned_results)
    generate_network_construction_diagram()
    generate_strength_distribution(scores)
    generate_comparison_type_breakdown(edge_summary)
    generate_validation_lambda_curves(validation)
    generate_results_tables(bootstrap, tuned_results)

    print("\nAll presentation figures and tables generated successfully.")
    print("No experiments were run; source results were read only.")


if __name__ == "__main__":
    main()
