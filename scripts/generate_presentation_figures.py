"""
Generate presentation figures and tables for the Newman Movie Reranking project.

Inputs:
    results/bert4rec_reranking/paired_user_bootstrap.json
    results/bert4rec_reranking/popularity_segment_summary.csv

Outputs:
    presentation/figures/paired_bootstrap_comparison.png
    presentation/figures/popularity_quartile_analysis.png
    presentation/tables/main_results.csv
    presentation/tables/main_results.md

No experiments are run and no source results are modified.

Run with PYTHONPATH=. python scripts/generate_presentation_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP_PATH = (
    ROOT / "results/bert4rec_reranking/paired_user_bootstrap.json"
)
SEGMENT_PATH = (
    ROOT / "results/bert4rec_reranking/popularity_segment_summary.csv"
)

FIGURE_DIR = ROOT / "presentation/figures"
TABLE_DIR = ROOT / "presentation/tables"

METHOD_ORDER = [
    "newman",
    "mean_rating",
    "shrunk_mean_rating",
    "popularity",
    "newman_shrink_k10",
    "newman_shrink_k25",
    "newman_shrink_k50",
    "newman_shrink_k100",
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

QUARTILE_ORDER = [
    "Q1_least_popular",
    "Q2",
    "Q3",
    "Q4_most_popular",
]

QUARTILE_NAMES = {
    "Q1_least_popular": "Q1 — least popular",
    "Q2": "Q2",
    "Q3": "Q3",
    "Q4_most_popular": "Q4 — most popular",
}


def load_inputs():
    if not BOOTSTRAP_PATH.exists():
        raise FileNotFoundError(BOOTSTRAP_PATH)

    if not SEGMENT_PATH.exists():
        raise FileNotFoundError(SEGMENT_PATH)

    with BOOTSTRAP_PATH.open(encoding="utf-8") as f:
        bootstrap = json.load(f)

    segments = pd.read_csv(SEGMENT_PATH)

    required = {
        "grouping",
        "segment",
        "n_users",
        "delta_HR@10",
        "delta_NDCG@10",
    }
    missing = required.difference(segments.columns)

    if missing:
        raise ValueError(
            f"Popularity segment CSV is missing columns: {sorted(missing)}"
        )

    missing_methods = set(METHOD_ORDER).difference(
        bootstrap["comparisons"].keys()
    )
    if missing_methods:
        raise ValueError(
            f"Bootstrap JSON is missing methods: {sorted(missing_methods)}"
        )

    return bootstrap, segments


def generate_bootstrap_figure(bootstrap):
    """Plot paired-user bootstrap effects and 95% confidence intervals."""

    comparisons = bootstrap["comparisons"]

    metrics = [
        ("HR@10", "HR@10"),
        ("NDCG@10", "NDCG@10"),
    ]

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(15, 8),
        sharey=True,
    )

    y_positions = np.arange(len(METHOD_ORDER))

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        effects = []
        lower_errors = []
        upper_errors = []

        for method in METHOD_ORDER:
            metric = comparisons[method]["metrics"][metric_key]

            effect = metric["difference_reranked_minus_baseline"]
            ci_low, ci_high = metric["ci_95_percentile"]

            # Convert to percentage points for presentation.
            effects.append(effect * 100)
            lower_errors.append((effect - ci_low) * 100)
            upper_errors.append((ci_high - effect) * 100)

        ax.errorbar(
            effects,
            y_positions,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(metric_title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Change from BERT4Rec baseline (percentage points)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [
                f"{METHOD_NAMES[m]} "
                f"(λ={comparisons[m]['validation_selected_lambda']:g})"
                for m in METHOD_ORDER
            ]
        )
        ax.invert_yaxis()

    fig.suptitle(
        "Reranking Performance: Paired-User Bootstrap",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.015,
        (
            f"95% percentile paired-user bootstrap CIs; "
            f"{bootstrap['n_users']:,} users; "
            f"{bootstrap['bootstrap_replicates']:,} replicates. "
            "Weights selected on validation NDCG@10. "
            "Intervals are not adjusted for method/lambda selection."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.93])

    output = FIGURE_DIR / "paired_bootstrap_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {output.relative_to(ROOT)}")


def generate_popularity_figure(segments):
    """Plot descriptive changes across target-popularity quartiles."""

    quartiles = segments[
        segments["grouping"] == "target_popularity_quartile"
    ].copy()

    quartiles["segment"] = pd.Categorical(
        quartiles["segment"],
        categories=QUARTILE_ORDER,
        ordered=True,
    )
    quartiles = quartiles.sort_values("segment")

    if len(quartiles) != 4:
        raise ValueError(
            "Expected exactly four target-popularity quartiles; "
            f"found {len(quartiles)}."
        )

    labels = [
        QUARTILE_NAMES[str(segment)]
        for segment in quartiles["segment"]
    ]

    metrics = [
        ("delta_HR@10", "Change in HR@10"),
        ("delta_NDCG@10", "Change in NDCG@10"),
    ]

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13, 6),
    )

    y_positions = np.arange(len(quartiles))
    hatches = ["///", "\\\\\\", "...", "xxx"]

    for ax, (column, title) in zip(axes, metrics):
        values = quartiles[column].to_numpy(dtype=float) * 100

        bars = ax.barh(
            y_positions,
            values,
            height=0.62,
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Change from BERT4Rec baseline (percentage points)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        max_abs = max(abs(values.min()), abs(values.max()))
        margin = max_abs * 0.32 if max_abs else 0.01
        ax.set_xlim(values.min() - margin, values.max() + margin)

        for bar, value in zip(bars, values):
            offset = max_abs * 0.035 if max_abs else 0.001
            if value >= 0:
                x = value + offset
                alignment = "left"
            else:
                x = value - offset
                alignment = "right"

            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha=alignment,
                fontsize=10,
            )

    fig.suptitle(
        "Popularity Reranking by Held-Out Target Popularity",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Descriptive subgroup analysis; 1,510 users per quartile. "
            "Quartiles are based on training interaction counts "
            "of each user's held-out target item. "
            "No subgroup confidence intervals are shown."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    output = FIGURE_DIR / "popularity_quartile_analysis.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {output.relative_to(ROOT)}")


def generate_results_tables(bootstrap):
    """Generate CSV and Markdown versions of the main test results."""

    comparisons = bootstrap["comparisons"]
    rows = []

    for method in METHOD_ORDER:
        result = comparisons[method]
        hr = result["metrics"]["HR@10"]
        ndcg = result["metrics"]["NDCG@10"]

        rows.append({
            "method": METHOD_NAMES[method],
            "lambda": result["validation_selected_lambda"],
            "baseline_HR@10": hr["baseline_mean"],
            "reranked_HR@10": hr["reranked_mean"],
            "delta_HR@10": hr["difference_reranked_minus_baseline"],
            "baseline_NDCG@10": ndcg["baseline_mean"],
            "reranked_NDCG@10": ndcg["reranked_mean"],
            "delta_NDCG@10": ndcg[
                "difference_reranked_minus_baseline"
            ],
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

    # Compact presentation table.
    md_path = TABLE_DIR / "main_results.md"

    lines = [
        "# Main Test Results",
        "",
        (
            "All reranking weights were selected using validation "
            "NDCG@10. Results below are test-set metrics."
        ),
        "",
        "| Method | λ | HR@10 | Δ HR@10 | NDCG@10 | Δ NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    baseline_hr = rows[0]["baseline_HR@10"]
    baseline_ndcg = rows[0]["baseline_NDCG@10"]

    lines.append(
        f"| BERT4Rec baseline | — | "
        f"{baseline_hr:.6f} | — | "
        f"{baseline_ndcg:.6f} | — |"
    )

    for row in rows:
        lines.append(
            f"| {row['method']} | {row['lambda']:g} | "
            f"{row['reranked_HR@10']:.6f} | "
            f"{row['delta_HR@10']:+.6f} | "
            f"{row['reranked_NDCG@10']:.6f} | "
            f"{row['delta_NDCG@10']:+.6f} |"
        )

    lines.extend([
        "",
        (
            f"Bootstrap: {bootstrap['n_users']:,} users, "
            f"{bootstrap['bootstrap_replicates']:,} replicates, "
            f"{bootstrap['confidence_interval']}."
        ),
        "",
        (
            "Confidence intervals are paired-user percentile intervals "
            "and are not adjusted for the full method/lambda selection "
            "process. Popularity-quartile subgroup results are descriptive."
        ),
        "",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {md_path.relative_to(ROOT)}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing experiment outputs...")
    bootstrap, segments = load_inputs()

    print(f"Bootstrap source: {BOOTSTRAP_PATH.relative_to(ROOT)}")
    print(f"Segment source:   {SEGMENT_PATH.relative_to(ROOT)}")
    print(
        f"Bootstrap metadata: {bootstrap['n_users']} users, "
        f"{bootstrap['bootstrap_replicates']} replicates"
    )
    print()

    generate_bootstrap_figure(bootstrap)
    generate_popularity_figure(segments)
    generate_results_tables(bootstrap)

    print("\nAll presentation artifacts generated successfully.")


if __name__ == "__main__":
    main()
