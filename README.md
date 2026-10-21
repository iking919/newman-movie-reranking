# Newman-Inspired Movie Ranking for Sequential Recommendation

A research project for **CSCI 6250: Frontiers of Network Science** at Rensselaer Polytechnic Institute.

This project investigates whether a global movie ranking inferred from heterogeneous pairwise comparisons can improve personalized sequential recommendations produced by BERT4Rec. It builds on M. E. J. Newman's *Ranking with Multiple Types of Pairwise Comparisons* and investigates an extension of comparison-based network ranking to sequential movie recommendation.

**Project status:** BERT4Rec baseline established in the existing research project; course-project baseline reproduction and network-ranking experiments in progress.

## Research Question

Can a global movie ranking inferred from multiple types of pairwise comparisons improve the top-10 recommendation performance of BERT4Rec on MovieLens-1M?

The primary hypothesis is that incorporating information from a global movie comparison network will improve NDCG@10 compared with BERT4Rec alone. HR@10 will be evaluated as a secondary metric.

The project will also investigate whether any improvement is attributable to the pairwise-ranking method or can be explained by a simpler popularity-based reranking baseline.

## Background

Sequential recommendation models learn from a user's interaction history to predict items they may interact with next. BERT4Rec uses a bidirectional Transformer and masked item prediction to learn representations of user interaction sequences.

Newman's ranking framework addresses the problem of inferring relative rankings from pairwise comparisons that may belong to different interaction types. Instead of treating every comparison as equally informative, the framework accounts for differences between comparison types.

This project investigates whether a global movie ranking inferred from pairwise comparisons can provide information complementary to the personalized recommendations produced by BERT4Rec.

## Research Objectives

1. Establish a reproducible BERT4Rec baseline using MovieLens-1M.
2. Construct a movie comparison network using training-set ratings.
3. Adapt Newman's multiple-comparison ranking approach to infer global movie scores.
4. Integrate the global ranking with BERT4Rec recommendation scores.
5. Compare the combined model with BERT4Rec alone and a popularity-based reranking baseline.
6. Evaluate recommendation quality and computational overhead.
7. Analyze the conditions under which global ranking information helps or hurts personalized recommendation.

## Dataset

The primary dataset is **MovieLens-1M**, published by the GroupLens Research Project at the University of Minnesota.

The original dataset contains:

| Statistic | Count |
|---|---:|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,706 |

The dataset includes user IDs, movie IDs, explicit ratings, and timestamps.

The existing recommender pipeline filters movies with fewer than five interactions and constructs chronological user interaction sequences.

The currently processed dataset contains:

| Statistic | Value |
|---|---:|
| Users | 6,040 |
| Mapped movies | 3,416 |
| Vocabulary entries, including special tokens | 3,418 |
| Maximum sequence length | 200 |
| Training interactions | 648,759 |

These statistics describe the existing processed data and will be verified against the exact artifacts used in the course-project repository.

Special token IDs in the existing preprocessing pipeline are:

| Token | ID |
|---|---:|
| Padding (`PAD`) | 0 |
| Mask (`MASK`) | 1 |
| Movie items | 2 and above |

### Data preparation

The existing preprocessing pipeline performs the following operations:

1. Loads the original MovieLens-1M ratings.
2. Sorts user interactions chronologically.
3. Filters movies with fewer than five interactions.
4. Constructs user interaction sequences.
5. Maps movie IDs to contiguous model item IDs.
6. Limits sequence length to 200.
7. Creates chronological training, validation, and test splits.
8. Generates model-specific training examples.

The course project will reuse the established preprocessing and split definitions wherever possible. The network-ranking method will use training-set ratings only to prevent information from validation or test interactions leaking into the global ranking.

Original dataset files and derived datasets will not be committed to the repository.

## Repository Structure

```text
newman-movie-reranking/
├── configs/
│   ├── bert4rec_baseline.yaml
│   └── bert4rec_tuned.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── checkpoints/
│   ├── bert4rec_baseline/
│   └── bert4rec_tuned/
├── results/
│   ├── bert4rec_baseline/
│   ├── bert4rec_tuned/
│   ├── bert4rec_grid_search/
│   ├── network_ranking/
│   └── score_fusion/
├── scripts/
│   ├── run_bert4rec_evaluation.py
│   ├── run_bert4rec_hyperparameter_search.py
│   ├── build_movie_network.py
│   ├── run_network_ranking.py
│   └── run_score_fusion.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── network/
│   ├── ranking/
│   ├── recommendation/
│   ├── evaluation/
│   └── utils/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure describes the intended organization. Existing BERT4Rec modules and script names should be preserved where practical, and any paths should be adjusted to match the actual implementation.

## Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.2 or later
- CUDA-compatible GPU recommended for training

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The appropriate PyTorch build depends on the operating system, GPU, and CUDA support. Follow the official PyTorch installation instructions when installing a CUDA-enabled version.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data Preparation

The expected raw dataset location is:

```text
data/raw/ml-1m/
├── README
├── movies.dat
├── ratings.dat
└── users.dat
```

If the existing download and preprocessing scripts are included in this repository, run:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

These script names are inherited from the existing project and should be verified after copying the relevant files.

Refer to `data/README.md` and the actual preprocessing implementation for the authoritative data preparation procedure.

The original dataset files should not be modified.

## BERT4Rec Baseline

BERT4Rec is the primary sequential recommendation baseline for this project. It uses a bidirectional Transformer and masked item prediction to learn from user interaction histories.

The existing implementation provides baseline and tuned configurations:

- `configs/bert4rec_baseline.yaml`
- `configs/bert4rec_tuned.yaml`

The tuned model is the primary baseline for evaluating the proposed network-ranking extension. The baseline configuration is retained for reference.

### Training the baseline

Run from the repository root:

```bash
python -m src.training.train_bert4rec \
    --config configs/bert4rec_baseline.yaml
```

### Training the tuned model

```bash
python -m src.training.train_bert4rec \
    --config configs/bert4rec_tuned.yaml
```

Training checkpoints and experiment outputs are written to the locations defined in the corresponding configuration files.

Verify those output paths before running the experiments in the new repository.

## BERT4Rec Hyperparameter Search

Hyperparameter search is used to identify promising BERT4Rec optimization settings. The search should select configurations using validation performance rather than the held-out test set.

The existing project provides the following search entry point:

```bash
python -m scripts.run_bert4rec_hyperparameter_search \
    --config configs/bert4rec_tuned.yaml \
    --output-dir results/bert4rec_grid_search \
    --allow-existing
```

The exact parameter grid is defined in the search script.

The `--allow-existing` flag permits reuse of an existing output directory if supported by the script. Review the script's resume and overwrite behavior before using it.

### Search outputs

The intended output structure is:

```text
results/bert4rec_grid_search/
├── search_summary.csv
├── grid_search_report.json
├── trial_1/
│   ├── config.yaml
│   ├── training_history.csv
│   └── run_summary.json
└── ...
```

The actual filenames and trial directory names depend on the implementation.

- `search_summary.csv` contains trial-level results.
- `grid_search_report.json` contains aggregate search metadata.
- Individual trial directories preserve configurations, training histories, and summaries.

Model checkpoints should not be saved for every trial unless explicitly required. Confirm that checkpoint saving is disabled or configured appropriately in the search implementation.

### Selecting the tuned configuration

The preferred selection criterion is validation NDCG@10.

After the search:

1. Identify the best valid trial using validation NDCG@10.
2. Record the selected configuration and its search results.
3. Keep the baseline configuration unchanged.
4. Preserve the selected tuned configuration and checkpoint.
5. Use the held-out test set only for final evaluation.

## BERT4Rec Evaluation

Evaluation uses a trained checkpoint and its corresponding configuration.

### Baseline evaluation

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_baseline.yaml \
    --checkpoint checkpoints/bert4rec_baseline/bert4rec_baseline_best.pt
```

### Tuned evaluation

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_tuned.yaml \
    --checkpoint checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
```

These commands are taken from the existing project's documented interface. Confirm that the corresponding modules, checkpoint filenames, and command-line arguments match the files copied into this repository.

For the course project, preserve the exact baseline checkpoint, configuration, preprocessing definitions, and evaluation protocol. This will allow the graph-based extension to be compared against the same underlying recommender.

## Existing BERT4Rec Results

The following results were obtained in the existing JEPA and BERT4Rec research project. They are preliminary reference results for establishing the course-project baseline and must be reproduced or verified under the final course-project evaluation protocol.

| Model | Configuration | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| BERT4Rec | Baseline | 0.1460 | 0.2207 | 0.0962 | 0.1202 |
| BERT4Rec | Tuned | 0.1902 | 0.2813| 0.1263 | 0.1559 |

The tuned BERT4Rec configuration is the primary comparison point for the network-ranking experiments.

### Interpretation

The existing tuned BERT4Rec configuration improves on the baseline across all four reported metrics.

The improvement in NDCG@10 is particularly relevant because the proposed extension is intended to improve the ordering of relevant movies in the recommendation list.

These values should not be treated as the final course-project results until the checkpoint, split definitions, candidate-item handling, and evaluation procedure have been verified in the new repository.

## Evaluation Protocol

The primary metrics are:

- **HR@10:** Whether the held-out target movie appears among the ten highest-ranked recommendations.
- **NDCG@10:** Ranking quality, with greater credit assigned to relevant movies appearing nearer the top of the recommendation list.

HR@5 and NDCG@5 will be reported as supplementary metrics.

For fair comparisons, the baseline and all extensions must use:

1. The same chronological training, validation, and test split definitions.
2. The same candidate-item set.
3. The same policy for excluding previously interacted-with movies.
4. The same definitions of HR and NDCG.
5. The same held-out target interactions.

Validation NDCG@10 will be the primary model-selection metric. Final test metrics will be reported only after model and fusion-weight selection.

## Experimental Design

### 1. BERT4Rec baseline

The tuned BERT4Rec model provides personalized sequential recommendation scores based on user interaction histories.

This experiment establishes the reference performance against which the proposed extension will be evaluated.

### 2. Movie comparison network

Movies will be represented as nodes. Pairwise comparisons will be derived from ratings assigned by the same user to different movies.

When a user rates one movie higher than another, the higher-rated movie will be treated as the winner of the comparison. Equal-rated pairs will initially be excluded.

The proposed comparison types are:

- **Close comparisons:** Rating differences of one star.
- **Large-difference comparisons:** Rating differences of two or more stars.

These categories are an initial operationalization for MovieLens-1M and will be justified and documented in the final methodology.

Only training-set ratings will be used to construct the comparison network. Validation and test interactions will not contribute to the inferred ranking.

The implementation will follow the relevant model assumptions and inference procedure in Newman's paper as closely as appropriate. Adaptations needed for movie ratings will be documented explicitly.

### 3. Global movie ranking

The network-ranking method will estimate a global score for each movie based on the comparison data.

The implementation will record:

- Comparison-type definitions.
- Pairwise-comparison construction.
- Ranking algorithm and inference settings.
- Treatment of movies with limited comparison data.
- Ranking score normalization.
- Network construction and inference runtime.

The ranking will be examined independently before it is integrated with BERT4Rec.

### 4. Score fusion

The global movie ranking will be combined with the personalized BERT4Rec scores:

\[
s_{\mathrm{combined}}(u,i)
=
s_{\mathrm{BERT4Rec}}(u,i)
+
\lambda g(i)
\]

where:

- \(u\) denotes a user.
- \(i\) denotes a candidate movie.
- \(s_{\mathrm{BERT4Rec}}(u,i)\) is the BERT4Rec score.
- \(g(i)\) is the normalized global movie-ranking score.
- \(\lambda\) controls the contribution of the global ranking.

The normalization procedure and fusion weights will be documented. The fusion weight will be selected using validation NDCG@10, with the test set reserved for final evaluation.

## Planned Experiments

| Experiment | Description | Status |
|---|---|---|
| A | Tuned BERT4Rec baseline | Existing results available; reproduction required |
| B | Global comparison-network ranking alone | Planned |
| C | BERT4Rec combined with the global ranking | Planned |
| D | BERT4Rec with different fusion weights | Planned |
| E | BERT4Rec combined with popularity-based ranking | Planned |

The popularity baseline is important for determining whether any improvement is attributable to the heterogeneous comparison-ranking method or can be explained by a simpler global ranking signal.

## Network-Ranking Results

This section will be updated after the comparison network and ranking implementation have been completed.

### Network construction

| Measurement | Result |
|---|---|
| Number of movie nodes | Pending |
| Number of pairwise comparisons | Pending |
| Number of comparison types | 2 initially proposed |
| Number of distinct comparison edges | Pending |
| Network construction time | Pending |
| Ranking computation time | Pending |

The definition of an edge, treatment of repeated comparisons, and whether the network is represented as directed or weighted will be documented with the implementation.

### Global ranking evaluation

| Method | HR@10 | NDCG@10 | Runtime |
|---|---:|---:|---:|
| Global comparison-network ranking | Pending | Pending | Pending |
| Popularity ranking | Pending | Pending | Pending |

Standalone ranking performance will be measured using the same held-out target interactions and candidate-item protocol used for the other experiments.

## Score-Fusion Results

This section will report whether the global movie ranking improves the personalized BERT4Rec baseline.

| Model | Fusion weight | HR@10 | NDCG@10 | Additional runtime |
|---|---:|---:|---:|---:|
| BERT4Rec baseline | 0.0 | 0.2207* | 0.1202* | 587.83s |
| BERT4Rec tuned  | 0.0 | 0.2813 | 0.1559 | 457.53s |
| BERT4Rec + network ranking | Pending | Pending | Pending | Pending |
| BERT4Rec + popularity ranking | Pending | Pending | Pending | Pending |

\* Existing-project results; to be verified under the course-project evaluation protocol.

Additional fusion weights will be reported as separate rows. The selected weight will be determined using validation data, not test results.

## Determining Success

The primary success criterion is an improvement in validation NDCG@10 over the BERT4Rec baseline, followed by evaluation on the held-out test set.

HR@10 provides complementary evidence about whether the combined model retrieves the target movie more frequently.

Results will be interpreted alongside the popularity baseline and computational overhead.

Possible outcomes include:

- **Improvement:** The global ranking provides complementary information that improves sequential recommendation.
- **No meaningful change:** The global ranking contributes little additional information beyond BERT4Rec.
- **Degradation:** The global ranking interferes with personalized recommendations.
- **Popularity-equivalent improvement:** The observed gain may be attributable to a simpler popularity signal rather than the heterogeneous comparison method.

All outcomes will be reported, including negative results.

## Computational Analysis

Computational measurements will include:

- Network construction time.
- Ranking inference time.
- BERT4Rec evaluation time.
- Additional score-fusion overhead.
- Memory usage where available.

The objective is to determine whether any improvement in ranking quality justifies the additional computational cost.

## Reproducibility

The project will preserve:

1. BERT4Rec configurations and checkpoint identifiers.
2. Preprocessing and chronological split definitions.
3. Comparison-type definitions and network construction parameters.
4. The ranking implementation and inference settings.
5. Score-normalization procedures and fusion weights.
6. Validation-based model-selection criteria.
7. Individual experiment results and aggregate evaluation tables.
8. Software versions, hardware details, and random seeds.
9. Final held-out test results.

Hyperparameter and fusion-weight selection will be based on validation performance. The test set will not be used to choose configurations.

## Relationship to Existing Research

This repository is a focused extension of the broader JEPA and BERT4Rec sequential recommendation research conducted for the author's master's research.

The existing research investigates JEPA-based sequential recommendation, compares it with BERT4Rec, and examines architecture, optimization, and training efficiency.

This course project instead focuses on network science: whether heterogeneous pairwise-comparison ranking can complement an established personalized recommender.

The BERT4Rec implementation and preprocessing pipeline will be reused where appropriate to avoid unnecessary reimplementation and to preserve a consistent baseline.

## References

Newman, M. E. J. *Ranking with Multiple Types of Pairwise Comparisons*. Cite the publication or preprint version used in the course.

Harper, F. Maxwell, and Joseph A. Konstan. 2015. “The MovieLens Datasets: History and Context.” *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872.

MovieLens-1M is provided by the GroupLens Research Project at the University of Minnesota. Follow the dataset's applicable usage terms.

## Author

Izaak King  
CSCI 6250 — Frontiers of Network Science  
Rensselaer Polytechnic Institute  
Fall 2026
