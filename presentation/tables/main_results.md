# Main Test Results

All reranking weights were selected using validation NDCG@10. Results below are test-set metrics.

| Method | λ | HR@10 | Δ HR@10 | NDCG@10 | Δ NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BERT4Rec baseline | — | 0.281291 | — | 0.155945 | — |
| Newman | 0.1 | 0.281788 | +0.000497 | 0.155101 | -0.000845 |
| Mean rating | 0.05 | 0.282450 | +0.001159 | 0.155729 | -0.000217 |
| Shrunk mean rating | 0.05 | 0.282119 | +0.000828 | 0.155184 | -0.000761 |
| Popularity | 0.3 | 0.287914 | +0.006623 | 0.160695 | +0.004750 |
| Newman shrink (k=10) | 0.05 | 0.282119 | +0.000828 | 0.155512 | -0.000434 |
| Newman shrink (k=25) | 0.05 | 0.282450 | +0.001159 | 0.155520 | -0.000426 |
| Newman shrink (k=50) | 0.05 | 0.282781 | +0.001490 | 0.155641 | -0.000305 |
| Newman shrink (k=100) | 0.05 | 0.282781 | +0.001490 | 0.155550 | -0.000395 |

Bootstrap: 6,040 users, 10,000 replicates, 95% percentile paired-user bootstrap.

Confidence intervals are paired-user percentile intervals and are not adjusted for the full method/lambda selection process. Popularity-quartile subgroup results are descriptive.
