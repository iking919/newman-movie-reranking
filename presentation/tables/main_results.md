# Main Test Results

All reranking weights were selected using validation NDCG@10. Results below are test-set metrics.

| Method | λ | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BERT4Rec baseline | — | 0.190232 | 0.281291 | 0.126341 | 0.155945 |
| Newman | 0.1 | 0.190397 | 0.281788 | 0.125458 | 0.155101 |
| Mean rating | 0.05 | 0.190066 | 0.282450 | 0.125754 | 0.155729 |
| Shrunk mean rating | 0.05 | 0.188907 | 0.282119 | 0.124917 | 0.155184 |
| Popularity | 0.3 | 0.197185 | 0.287914 | 0.131411 | 0.160695 |
| Newman shrink (k=10) | 0.05 | 0.189735 | 0.282119 | 0.125542 | 0.155512 |
| Newman shrink (k=25) | 0.05 | 0.189570 | 0.282450 | 0.125385 | 0.155520 |
| Newman shrink (k=50) | 0.05 | 0.189570 | 0.282781 | 0.125417 | 0.155641 |
| Newman shrink (k=100) | 0.05 | 0.190232 | 0.282781 | 0.125563 | 0.155550 |

Bootstrap: 6,040 users, 10,000 replicates, 95% percentile paired-user bootstrap.

Confidence intervals are paired-user percentile intervals and are not adjusted for the full method/lambda selection process. Popularity-quartile subgroup results are descriptive.
