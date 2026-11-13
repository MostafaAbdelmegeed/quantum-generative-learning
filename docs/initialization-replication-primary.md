# Replication: prespecified comparisons

The primary comparison is near-identity minus uniform TVD under order-penalized weighting with one 1,600-update run. The frozen descriptive criterion requires a mean difference below -0.01 and improvements on at least eight of ten seeds. Negative differences favor near-identity starts.

| Condition | Uniform mean | Near-identity mean | Mean difference | Median difference | Difference range | Improved / tied / worse | Criterion met |
|---|---:|---:|---:|---:|---|---|---|
| Adjacent pairs | 0.3246 | 0.0372 | -0.2874 | -0.4488 | -0.6002 to +0.0248 | 7 / 0 / 3 | No |
| Shuffled pairs | 0.2658 | 0.0372 | -0.2286 | -0.2363 | -0.4682 to +0.0193 | 6 / 0 / 4 | No |
| Noisy pairs | 0.1746 | 0.0828 | -0.0917 | -0.0225 | -0.2739 to +0.0065 | 8 / 0 / 2 | Yes |
| Three-bit parity | 0.4343 | 0.0477 | -0.3866 | -0.4566 | -0.5350 to +0.0013 | 9 / 0 / 1 | Yes |
| Four-bit parity | 0.0655 | 0.0706 | +0.0051 | +0.0017 | -0.0169 to +0.0520 | 5 / 0 / 5 | No |

## Comparison with the selected classical model

Both methods use the same order-penalized selected features and training data. Capacities and fitting costs are not matched. The two parity comparisons were declared separately in the protocol; all conditions are retained below.

| Condition | Circuit mean | Classical mean | Mean circuit minus classical | Paired range | Circuit lower TVD |
|---|---:|---:|---:|---|---:|
| Adjacent pairs | 0.0372 | 0.0369 | +0.0003 | -0.0003 to +0.0016 | 4/10 |
| Shuffled pairs | 0.0372 | 0.0369 | +0.0003 | -0.0003 to +0.0016 | 4/10 |
| Noisy pairs | 0.0828 | 0.0507 | +0.0322 | +0.0112 to +0.0637 | 0/10 |
| Three-bit parity | 0.0477 | 0.0515 | -0.0038 | -0.0183 to +0.0054 | 4/10 |
| Four-bit parity | 0.0706 | 0.0647 | +0.0059 | -0.0179 to +0.0448 | 6/10 |

## Every primary paired difference

| Seed | Adjacent pairs | Shuffled pairs | Noisy pairs | Three-bit parity | Four-bit parity |
|---|---:|---:|---:|---:|---:|
| 1001 | +0.0158 | +0.0096 | +0.0044 | -0.4455 | -0.0150 |
| 1002 | +0.0000 | +0.0146 | +0.0065 | -0.4673 | +0.0226 |
| 1003 | -0.4582 | -0.4607 | -0.2739 | -0.5179 | +0.0036 |
| 1004 | -0.4646 | -0.4682 | -0.0084 | +0.0013 | +0.0169 |
| 1005 | -0.0239 | -0.0241 | -0.0025 | -0.0086 | -0.0169 |
| 1006 | -0.4613 | -0.4609 | -0.1799 | -0.4755 | -0.0002 |
| 1007 | +0.0248 | +0.0193 | -0.0366 | -0.5294 | -0.0088 |
| 1008 | -0.4674 | -0.4674 | -0.2690 | -0.4420 | -0.0102 |
| 1009 | -0.6002 | +0.0002 | -0.0044 | -0.4459 | +0.0520 |
| 1010 | -0.4395 | -0.4485 | -0.1536 | -0.5350 | +0.0073 |

The [complete policy tables](initialization-replication-tables.md) and [summary](initialization-replication-summary.json) retain every candidate and all secondary comparisons. The old three-seed discovery results are not pooled into these estimates.
