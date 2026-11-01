# DSCMR Assignment 3 — controlled loss ablations

Fixed author features; 720 train / 80 validation / 100 test. No full-train refit.
Checkpoint selection: mean bidirectional validation mAP. Test evaluated after all training.
R@K is query hit rate (at least one same-class candidate), NOT fraction of all relevant items.
Values below are percentages, mean ± sample standard deviation over 3 training seeds.

| Variant | Direction | mAP@ALL | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| B0_original | I2T | 66.77 ± 0.30 | 66.67 ± 0.58 | 88.67 ± 2.31 | 95.67 ± 0.58 |
| B0_original | T2I | 69.32 ± 0.88 | 73.67 ± 0.58 | 94.00 ± 1.73 | 97.33 ± 0.58 |
| B1_supcon | I2T | 66.76 ± 1.03 | 67.67 ± 1.53 | 85.33 ± 1.53 | 94.67 ± 0.58 |
| B1_supcon | T2I | 68.35 ± 1.40 | 72.33 ± 2.08 | 94.33 ± 1.15 | 97.33 ± 0.58 |
| B2_adaptive | I2T | 66.72 ± 0.92 | 67.33 ± 1.53 | 87.33 ± 2.89 | 95.33 ± 1.15 |
| B2_adaptive | T2I | 69.04 ± 1.88 | 74.67 ± 2.52 | 94.00 ± 1.00 | 98.00 ± 1.00 |
| B3_combined | I2T | 66.92 ± 0.86 | 66.67 ± 2.52 | 86.00 ± 1.73 | 95.67 ± 0.58 |
| B3_combined | T2I | 68.31 ± 0.61 | 73.33 ± 0.58 | 93.67 ± 1.53 | 97.67 ± 1.15 |

## Limits

- This first ablation uses preregistered coefficients (1, 0.001, 0.1), temperature 0.1; no hyperparameter search.
- Fixed-weight tuning is still required before claiming superiority over a tuned baseline.
- Shared fixed split, three initialization/batch seeds; this is not cross-validation.
- SupCon keeps image-image, text-text and symmetric cross-modal relations; excludes intra-modal self comparisons.
- Adaptive method is bounded EMA relative-loss-speed weighting, not GradNorm; novelty is not established.
- Author bundle has 100 test pairs. Results are not exact paper reproduction and not directly comparable with older refit runs.
- Historical test results have already been seen; this test is not a newly blinded holdout. No test-driven tuning in this run.
