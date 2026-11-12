## Baselines (frozen features, test 200)

| Features | Method | I2T | T2I | Mean | Acc img | Acc txt |
|---|---|---:|---:|---:|---:|---:|
| RN50 | zero-shot (concat) | 55.27 | 55.02 | 55.15 | nan | nan |
| RN50 | zero-shot (caption mean) | 59.40 | 57.68 | 58.54 | nan | nan |
| RN50 | classify-then-rank (zero-shot clf) | 71.54 | 72.27 | 71.91 | 78.0 | 78.0 |
| RN50 | classify-then-rank (linear probe) | 69.96 | 69.96 | 69.96 | 76.0 | 83.0 |
| DFN5B | zero-shot (concat) | 58.80 | 58.78 | 58.79 | nan | nan |
| DFN5B | zero-shot (caption mean) | 61.34 | 60.77 | 61.06 | nan | nan |
| DFN5B | classify-then-rank (zero-shot clf) | 74.30 | 75.83 | 75.07 | 83.5 | 78.5 |
| DFN5B | classify-then-rank (linear probe) | 73.57 | 76.67 | 75.12 | 82.5 | 82.5 |

## Models: mean mAP over seeds, 95% bootstrap interval over test items

| Model | mAP ± SD (seeds) | 95% CI | Acc img / txt |
|---|---:|---|---|
| C4 base RN50 | 68.17 ± 0.51 | [63.94, 72.32] |  |
| C4 A+B RN50 | 70.76 ± 0.26 | [66.91, 74.47] |  |
| C4 A+B DFN5B frozen | 73.62 ± 0.18 | [69.87, 77.21] | 82.3 / 81.4 |
| C4 base DFN5B FT2 | 75.34 ± 0.31 | [71.37, 79.17] |  |
| C4 A+B DFN5B FT2 | 76.15 ± 0.46 | [72.46, 79.63] | 85.3 / 79.7 |
| DSCMR RN50 | 69.38 ± 0.39 | [65.24, 73.40] |  |
| DSCMR ens3 | 74.06 ± 0.37 | [70.16, 77.80] |  |
| DSCMR DFN5B FT4 | 73.74 ± 0.39 | [69.88, 77.43] |  |
| CtR zero-shot RN50 | 71.91 ± 0.00 | [67.39, 76.09] |  |
| CtR probe RN50 | 69.96 ± 0.00 | [65.83, 73.95] |  |
| CtR zero-shot DFN5B | 75.07 ± 0.00 | [70.69, 79.16] |  |
| CtR probe DFN5B | 75.12 ± 0.00 | [71.06, 78.99] |  |

## Paired differences (points)

| Comparison | Diff | 95% CI | P(boot <= 0) |
|---|---:|---|---:|
| C4 A+B RN50 - C4 base RN50 | +2.58 | [+1.53, +3.67] | 0.0000 |
| C4 A+B DFN5B frozen - C4 base RN50 | +5.44 | [+3.89, +7.10] | 0.0000 |
| C4 A+B DFN5B FT2 - C4 A+B DFN5B frozen | +2.54 | [+1.75, +3.33] | 0.0000 |
| C4 A+B DFN5B FT2 - C4 base DFN5B FT2 | +0.82 | [-0.26, +1.88] | 0.0692 |
| C4 A+B DFN5B FT2 - C4 base RN50 | +7.98 | [+6.32, +9.72] | 0.0000 |
| C4 base DFN5B FT2 - C4 base RN50 | +7.16 | [+5.60, +8.85] | 0.0000 |
| DSCMR DFN5B FT4 - DSCMR RN50 | +4.37 | [+3.01, +5.83] | 0.0000 |
| DSCMR DFN5B FT4 - DSCMR ens3 | -0.32 | [-1.11, +0.51] | 0.7868 |
| C4 A+B DFN5B FT2 - DSCMR DFN5B FT4 | +2.41 | [+1.64, +3.20] | 0.0000 |
| C4 base RN50 - CtR zero-shot RN50 | -3.73 | [-5.82, -1.63] | 0.9997 |
| C4 A+B RN50 - CtR zero-shot RN50 | -1.15 | [-2.86, +0.58] | 0.9061 |
| C4 A+B DFN5B frozen - CtR zero-shot DFN5B | -1.45 | [-2.97, +0.07] | 0.9695 |
| C4 A+B DFN5B FT2 - CtR zero-shot DFN5B | +1.09 | [-0.30, +2.53] | 0.0661 |
| C4 A+B DFN5B FT2 - CtR probe DFN5B | +1.03 | [-0.21, +2.30] | 0.0534 |
| DSCMR DFN5B FT4 - CtR zero-shot DFN5B | -1.32 | [-2.79, +0.15] | 0.9621 |
