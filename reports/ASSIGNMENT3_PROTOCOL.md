# Assignment 3: DSCMR loss ablations

## Scope and budget

This experiment implements the two selected modifications independently and together.
It does not change or overwrite the historical DSCMR/CLIP4CMR runs under `.`.

| Configuration | Common-space term | Weighting |
|---|---|---|
| B0_original | Author DSCMR term | Fixed |
| B1_supcon | Multi-positive supervised contrastive | Fixed |
| B2_adaptive | Author DSCMR term | Bounded EMA loss-speed |
| B3_combined | Multi-positive supervised contrastive | Bounded EMA loss-speed |

- Smoke: 4 runs x 10 epochs.
- Pilot: 4 runs x 200 epochs, seed 42, validation only.
- Main: 4 configurations x seeds 42, 43, 44; at most 500 epochs each.
- Main early stopping: 50 consecutive epochs without mean validation mAP improvement > 1e-8.
- Fresh initialization for every run, including pilot and main. No full-train refit.
- No unbudgeted hyperparameter search. All settings are fixed before test access.

## Data and fairness

Use the author's existing Pascal feature files: 800 original train pairs, image 4096D,
text 300D, 20 classes. A fixed stratified split (seed 42) yields 720 train and 80
validation pairs for every run. The author's test bundle contains 100 pairs.

Training seeds vary initialization and batch order, not the split. Each batch contains
up to five samples per class, each training row appears exactly once per epoch.
The last batch may be smaller. All four configurations use the same batch sequence
for a given seed. This balanced batching differs from the previous historical runner;
therefore B0 must be rerun and older refit results are not the direct baseline.

The architecture is loaded from the author's model class definitions. Adam uses
learning rate 1e-4 and betas (0.5, 0.999). No feature normalization or encoder fine-tuning
is added. Common-space SupCon normalizes the embeddings internally for similarity.

## Loss definitions

Label prediction and paired Euclidean terms remain unchanged. All configurations
start with base coefficients (1, 0.001, 0.1).

SupCon replaces ONLY the common-space term. Temperature is 0.1. It sums the
image-image and text-text losses and the mean of image-text/text-image losses.
Each anchor averages negative log probability over every same-class positive.
The softmax denominator includes positives and negatives. Intra-modal self-pairs
are excluded; anchors without an intra-modal positive are skipped. Corresponding
cross-modal pairs remain valid positives. Keeping intra-modal terms reduces a
confound relative to replacing three relation types by cross-modal relations alone.

Adaptive weights use a 10-epoch fixed-weight warmup. For each component, epoch-mean
training loss is smoothed with EMA decay 0.9. The next epoch's multiplier is
exp((EMA_ratio - mean_of_three_ratios) / 0.5), clipped to [0.5, 2]. Ratios are
clipped to [0.1, 10]. Effective coefficients equal base coefficients times these
multipliers. Weights have no gradient and do not use validation/test labels.
This is an explicit bounded loss-speed heuristic, NOT GradNorm. Its novelty and
benefit are not established; the unmodified method and separate ablations are essential.

SupCon changes the scale of the common-space term. These fixed-coefficient ablations
do not establish superiority over a tuned baseline. Equal-budget validation tuning
of fixed weights and temperature is a future experiment, not silently included here.

## Selection and evaluation

Choose each checkpoint by the mean of I2T/T2I validation mAP@ALL. All main training
finishes before the runner loads test arrays. Each selected checkpoint is then
evaluated once and its test result cached. The historical test has been seen before;
it is not a newly blinded holdout, although this run performs no test-driven tuning.

Cosine ranking, stable gallery-order tie breaking, relevance defined by class equality.
Report mAP@ALL and R@1/5/10 in both directions. R@K here means query hit rate
(at least one relevant item in top K), not fraction of all relevant documents retrieved.
Mean and sample standard deviation are computed across three training seeds, not CV folds.

## Verification and execution

The runner checks original-loss values AND gradients against the upstream function,
SupCon against an independent loop implementation, edge cases and gradient flow,
adaptive bounds/warmup/state restoration, and metrics against the existing evaluator.

From the workspace, using the existing Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.venv310\Scripts\python.exe' -u scripts/dscmr/loss_ablation.py --source-root '.' --out reports/assignment3_20261029 --stage all
```

Outputs: `protocol.json`, `verification.json`, `timing.json`, `status.json`,
per-run `history.json`, `best.pt`, `training_result.json`, and main `result.json`.
At completion: `results.csv`, `aggregate.json`, `REPORT.md`.

To pause, create a file named `STOP` in the output directory. The runner saves a
checkpoint at the next epoch boundary. Remove that file before resuming the same
command. Unfinished runs resume from `last.pt` (also saved every 10 epochs), completed
runs are skipped. Protocol and code hashes must match. A crash can leave `run.lock`;
verify its PID is no longer running before removing that exact lock manually.
