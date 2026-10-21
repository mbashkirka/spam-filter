# M4 — Calibration and Cost-Sensitive Decision System

M3 optimizes ranking quality, but `LinearSVC.decision_function` returns an uncalibrated margin rather than a spam probability. M4 separates **ranking**, **probability estimation**, and **product policy**.

## Calibration

1. Keep the M3-selected model family frozen.
2. Produce 5-fold out-of-fold decision margins on the M1 train split.
3. Fit a two-parameter Platt sigmoid (logistic regression) on those OOF margins.
4. Fit the base M3 classifier on all M1 train examples.
5. Convert future margins into calibrated probabilities with the frozen sigmoid.

This avoids learning the calibration mapping from validation or test labels.

## Threshold policy

The default example policy assigns a false positive a cost of 10 and a false negative a cost of 1. For each candidate threshold on validation:

`cost = 10 * FP + 1 * FN`

The minimum-cost threshold is frozen into the M4 artifact. The ratio is configurable and represents a product assumption: a real deployment must set it from business/user impact rather than copy the default blindly.

## Outputs

- `artifacts/m4_calibrated_champion.joblib`
- `reports/m4_results.json`
- `reports/m4_report.md`
- `reports/m4_threshold_curve.csv`
