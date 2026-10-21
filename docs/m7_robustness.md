# M7 robustness and failure-analysis contract

M7 is diagnostic. It does not fit or tune a model on Enron1 or on perturbed examples.

## External slices

The M6 per-message prediction audit is joined back to Enron1 by text SHA-256. M7 reports M3 and M5 behavior for:

- SMS-like messages (<=160 characters);
- short emails (161-512);
- standard emails (513-2048);
- long emails (>2048);
- messages containing URL-like strings, digits, HTML-like markup, or a Subject header.

The report includes FPR/FNR and classification/ranking metrics when both classes are present.

## Error taxonomy

Every Enron1 message is assigned to one of four model-agreement groups:

- both correct;
- only M3 sparse correct;
- only M5 Transformer correct;
- both wrong.

The audit file keeps identifiers, slices, predictions and scores, but does not duplicate full message text.

## Robustness stress test

The frozen M1 test spam subset is perturbed using four deterministic surface attacks:

1. partial leetspeak substitutions;
2. punctuation inserted inside long words;
3. zero-width Unicode separators;
4. visually similar Cyrillic confusables.

These are synthetic evasion probes, not estimates of attacker frequency. M7 measures spam detection rate, clean-correct to attacked-miss rate, and score shift for M3 and M5. No perturbed sample is used for fitting.

## Sparse explainability

For the M3 LinearSVC champion, M7 exports the strongest positive and negative global character n-gram coefficients. These weights explain the linear decision surface but are not causal explanations.
