# Observable selection and optimizer diagnostics

Frozen on 11 November 2026 after observing `observable-generalization-v1` and before running this series. This is a follow-up to those failures, not an independent confirmation study.

## Questions

The first comparison confounded the preference for composite observables with optimization difficulty. It also left mask-sampling noise in its enumerated-moment control. This factorial comparison asks whether a preference for lower-order observables, four times as many Adam updates, or a fully deterministic gradient changes the results.

Keep all five targets, three seeds, 512 training examples, 1,024 held-out examples, 21-parameter dense IQP circuit and known expressibility witnesses from the [previous protocol](observable-generalization-protocol.md). Keep initial angles, evaluation samples, data streams, ordinary-mask streams and priority-index streams unchanged. The existing exact-support and Qiskit tests remain prerequisites.

## Frozen changes

Add one selector to ordinary MMD, pair weighting and the previous higher-order weighting. The new selector enumerates the same 50 degree-two-through-four masks, computes the same absolute departure from independent marginals, and divides that score by `degree - 1`. Sort by descending penalized score, then increasing degree and lexicographic support. Select three masks using the same GF(2) independence filter. Degree penalties and the selected count are fixed priors. No test values or target labels enter selection. Record raw scores, penalized scores, masks, counts and runtime. Fit an additional nine-parameter classical maximum-entropy model using these selected features and the six singleton features.

Run 400 and 1,600 updates from the same initial parameters. Each is a fresh fit whose cosine schedule runs from 0.03 to 0.003 over its entire budget. The long run is not a continuation of the short run; the comparison changes optimization duration and the decay schedule together. Adam beta values and numerical epsilon remain unchanged. Do not pick a best seed or checkpoint.

For every loss and budget use three gradient estimators:

1. Monte Carlo moments with 64 sampled masks and two independent halves of 64 uniform strings.
2. Enumerated moments with the same 64 sampled masks and all 64 uniform strings, as before.
3. Fully enumerated moments and masks: sum all 64 mask contributions with their exact Fourier weights, including the identity. For weighted losses, put half the weight on the original measure and half uniformly on the selected masks. This removes both sampling sources. Use the same Adam schedule and initial parameters.

The Monte Carlo cost is 8,192 phase evaluations per update; enumerated and full controls each use 4,096. The full control counts 64 enumerated masks per step and no sampled masks or random strings. Budgets match between loss variants within an estimator; cross-estimator costs are reported separately. Full-kernel initial and final diagnostics are additional evaluations.

This makes 360 circuit fits: five conditions, three seeds, four losses, three estimators and two budgets. Repeat the original 90 short-budget fits and verify that parameters and histories reproduce the published records. Reuse the six learned classical references and oracle from those records, with their file hashes; fit 15 new order-penalized classical controls. The old three classical iteration-limit failures remain visible.

## Outcomes and interpretation

Primary outcome is exact TVD. Report all paired changes for selector, budget and estimator, without selecting favorable combinations. Retain Jensen-Shannon divergence, original held-out MMD, true block moments, marginal error, noisy-target all-even probability error, final full-objective gradients and complete optimizer histories. Draw 2,048 samples per circuit and new classical fit. Target knowledge remains restricted to expressibility checks and evaluation.

Neither a longer budget nor an exact gradient proves convergence. Compare new losses to their own classical controls as well as ordinary MMD. The order penalty is motivated by observed failures, so improvements remain exploratory. Three initializations measure sensitivity without establishing a success probability. The previous archived records stay unchanged.

## Execution

Use one worker and one BLAS thread after a preflight on arbitrary uniform data. Freeze source, configuration, protocol, environment and dependency lockfile before collection. Save failures and every condition/seed record. Audit all metrics, selectors, budgets, Qiskit probabilities and sample draws; check the old short-budget results against the published archive. Archive only a complete audited series in a new directory. Publish the negative results as well as improvements.
