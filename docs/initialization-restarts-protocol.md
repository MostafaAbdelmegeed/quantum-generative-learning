# Initialization and restarts at a fixed update budget

Frozen on 13 November 2026 before collecting `initialization-restarts-v1`. The previous comparison found poor fits even with exact gradients and known target parameters. This experiment tests whether the starting distribution and the allocation of updates across starts change that result.

## Scope and controls

Reuse the five six-bit targets, three dataset seeds, 512 training examples and 1,024 held-out examples from `observable-diagnostics-v1`. Keep the same dense IQP family with six single-X and fifteen XX generators. All targets have verified parameter witnesses in this family. Use all four existing losses: ordinary multiscale MMD, pair weighting, higher-order weighting and order-penalized weighting. Keep their selectors, three-feature budget and half-priority mixture unchanged. Reuse the seven learned classical controls and oracle from the diagnostic archive, with their provenance hashes.

Use fully enumerated moments and masks for every fit. This isolates the initialization and restart comparison from gradient sampling. All fits use Adam with beta1=0.9, beta2=0.999, epsilon=1e-8, and a cosine learning-rate schedule from 0.03 to 0.003 over that candidate's allocated updates. No early stopping or checkpoint selection is allowed.

## Starting parameters

Compare three initialization families. Each supplies four candidate starts per dataset, shared across losses and allocation policies.

- Uniform angles use the existing uniform distribution on [-pi, pi]. Candidate zero is exactly the previous seed's initialization. Candidates one through three use `SeedSequence([dataset_seed, 3107, candidate_index])` with the same uniform sampler.
- Near-identity angles use independent normal draws with standard deviation 0.1 around zero. A zero angle vector produces the all-zero basis state; perturbations avoid starting exactly at its zero gradient. Use `SeedSequence([dataset_seed, 7919, candidate_index])`.
- Marginal-product angles begin at the independent distribution fitted to the training marginals: single-X angle `arcsin(sqrt(training_probability_of_one))`, with all XX angles zero. Add independent normal perturbations of standard deviation 0.05 to every angle, using the same normal stream as the near-identity family. The perturbation means the actual start is only near the fitted product distribution. This initialization uses training data only.

The two standard deviations are frozen choices, not tuned against these outcomes. Save every starting vector and the training marginals. A numerical test checks the unperturbed product construction independently through circuit probabilities.

## Allocation and selection

Each policy receives exactly 1,600 gradient updates and 6,553,600 training phase evaluations. The single-start policy gives all updates to candidate zero. The restart policy gives 400 updates to each of the four candidates. Every candidate starts fresh, including Adam's moment accumulators. Each schedule decays over its own duration, so the allocation comparison changes both the number of starts and the per-start trajectory.

Select the restart with the lowest final exact empirical training loss under the loss used for that fit. Break exact ties by candidate index. The selection function accepts only final training losses. It receives no target probabilities, held-out examples, TVD or sample diagnostics. Select before evaluating any candidate, then evaluate and retain all candidates, including those not selected. Do not choose across losses or initialization families.

Initial and final exact-objective diagnostics add two phase computations per candidate outside the optimizer budget. Count these explicitly: the single policy uses 6,561,792 phase evaluations including diagnostics; the four-start policy uses 6,586,368. The latter's extra diagnostic cost is 24,576, or 0.375% of the common training budget. Report this difference rather than claiming identical total cost. Final diagnostic losses also provide the selection scores; selection does not trigger extra objective evaluations.

The factorial has 360 policy outputs and 900 candidate fits: five conditions, three seeds, four losses, three initializations and two allocations. The uniform single-start fits should reproduce 60 published 1,600-update full-gradient fits. Candidate zero of uniform restarts should reproduce the corresponding 60 published 400-update fits. Verify those 120 comparisons against the saved diagnostic records.

## Outcomes and verification

Primary outcome is exact TVD to the known target for the training-selected candidate. Report every paired restart-minus-single and initialization-minus-uniform difference across the three dataset seeds. Keep Jensen-Shannon divergence, original held-out MMD, block moments, marginal error and noisy-target all-even probability error. Save complete optimizer histories, final full-objective gradients and 2,048 Qiskit samples for every candidate. All candidate outcomes may be shown descriptively, but an evaluation-selected best candidate is not a policy result.

Training loss and target TVD measure different quantities. A restart can lower empirical loss without improving the population distribution. The comparison does not identify global minima, estimate success probabilities from only three dataset seeds or demonstrate quantum advantage. It is an exploratory response to previously observed failures.

Run one worker and one BLAS thread after a runtime and memory preflight on arbitrary data. Preserve source, configuration, protocol, lockfile, data, original reference records, all candidates and failures. Numerical tests must verify the initialization construction, deterministic starts, shared budgets, training-only selection and agreement with the existing optimizer. Audit probabilities with Qiskit and the NumPy engine, metrics, selector regeneration, resources, samples and both published replay comparisons. Replay the complete archived source before publishing a new series. Previous snapshots remain unchanged.
