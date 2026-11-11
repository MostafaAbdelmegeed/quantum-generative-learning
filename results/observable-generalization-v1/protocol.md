# Observable weighting beyond adjacent pairs

Frozen on 11 November 2026 before collecting comparison results. Configuration: [observable-generalization-v1.json](../config/observable-generalization-v1.json). This is a bounded exploratory comparison, with no hyperparameter search or checkpoint selection.

## Targets and expressibility

All conditions use six bits and the same dense IQP circuit: six single-X generators and all fifteen XX generators, with 21 independent angles. Columns use Qiskit's little-endian qubit indices.

| Condition | Independent blocks | Odd-parity probability per block |
|---|---|---:|
| Adjacent pairs | (0,1), (2,3), (4,5) | 0 |
| Shuffled pairs | (0,3), (1,4), (2,5) | 0 |
| Noisy pairs | (0,3), (1,4), (2,5) | 0.1 |
| Three-bit parity | (0,2,4), (1,3,5) | 0 |
| Four-bit parity | (0,2,3,5); bits 1 and 4 are independent fair bits | 0 |

Each block is uniform conditional on its parity. The noisy condition flips one designated bit of each clean pair independently with probability 0.1. This is target corruption, not gate noise. Its correct all-even probability is 0.9^3 = 0.729; treating odd outcomes as invalid would reward fitting the wrong distribution.

There is an explicit parameter witness for every target. Within each block choose its first bit as a root and set every root-to-other-bit XX angle to pi/4. The independent star generators map their binary choices bijectively onto the even-parity subspace; each output amplitude has the same magnitude. Set the root's single-X angle to arcsin(sqrt(epsilon)) for odd-parity probability epsilon. The resulting even and odd sectors have disjoint support, so their probabilities mix without interference. Set every unconstrained single-X angle to pi/4 and all remaining angles to zero. Check this construction against both the NumPy circuit and Qiskit before training. The dense support set removes the line-connectivity confound under column permutation. Shuffled and adjacent datasets use the same latent draws with permuted columns; initial angles are identical across losses and estimators within each condition, but are not permuted between conditions.

## Frozen observables and training

Use 512 training examples, 1,024 independent held-out examples and seeds 42, 137, 2026. Select features from training data only, once per dataset.

- Ordinary loss: the existing six-band multiscale MMD Fourier measure.
- Pair weighting: the existing selector's top three absolute centered spin covariances, with lexicographic tie breaking.
- Higher-order weighting: enumerate all 50 masks of degree two through four. Score each by abs(E[product Z] - product E[Z]). Sort by decreasing score, then increasing degree, then lexicographic support. Greedily take three masks independent over GF(2), skipping masks in the span of earlier selections. This avoids spending the small feature budget on products of already selected constraints. The score measures departure from independent marginals; it is not a higher-order cumulant and can include lower-order dependencies.

The selected count of three, maximum degree four and independence filter are explicit priors. They are frozen across targets. Selection can choose sample fluctuations where fewer than three constraints exist. Record all candidate scores, selected supports, character evaluations and selection runtime.

Weighted losses allocate 32 of each step's 64 masks to uniformly sampled selected features and 32 to the original kernel. Ordinary loss uses all 64 ordinary masks. Draw the full ordinary-mask batch for every variant and replace its final half for weighted losses. Independent RNG streams isolate ordinary masks, priority indices and uniform strings; pair and higher-order variants therefore share the same ordinary half. Retain identity masks and signed stochastic losses.

Use 400 Adam updates, the existing uniform-angle initialization, learning rate 0.03 with cosine decay to one tenth, beta1=0.9, beta2=0.999 and epsilon=1e-8. Report final parameters only. Every Monte Carlo fit uses two independent batches of 64 uniform strings per update: 3,276,800 phase evaluations, 51,200 strings and 25,600 masks. All loss variants have the same circuit, data, initialization, update count and estimator budget. No training shots are used.

For every loss run an enumerated-moment Adam control. It uses exactly the same sampled masks and optimizer schedule but averages each moment over all 64 possible strings. Thus it removes moment-estimation noise while retaining mask-sampling noise. It is not a fully deterministic-gradient control. Its phase count is 1,638,400 per fit, with no random strings; report that computational difference rather than claiming equal costs between estimators. Initial/final full-kernel objective and gradient diagnostics are separate from training.

## Classical controls and evaluation

Fit the existing empirical, independent-bit, pairwise maximum-entropy and autoregressive MLP controls on the same training samples. Also fit a maximum-entropy distribution with single-bit and selected higher-order characters, and a larger maximum-entropy model with every character of degree one through four (56 parameters). Use L-BFGS-B, at most 300 iterations and L2=0.0001 for these two added controls. Save optimizer status and warnings, including iteration-limit failures. Classical capacity and optimization costs are not matched to IQP; these are strong fidelity references. An oracle equal to the known target is a separately labeled expressibility ceiling, not a learned baseline.

Primary outcome: exact total variation distance to the known target, with paired differences for all seeds. Also report Jensen-Shannon divergence in nats, the original exact-probability MMD against the held-out empirical distribution, marginal error, each true block moment and all-even probability error. Sample 2,048 outputs from each fitted distribution, using Qiskit Statevector sampling for circuit fits, and save counts and sample diagnostics. The known target enters evaluation and expressibility checks only. The selected-kernel loss is not the evaluation metric.

There are 90 IQP fits and 90 learned classical fits across 15 condition/seed records, plus 15 oracle references. Retain every fit and failure. Three seeds describe sensitivity and do not establish a general population effect or quantum advantage. These small synthetic targets have simple classical generators.

## Execution and records

Before comparison collection, run numerical tests and a timing/memory preflight with arbitrary angles and uniform synthetic training bits. Use one process and one numerical-library thread. Save the preflight receipt; do not use target-fit outcomes to tune settings. The runner refuses existing output directories and snapshots the package source, configuration, protocol, lockfile and environment. Save data, selectors, witnesses, initial/final parameters, complete loss and gradient-norm histories, resource counts, samples and checksums. Audit data regeneration, selection, circuit probabilities, metrics, samples and budgets before copying a complete new series into the archive. Published snapshots remain unchanged.

Reproduction commands will be recorded with the results. A larger-budget follow-up, alternative selector or topology study requires its own configuration and protocol.
