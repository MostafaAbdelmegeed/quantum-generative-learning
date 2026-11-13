# Initialization replication on untouched seeds

Frozen on 13 November 2026 before collecting replication outcomes. The first initialization study found large improvements under order-penalized weighting with small random starting angles, but used only three dataset seeds. This replication tests the same policies on new data and initialization draws.

## Frozen comparison

Use seeds 1001 through 1010 inclusive. These ten consecutive seeds do not occur in the existing experiment configurations, whose dataset seeds are 42, 137 and 2026. Do not screen seeds, discard poor outcomes, tune parameters or add runs after inspecting results. Development tests use other seeds. The old three-seed results remain a discovery set and are reported separately.

Retain all five six-bit targets, the dense 21-parameter IQP circuit, 512 training examples, 1,024 held-out examples, four losses, selectors, initialization scales, schedules and update allocations from the [first protocol](initialization-restarts-protocol.md). Uniform initialization uses the original random stream for candidate zero and separate indexed streams for the other three. Near-identity initialization uses normal angles with standard deviation 0.1. Marginal-product initialization fits the training marginals and adds normal perturbations with standard deviation 0.05. The two normal families share standardized draws. All starting vectors are shared across losses and allocations within a dataset seed.

Use exact enumerated gradients throughout. Each policy receives 1,600 Adam updates, either in one run or four 400-update runs with fresh optimizer state and a learning-rate schedule over each candidate's own duration. Select a restart by its final empirical training loss, breaking exact ties by candidate index. Selection must happen before target or held-out evaluation. Retain and evaluate every candidate.

The factorial contains 50 condition/seed records, 1,200 policy outputs and 3,000 circuit fits. Each policy uses 6,553,600 training phase evaluations. Initial and final diagnostics bring the single-start total to 6,561,792 and the restart total to 6,586,368. The extra diagnostic cost is unchanged from the first study. Every candidate receives 2,048 Qiskit evaluation samples; evaluation work is separate from training.

## Primary outcome and decision rule

The primary comparison is near-identity minus uniform exact target TVD under order-penalized weighting with one 1,600-update run. Compute the paired difference for every new seed in each of the five conditions. Negative values favor near-identity initialization. Report the mean, median, minimum, maximum and number of improvements, ties and regressions, with a numerical tie threshold of 1e-8.

For a bounded descriptive replication criterion, call an initialization benefit replicated in a condition only if its mean paired TVD difference is below -0.01 and at least eight of ten paired differences are below -1e-8. Report conditions that fail either requirement. This rule measures consistency and a practical effect size; it is not a significance test or a population guarantee. Adjacent and shuffled pairs share latent data draws, so they are not independent evidence.

For the two parity conditions, separately compare the near-identity single policy with the order-penalized selected classical model on each seed. Report all paired differences and their mean and range. Treat the previous small classical-comparison gains as exploratory until they persist on these new datasets. Do not choose a different loss, initialization or allocation after seeing target TVD.

Report all other policies, all seven learned classical controls, secondary metrics, training-loss differences and restart-minus-single differences. Do not pool the old and new seeds for the primary conclusions. Do not rank policies by target performance and present the winner as a prespecified method.

## Classical controls and verification

Refit empirical, independent, pairwise, autoregressive MLP, higher-order selected maximum-entropy, order-four maximum-entropy and order-penalized selected maximum-entropy models on each new training sample. Use the same fitting code and settings as the previous studies: 300 classical optimizer iterations and 0.0001 regularization for the log-linear models, with the existing autoregressive settings. This adds 350 learned fits and 50 known-target oracle references. Save optimizer warnings and unsuccessful stopping outcomes rather than excluding them. Capacities and training costs are not matched between classical models and circuits.

Fresh classical controls are stored directly in each replication record. No old-seed fit can serve as a reference for new data, and the replication does not claim to reproduce old-seed fits. Check fresh data generation, selector and initialization regeneration, complete policy and candidate sets, budgets, exact probabilities through Qiskit and NumPy, independent objective diagnostics, all metrics and sampling. Verify classical masks and saved probabilities directly where parameters are stored; repeat every classical fit during the local replay audit. A separate complete archived-source run must reproduce all new circuit and classical records apart from elapsed times in the recorded environment.

Run one worker with one BLAS thread after the arbitrary-data runtime and memory preflight. Preserve the frozen configuration, protocol, source, lockfile, raw data, all starts, all histories, evaluations and any failures. Publish only a complete passing series; keep earlier snapshots unchanged. These experiments use synthetic six-bit distributions and simulators. They do not establish quantum advantage or larger-system performance.
