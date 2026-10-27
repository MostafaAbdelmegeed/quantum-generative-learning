# Work plan

Started 23 October 2026. The goal is a standalone portfolio project with runnable experiments and a documented increment to existing quantum generative research.

## Completed

- [x] Establish the package, pinned environment and frozen protocols.
- [x] Check distributions, bit order, circuit export, adjoint gradients and losses against independent references.
- [x] Train six-qubit QCBMs and four classical baselines on paired samples.
- [x] Compare losses under a fixed finite-measurement budget; retain the weak SPSA results.
- [x] Compare dense, line and equally sparse random-tree IQP circuits after routing and simulated noise.
- [x] Rerun QCBMs with a larger iteration cap after the initial fits failed to converge.
- [x] Implement classical Monte Carlo IQP training and compare with exact training.
- [x] Train and sample 8-, 16- and 32-qubit circuits on a classically tractable target.
- [x] Test training-only pair selection at the same mask/update budget. Mean 32-qubit sample validity improved from 81.0% to 99.6%.
- [x] Audit all seven series, replay simulations, reproduce an archived experiment, and build the demo and explorer.

The release contains 153 experiment records, 31 passing local tests and a CI workflow. GitHub CI and physical quantum hardware have not been used.

## Next bounded research step

Test whether learned observables help beyond the easy paired-bit construction. Freeze permuted pairs, partially corrupted pairs and higher-order parity blocks before running them. Compare generic MMD, learned pair weighting and a higher-order observable selector at equal budgets. Include a classical model that can learn the same structures. Report complete-sample validity as well as MMD.

A separate control should compare exact and estimated gradients under the same Adam schedule. The present L-BFGS-B versus Adam comparison cannot isolate estimator noise. A hardware-oriented extension should fix parameter count while selecting interactions, then evaluate measured routing cost under more than one device layout.

Publishing and running a physical device are separate release steps. A hardware experiment needs a concrete shot budget and device configuration; the current results do not require one.

Each stage depends on the numerical checks from the previous stage. All runs use local resources. A hardware run would be a separate experiment with its own resource budget.
