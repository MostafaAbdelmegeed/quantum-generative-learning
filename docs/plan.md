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

The published baseline contains 153 experiment records and passed 31 tests. Its Windows and Ubuntu GitHub checks passed on 11 November 2026. Physical quantum hardware has not been used.

## Observable generalization follow-up

- [x] Freeze six-bit adjacent pairs, shuffled pairs, noisy pairs and three- and four-bit parity targets.
- [x] Verify explicit target parameters with NumPy and Qiskit in the same 21-parameter dense circuit.
- [x] Compare ordinary MMD, learned pair weighting and higher-order weighting with matched data, initialization and Monte Carlo budgets.
- [x] Compare sampled and enumerated moments using identical masks and Adam schedules.
- [x] Run 90 circuit fits and 90 learned classical fits; retain three classical iteration-limit failures and poor circuit solutions.
- [x] Audit all 195 fitted/reference distributions and replay all 90 Adam fits. The expanded numerical suite passes 42 tests.

The [follow-up report](observable-generalization-results.md) finds no reliable benefit from learned weighting under this bounded schedule. The classical model using the selected observables has lower mean TVD in every condition. The new archive adds 15 condition/seed records without changing any published baseline snapshot.

## Next bounded research step

Separate selector design from optimization sensitivity before increasing width. The higher-order selector often chooses products of pair constraints on clean-pair data, while some poor solutions persist with enumerated moments. A new protocol could test a preference for lower-order features and a separately controlled initialization or longer-budget comparison. Keep the current negative results as the baseline; do not retune this archived series.

The new Adam control isolates moment-estimation noise while retaining mask-sampling noise. A fully enumerated-mask control would answer a different question. A hardware-oriented extension should fix parameter count while selecting interactions, then evaluate measured routing cost under more than one device layout.

Publishing and running a physical device are separate release steps. A hardware experiment needs a concrete shot budget and device configuration; the current results do not require one.

Each stage depends on the numerical checks from the previous stage. All runs use local resources. A hardware run would be a separate experiment with its own resource budget.
