# What changed when the selector, budget and gradient changed?

11 November 2026. The follow-up completed 360 circuit fits across five targets and three seeds. It tests three possible explanations for the previous failures: composite observables receiving too much weight, insufficient optimization, and noisy gradients.

The result is mixed. Favoring lower-order observables improves some fits, especially four-bit parity, but does not make learned weighting reliable across the targets. Increasing the update budget and removing both sampling sources also leave poor solutions. The classical models using the selected features remain stronger on mean TVD in every condition.

See the [complete tables and figure](observable-diagnostics-tables.md), [all seed-level metrics and paired differences](observable-diagnostics-summary.json), and [frozen protocol](observable-diagnostics-protocol.md). The experiment was designed after inspecting the first comparison, so these are exploratory follow-up results.

## A preference for simpler observables helps selectively

The new selector divides the previous dependence score by `degree - 1`. It retains the same candidate masks, three-feature budget and GF(2) independence rule. On clean and noisy pairs it selects the same individual pairs as the existing pair selector. On parity targets it still selects the true higher-order constraints, while changing some of the additional empirical features.

At 1,600 Monte Carlo updates, adjacent-pair mean TVD falls from 0.5001 with the original higher-order selector to 0.1810 with the order penalty. Shuffled-pair TVD falls from 0.4195 to 0.1883. Both remain substantially worse than the corresponding selected classical model at 0.0270.

For four-bit parity, the order penalty gives TVD 0.0686, compared with 0.0862 for the original higher-order selector and 0.0996 for ordinary MMD. The matched classical model gives 0.0600. On three-bit parity, ordinary MMD remains better: 0.0729 versus 0.2006 for the order penalty. These examples support a narrower conclusion than a generally better selector.

## More updates do not consistently improve target fidelity

Increasing the budget from 400 to 1,600 Monte Carlo updates improves pair-weighted shuffled pairs from 0.4274 to 0.1883, and noisy pairs from 0.3082 to 0.1860. It barely changes ordinary MMD on shuffled pairs, 0.3484 to 0.3483, and makes some parity fits slightly worse against the population target.

The long fits start from the same angles, with a cosine schedule spread over their full budget. They are fresh runs, not continuations. The comparison therefore changes both duration and the learning-rate trajectory. Fitting the empirical training distribution for longer need not improve population TVD.

## Fully deterministic gradients do not remove every failure

The new full control sums all 64 Fourier masks with their exact weights and averages every moment over all 64 uniform strings. It removes both mask and moment sampling. Its gradients match the independent probability-based adjoint oracle in numerical tests.

At 1,600 updates, ordinary full-gradient MMD reaches mean TVD 0.0443 on shuffled pairs and 0.0753 on noisy pairs, compared with 0.3483 and 0.1283 under Monte Carlo moments. But full-gradient higher-order weighting remains near 0.5000 on adjacent pairs and 0.3783 on shuffled pairs. Pair weighting on adjacent pairs is worse with the full control than with Monte Carlo training. Stochasticity can change the optimization trajectory in either direction.

These outcomes show that estimator noise is not the sole explanation for the observed poor fits. They do not identify a unique optimization mechanism or establish convergence. Every fit retains its full-objective loss, final gradient norm and complete history.

## Controls, failures and verification

The five known targets are still exactly expressible by the same 21-parameter circuit. Each condition uses the original 512 training and 1,024 held-out examples. All loss comparisons share data, starting angles and optimizer settings within a budget and estimator. Monte Carlo training uses 8,192 phase evaluations per update; enumerated-moment and full controls use 4,096. There are no quantum training shots.

The archive includes 15 new order-penalized classical fits and reuses the original six learned classical references and oracle per dataset with provenance hashes. Their capacities and optimizer costs are not matched to IQP. The three original order-four maximum-entropy iteration-limit failures remain visible. The new circuit fits completed their fixed budgets without execution exceptions; none asserts optimizer convergence.

The audit checked 480 circuit/classical/reference distributions, reproduced all 90 original short-budget circuit fits exactly apart from elapsed times, and replayed all 360 new-series Adam fits. Numerical tests cover the new selector, full-gradient oracle, deterministic training, resource accounting and rejection of corrupted budgets.

A separate run loaded the archived package source and reproduced all 15 complete records exactly except elapsed times, including the new classical controls and sampled outputs. Its [comparison receipt](verification/observable-diagnostics-v1-replay.json) records the check in the saved Windows environment.

```bash
uv run --locked python -m qgen.diagnostics run --output artifacts/observable-diagnostics-new
uv run --locked python -m qgen.diagnostics audit --output artifacts/observable-diagnostics-new --replay
uv run --locked python scripts/replay_generalization.py --snapshot results/observable-diagnostics-v1 --output artifacts/diagnostics-source-replay-new
uv run --locked python scripts/build_diagnostics_report.py --series results/observable-diagnostics-v1 --destination docs
```

The [new archive](../results/observable-diagnostics-v1/) preserves its protocol, source, configuration, dependency lock, references, data, selected masks, all fits and audit receipt. Previous archives are unchanged. The next optimization question is whether a separately controlled initialization or restart policy can improve success without selecting favorable test results; this experiment did not test that policy.
