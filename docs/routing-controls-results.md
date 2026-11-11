# Equal-size circuits, routing cost and simulated noise

11 November 2026. Thirty sparse IQP fits, 90 compiled circuits and 270 noise evaluations. The [protocol](routing-controls-protocol.md) holds parameter count at 11 and provides an exact target construction for both circuit families. The [complete tables and figure](routing-controls-tables.md) and [seed-level summary](routing-controls-summary.json) retain every comparison.

Both sparse families fit the pair targets much better than several dense-circuit configurations in the preceding studies. They still struggle on three- and four-bit parity despite known exact solutions. Routing to a star device graph reduces average CX counts and the simulated change from ideal output compared with routing to a line. The result depends on these informed architectures, fixed mappings and synthetic noise settings.

## Ideal fitting remains the first limitation

Each circuit contains the target block's star generators plus extra edges connecting its components. The two families differ in whether those bridge edges form a path between components or connect every component to a hub. Both have six single-X terms and five XX terms. They receive target block labels, so this is not an architecture learned from data.

Both use ordinary multiscale MMD and fully enumerated gradients for 1,600 Adam updates. The table shows three-seed mean exact TVD before routing or noise.

| Target | Path bridges | Hub bridges | Classical selected features |
|---|---:|---:|---:|
| Adjacent pairs | 0.0265 | 0.0265 | 0.0270 |
| Shuffled pairs | 0.0264 | 0.0265 | 0.0270 |
| Noisy pairs | 0.0447 | 0.0400 | 0.0392 |
| Three-bit parity | 0.4341 | 0.4340 | 0.0468 |
| Four-bit parity | 0.3589 | 0.5000 | 0.0658 |

The small clean-pair differences from the classical reference do not establish superiority. The circuit architectures use target structure, only three seeds are tested, and the distributions are easy to sample classically. Equal parameter counts and target expressibility also do not equalize the complete model families or optimizer difficulty.

## Device connectivity changes implementation cost

Compile every fit onto six-node bidirectional line, ring and star device graphs, with identity initial mapping, optimization level three and transpiler seed 17. The table pools all 15 condition/seed fits per logical family. Output change is TVD between the noisy output and that same fitted circuit's ideal output at CX depolarizing parameter 0.02.

| Logical family | Device graph | Mean routed CX | Mean output change |
|---|---|---:|---:|
| Path bridges | Line | 18.47 | 0.1357 |
| Path bridges | Ring | 17.20 | 0.1325 |
| Path bridges | Star | 14.27 | 0.1217 |
| Hub bridges | Line | 20.00 | 0.1612 |
| Hub bridges | Ring | 16.80 | 0.1346 |
| Hub bridges | Star | 13.80 | 0.1167 |

The fixed star layout reduces mean CX cost and mean implementation change here. This is not a general device ranking. Per-condition values and all seeds remain in the complete tables, including the lower error level of 0.005. The noise model includes depolarizing CX and SX/X errors but omits readout errors, relaxation, crosstalk and calibration data.

Noise-induced output change is distinct from target error. Noise can move a poorly fitted distribution toward the target by chance. The saved records therefore include ideal target TVD, noisy target TVD, their signed difference and the distance from ideal circuit output separately.

## Numerical failures and the final verification bound

The first run failed two noiseless compilation checks for shuffled-pair seed 137. The first retry still failed its hub circuit. Both incomplete attempts are preserved with their original source, configuration, protocol, records and tracebacks; they are not successful experiment series.

A regression test and a [check of all 90 compiled circuits](verification/routing-compilation-preflight.json) traced the discrepancy to optimized near-identity rotations. Level-three transpilation changes a probability by at most 8.86e-10 in these circuits. Level-zero compilation of the failing cases agrees within 7e-16. Changing Aer's truncation threshold does not remove the discrepancy, and the compiled statevector agrees with the density-matrix calculation.

The final numerical-check amendment sets a compiled-probability tolerance of 1e-8 and records every observed noiseless error. Training, noise, transpilation settings and seeds remain unchanged. The final run completed all 30 fits. This is a documented verification adjustment after observing failures, not a new performance result chosen from different fitting budgets.

The audit replayed all 30 training fits and all 270 compiled noise evaluations, including the sampled counts. Target witnesses, logical bit order, model probabilities, metric values, parameter counts and budgets passed. Each noisy evaluation uses 2,048 samples, while the reported distributional distances use exact simulated probabilities.

A separate archived-source run reproduced all 15 complete records exactly except elapsed times. The [source-replay receipt](verification/routing-controls-v1-replay.json) records that check; the [attempt-comparison receipt](verification/routing-attempts.json) confirms that every completed training fit in the incomplete attempts matches the final run.

```bash
uv run --locked python -m qgen.routing_controls run --output artifacts/routing-controls-new
uv run --locked python -m qgen.routing_controls audit --output artifacts/routing-controls-new --replay
uv run --locked python scripts/replay_generalization.py --snapshot results/routing-controls-v1 --output artifacts/routing-controls-source-replay-new
uv run --locked python scripts/build_routing_report.py --series results/routing-controls-v1 --destination docs
```

The [complete final archive](../results/routing-controls-v1/), [first attempt](../results/routing-controls-attempt1/) and [first retry](../results/routing-controls-attempt2/) are separate. All execution used local simulation. Hardware execution remains a separate experiment requiring a device, calibration snapshot and concrete shot budget.
