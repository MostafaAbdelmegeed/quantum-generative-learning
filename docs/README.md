# Documentation

Start with the [results report](results.md) for the measured outcomes, or the [method notes](iqp-method-notes.md) for the IQP estimator and its conventions.

| Document | Contents |
|---|---|
| [Reproduction guide](reproduction.md) | Installation, new runs, archived-source replay and figure generation |
| [Initial protocol](protocol.md) | Six-qubit Born machines, finite-shot training and connectivity experiments |
| [Optimization and estimator follow-up](protocol-v2.md) | Larger optimization budget and classical Monte Carlo IQP training |
| [Scaling protocol](scaling-protocol.md) | Paired-bit tasks at 8, 16 and 32 qubits |
| [Pair-weighting protocol](pair-priority-protocol.md) | Training-data correlation selection under a fixed estimator budget |
| [Observable generalization protocol](observable-generalization-protocol.md) | Expressible shuffled/noisy pairs and higher-order parity, selectors and matched loss budgets |
| [Observable generalization results](observable-generalization-results.md) | The bounded follow-up, classical controls, optimizer failures and every paired seed difference |
| [Observable diagnostics](observable-diagnostics-results.md) | Order penalties, longer budgets and fully enumerated gradients; 360 fits |
| [Routing controls](routing-controls-results.md) | Equal-size expressible circuits, three device graphs and 270 simulated noise evaluations |
| [Initialization and restarts](initialization-restarts-results.md) | Three starting distributions and two allocations of 1,600 updates; 900 candidate fits |
| [Results report](results.md) | Classical baselines, every experiment family and limits of the conclusions |
| [IQP method notes](iqp-method-notes.md) | Derivations, implementation conventions and pinned upstream references |
| [Work plan](plan.md) | Completed work and remaining research questions |

The [release summary](release-summary.json) contains audit receipts and all six-qubit metric rows. The [experiment index](../results/README.md) maps those rows to their complete records. The [replay receipt](verification/reproduction-v1-replay.json) records the initial series' successful rerun from archived source, excluding elapsed-time fields from the comparison.

Open [the explorer](explorer.html) from a local clone to inspect results interactively. It needs no server or network connection. GitHub's file viewer shows its source rather than running it.
