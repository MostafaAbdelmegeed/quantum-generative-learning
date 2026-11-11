# Experiment archive

The seven published baseline series contain 153 records. The observable-generalization follow-up adds 15 records, bringing the archive to eight series and 168 records. Each series uses seeds 42, 137 and 2026. A record is one configured job; each of the nine classical jobs in `reproduction-v1` contains four baseline fits. Each follow-up record contains six circuit fits, six learned classical fits and one oracle. Thus the record count is not the number of individual fitted models.

| Series | Records | Question | Protocol |
|---|---:|---|---|
| [reproduction-v1](reproduction-v1/) | 27 | Six-qubit Born machines and four classical baselines | [Initial](../docs/protocol.md) |
| [shots-v1](shots-v1/) | 36 | Fixed-schedule SPSA at 256 and 1,024 shots per call | [Initial](../docs/protocol.md) |
| [connectivity-v1](connectivity-v1/) | 27 | Dense, line and tree IQP circuits before and after noisy routing | [Initial](../docs/protocol.md) |
| [reproduction-v2](reproduction-v2/) | 18 | QCBM convergence with a 2,000-iteration cap | [Follow-up](../docs/protocol-v2.md) |
| [iqp-mc-v1](iqp-mc-v1/) | 27 | Classical Monte Carlo IQP training | [Follow-up](../docs/protocol-v2.md) |
| [scaling-v1](scaling-v1/) | 9 | End-to-end training and sampling at 8, 16 and 32 qubits | [Scaling](../docs/scaling-protocol.md) |
| [scaling-pairs-v1](scaling-pairs-v1/) | 9 | Learned pair weighting under the same mask/update budget | [Pair weighting](../docs/pair-priority-protocol.md) |
| [observable-generalization-v1](observable-generalization-v1/) | 15 | Shuffled/noisy pairs, higher-order parity and a matched-Adam moment-estimation control | [Observable generalization](../docs/observable-generalization-protocol.md) |

Each directory contains:

- `manifest.json`: the frozen configuration, environment and source hashes.
- `index.json`: the complete job list and result-file hashes.
- `runs/`: data, fitted parameters, diagnostics and saved evaluation outputs.
- `source/`: the package source used for that series.
- `audit.json`: the saved audit receipt.

The connectivity records include every tested error level, so the 135 noisy evaluations are nested inside 27 job records. The two larger-system series include initial and final Qiskit samples, classical references and evaluation metrics. All data in this archive are synthetic.

These are immutable experiment snapshots. Use an ignored `artifacts/` directory for new runs, and use the [reproduction guide](../docs/reproduction.md) to replay archived source or recompute audits. The [release summary](../docs/release-summary.json) provides a compact machine-readable view of the six-qubit results; the [results report](../docs/results.md) supplies the interpretation.

The follow-up has a separate [report](../docs/observable-generalization-results.md), [summary](../docs/observable-generalization-summary.json) and replay command. It also saves its protocol and lockfile beside the manifest. The original explorer and release summary remain scoped to the seven baseline series.
