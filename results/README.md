# Experiment archive

The seven published baseline series contain 153 records. Four complete follow-up series add 60 records, bringing the successful archive to eleven series and 213 records. Two incomplete routing attempts are retained separately. Each series uses seeds 42, 137 and 2026. A record is one configured job, not one fitted model; the contents per record differ across series.

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
| [observable-diagnostics-v1](observable-diagnostics-v1/) | 15 | Four losses, three estimators and two budgets: 360 circuit fits | [Diagnostics](../docs/observable-diagnostics-protocol.md) |
| [routing-controls-v1](routing-controls-v1/) | 15 | Thirty equal-size circuit fits, 90 compilations and 270 noise evaluations | [Routing controls](../docs/routing-controls-protocol.md) |
| [initialization-restarts-v1](initialization-restarts-v1/) | 15 | Three starting distributions, two update allocations and 900 candidate fits | [Initialization and restarts](../docs/initialization-restarts-protocol.md) |

The [first routing attempt](routing-controls-attempt1/) and [first retry](routing-controls-attempt2/) each contain 15 attempted job records, including execution failures. They are not counted among the eleven complete series. Their original source, configuration, protocol and failed-job index remain intact. The [numerical-check amendment](../docs/routing-controls-protocol.md#numerical-check-amendment-before-the-retry) explains the failure and final tolerance; a [verification receipt](../docs/verification/routing-attempts.json) confirms that all completed training fits match the final run.

Each directory contains:

- `manifest.json`: the frozen configuration, environment and source hashes.
- `index.json`: the complete job list and result-file hashes.
- `runs/`: data, fitted parameters, diagnostics and saved evaluation outputs.
- `source/`: the package source used for that series.
- `audit.json`: the saved audit receipt.

The connectivity records include every tested error level, so the 135 noisy evaluations are nested inside 27 job records. The two larger-system series include initial and final Qiskit samples, classical references and evaluation metrics. All data in this archive are synthetic.

These are immutable experiment snapshots. Use an ignored `artifacts/` directory for new runs, and use the [reproduction guide](../docs/reproduction.md) to replay archived source or recompute audits. The [release summary](../docs/release-summary.json) provides a compact machine-readable view of the six-qubit results; the [results report](../docs/results.md) supplies the interpretation.

The follow-up has a separate [report](../docs/observable-generalization-results.md), [summary](../docs/observable-generalization-summary.json) and replay command. It also saves its protocol and lockfile beside the manifest. The original explorer and release summary remain scoped to the seven baseline series.

The [diagnostic](../docs/observable-diagnostics-results.md) and [routing](../docs/routing-controls-results.md) reports have separate figures and machine-readable summaries. Their `reference/` directories preserve the original classical comparison records and provenance hashes. Use `scripts/replay_generalization.py` with any complete follow-up directory to rerun its archived source.

The [initialization and restart report](../docs/initialization-restarts-results.md) adds 360 policies with training-only candidate selection. Its [complete tables](../docs/initialization-restarts-tables.md) and [summary](../docs/initialization-restarts-summary.json) retain every candidate and paired seed difference. The archive includes the original diagnostic references for 120 exact fit comparisons.
