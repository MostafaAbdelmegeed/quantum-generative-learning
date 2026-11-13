# Reproducing the experiments

The recorded environment uses Python 3.11.15, Qiskit 2.6.1 and Qiskit Aer 0.18.0. `uv.lock` pins the remaining dependencies. All experiments use local CPU simulation and synthetic data.

```bash
uv sync --locked --extra dev --python 3.11.15
uv run --locked --extra dev python -m pytest
uvx ruff@0.17.2 check src tests examples scripts
uvx ruff@0.17.2 format --check src tests examples scripts
```

## Run from the current source

Use a fresh directory under `artifacts/`, which Git ignores. The runner can resume an existing directory only if its configuration and package-source hashes match. The published result folders preserve older source versions and should remain unchanged.

```bash
uv run --locked qgen run --config config/reproduction-v1.json --output artifacts/reproduction-v1 --workers 3
uv run --locked qgen run --config config/shots-v1.json --output artifacts/shots-v1 --workers 3
uv run --locked qgen run --config config/connectivity-v1.json --output artifacts/connectivity-v1 --workers 3
uv run --locked qgen run --config config/reproduction-v2.json --output artifacts/reproduction-v2 --workers 3
uv run --locked qgen run --config config/iqp-mc-v1.json --output artifacts/iqp-mc-v1 --workers 3
uv run --locked qgen scaling --config config/scaling-v1.json --output artifacts/scaling-v1 --workers 3
uv run --locked qgen scaling --config config/scaling-pairs-v1.json --output artifacts/scaling-pairs-v1 --workers 3
```

Each command runs the three frozen seeds. The six-qubit runs save exact evaluation probabilities; the larger runs save sampled bitstrings and avoid full statevectors. See the [archive index](../results/README.md) for record counts and protocol links.

## Replay the archived training source

```bash
uv run --locked python scripts/replay_snapshot.py --snapshot results/reproduction-v1 --output artifacts/reproduction-replay --workers 3
```

The replay script checks Python and dependency versions, the lockfile hash and every archived source file before running the saved package. It audits the new outputs, then compares every field except `seconds` and `training_seconds`. Change `--snapshot` and use a fresh `--output` to replay another series.

The initial 27-record series matched under the recorded Windows environment; its [comparison receipt](verification/reproduction-v1-replay.json) is tracked. Exact field equality is stricter than numerical agreement. BLAS, floating-point and compiler differences can prevent identical records on another platform. CI's numerical tests and artifact audits are separate from a full training replay.

## Observable generalization follow-up

This series uses its own runner and auditor. It refuses existing output directories, uses one worker and keeps the five-condition comparison separate from the baseline release report.

```bash
uv run --locked python -m qgen.generalization --config config/observable-generalization-v1.json --output artifacts/observable-generalization-new
uv run --locked python -m qgen.generalization_audit --output artifacts/observable-generalization-new --replay
uv run --locked python scripts/replay_generalization.py --snapshot results/observable-generalization-v1 --output artifacts/observable-generalization-replay-new
uv run --locked python scripts/build_generalization_report.py --series results/observable-generalization-v1 --destination docs
```

The `--replay` audit repeats all 90 Adam fits using the loaded source. The separate replay script loads the saved package source, configuration and protocol, checks dependency versions, reruns all circuit and classical fits, and compares complete records excluding elapsed times. All 15 archived records reproduced exactly in the recorded Windows environment; the [receipt](verification/observable-generalization-v1-replay.json) preserves that check. The [report](observable-generalization-results.md) includes all seed differences, classical optimizer-limit failures and limitations.

## Diagnostic and routing follow-ups

```bash
uv run --locked python -m qgen.diagnostics run --output artifacts/observable-diagnostics-new
uv run --locked python -m qgen.diagnostics audit --output artifacts/observable-diagnostics-new --replay
uv run --locked python -m qgen.routing_controls run --output artifacts/routing-controls-new
uv run --locked python -m qgen.routing_controls audit --output artifacts/routing-controls-new --replay
uv run --locked python scripts/replay_generalization.py --snapshot results/observable-diagnostics-v1 --output artifacts/diagnostics-source-replay-new
uv run --locked python scripts/replay_generalization.py --snapshot results/routing-controls-v1 --output artifacts/routing-source-replay-new
uv run --locked python scripts/build_diagnostics_report.py --series results/observable-diagnostics-v1 --destination docs
uv run --locked python scripts/build_routing_report.py --series results/routing-controls-v1 --destination docs
```

Both runners require fresh directories. Diagnostic auditing replays 360 Adam fits and checks the 90 original short-budget fits against their published references. Routing auditing replays 30 fits and 270 noise evaluations, including sample counts. The source-replay command copies the archived package, frozen protocol and saved reference records into a temporary environment and compares full records apart from timings. The incomplete routing attempts are preserved for diagnosis; this replay command intentionally rejects incomplete series.

Both complete series reproduced every record from archived source in the recorded Windows environment. See the [diagnostic receipt](verification/observable-diagnostics-v1-replay.json) and [routing receipt](verification/routing-controls-v1-replay.json). This exact-equality claim is separate from numerical portability across operating systems.

The diagnostic auditor reconstructs the added classical model's probabilities from its saved parameters and training-selected feature masks. Repeating its L-BFGS-B fit belongs to `--replay`: a fresh Ubuntu fit did not match the Windows parameter vector within 1e-11, even with pinned package versions. Ordinary audits retain the 1e-11 tolerance for reconstructed probabilities and metrics. Archived source and result bytes are unchanged.

CI runs saved-result audits and numerical tests on both Windows and Ubuntu. Exact routing compilation and simulation replay runs on Windows, the archive's original platform. Ubuntu reached the compilation comparison but produced different compilation metadata, which includes the complete QASM text. This platform restriction leaves the strict replay check intact; it does not assert identical compiled circuits across operating systems.

## Initialization and restart follow-up

```bash
uv run --locked python -m qgen.restarts run --output artifacts/initialization-restarts-new
uv run --locked python -m qgen.restarts audit --output artifacts/initialization-restarts-new --replay
uv run --locked python scripts/replay_generalization.py --snapshot results/initialization-restarts-v1 --output artifacts/initialization-source-replay-new
uv run --locked python scripts/build_restarts_report.py --series results/initialization-restarts-v1 --destination docs
```

The runner requires a fresh directory and copies the original diagnostic references. Each of the 360 policies receives 1,600 gradient updates, split into one run or four independent starts. Candidate selection sees only final training loss. Auditing checks every candidate's probabilities, samples, metrics, initialization and resource count; `--replay` repeats all 900 fits. It also checks exact agreement with 120 previously published fits. The [source-replay receipt](verification/initialization-restarts-v1-replay.json) records agreement for all 15 records apart from elapsed times in the recorded Windows environment.

The [results report](initialization-restarts-results.md) explains the initialization effects and the extra diagnostic and evaluation work required by restarts. The [tables](initialization-restarts-tables.md) retain all configurations and paired differences.

## Audit saved results

```bash
uv run --locked qgen audit --output results/iqp-mc-v1
uv run --locked qgen audit --output results/connectivity-v1 --deep
uv run --locked qgen audit --output results/scaling-pairs-v1
```

The small-system auditor checks hashes, budgets, regenerated inputs, independent metrics and Qiskit probabilities. The deep connectivity audit also replays 135 noisy evaluations. The larger-run audit regenerates the inputs and classical baselines, checks selected pair masks, and replays initial and trained MPS samples.

Audits verify the saved evidence; they do not establish optimizer optimality or quantum advantage. The [results report](results.md) describes the scientific limits.

## Rebuild figures and the explorer

```bash
uv run --locked qgen report --results results/reproduction-v1 results/shots-v1 results/connectivity-v1 --output docs
uv run --locked python scripts/build_release_report.py
uv run --locked python scripts/build_overview.py
uv run --locked python scripts/build_explorer.py
```

The original report command accepts only the frozen three-series comparison so that it cannot silently combine different training budgets. `build_release_report.py` audits all seven series before producing the follow-up figures and `docs/release-summary.json`. The overview is an explanatory diagram; its sample rows come from the archived eight-qubit run. The explorer verifies its inputs against the audit and index hashes.

The repository stores archived result bytes without Git line-ending conversion. Keep `.gitattributes` in place: the receipts hash files as bytes, including their original line endings. New exploratory runs belong in `artifacts/`; add a new versioned series to `results/` only with its configuration, source snapshot, complete records and passing audit.
