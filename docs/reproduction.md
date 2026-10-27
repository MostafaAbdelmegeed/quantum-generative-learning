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
