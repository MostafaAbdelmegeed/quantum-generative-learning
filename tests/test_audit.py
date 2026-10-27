import json
import subprocess
import sys

import pytest

from qgen.audit import audit_experiment
from qgen.experiment import file_digest, run_experiment, write_json


@pytest.fixture
def tiny_run(tmp_path):
    config = {
        "experiment": "audit-test",
        "mode": "exact",
        "n_qubits": 2,
        "datasets": ["even_parity"],
        "seeds": [7],
        "train_size": 16,
        "test_size": 32,
        "family": "iqp",
        "topologies": ["line"],
        "depth": 1,
        "losses": ["multiscale"],
        "maxiter": 15,
        "classical_baselines": False,
    }
    path = tmp_path / "config.json"
    output = tmp_path / "run"
    write_json(path, config)
    run_experiment(path, output, workers=1)
    run_experiment(path, output, workers=1)
    assert audit_experiment(output)["jobs"] == 1
    return output


def test_audit_rejects_changed_result_even_with_updated_checksum(tiny_run):
    path = next((tiny_run / "runs").glob("*.json"))
    record = json.loads(path.read_text())
    record["fit"]["metrics"]["tvd"] += 0.01
    write_json(path, record)
    index = json.loads((tiny_run / "index.json").read_text())
    index["completed"][path.stem] = file_digest(path)
    write_json(tiny_run / "index.json", index)
    with pytest.raises(ValueError, match="tvd"):
        audit_experiment(tiny_run)
    result = subprocess.run(
        [sys.executable, "-O", "-m", "qgen.cli", "audit", "--output", str(tiny_run)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "tvd" in result.stderr


def test_audit_rejects_changed_source_snapshot(tiny_run):
    snapshot = tiny_run / "source" / "circuits.py"
    snapshot.write_text(snapshot.read_text() + "\n# local modification\n")
    with pytest.raises(ValueError, match="Source snapshot mismatch"):
        audit_experiment(tiny_run)
