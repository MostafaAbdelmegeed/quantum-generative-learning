"""Reproduce the observable comparison using its archived package and protocol."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from qgen.experiment import digest, file_digest, write_json


def without_times(value):
    if isinstance(value, dict):
        return {
            k: without_times(v)
            for k, v in value.items()
            if k not in {"seconds", "training_seconds"}
        }
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def replay(snapshot, output):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output == snapshot or snapshot in output.parents or output in snapshot.parents:
        raise ValueError("Replay output must be separate from the archive")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    index = json.loads((snapshot / "index.json").read_text())
    if manifest["config_sha256"] != digest(manifest["config"]):
        raise ValueError("Configuration receipt mismatch")
    if manifest["source"]["sha256"] != digest(manifest["source"]["files"]):
        raise ValueError("Source receipt mismatch")
    if platform.python_version() != manifest["python"]:
        raise ValueError("Python version mismatch")
    for name, expected in manifest["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Dependency mismatch: {name}")
    with TemporaryDirectory(prefix="qgen-observable-replay-") as temporary:
        root = Path(temporary)
        package = root / "src/qgen"
        package.mkdir(parents=True)
        for name, checksum in manifest["source"]["files"].items():
            if (
                Path(name).name != name
                or file_digest(snapshot / "source" / name) != checksum
            ):
                raise ValueError("Source file mismatch")
            shutil.copyfile(snapshot / "source" / name, package / name)
        (root / "docs").mkdir()
        for name, checksum in manifest["frozen_files"].items():
            if Path(name).name != name or file_digest(snapshot / name) != checksum:
                raise ValueError("Frozen file mismatch")
            destination = root / (
                "docs/observable-generalization-protocol.md"
                if name == "protocol.md"
                else name
            )
            shutil.copyfile(snapshot / name, destination)
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "qgen.generalization",
                "--config",
                str(root / "config.json"),
                "--output",
                str(output),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "qgen.generalization_audit",
                "--output",
                str(output),
            ],
            cwd=root,
            env=env,
            check=True,
        )
    matched = []
    for name, checksum in index["completed"].items():
        before_path = snapshot / "runs" / f"{name}.json"
        if file_digest(before_path) != checksum:
            raise ValueError(f"Changed archive record: {name}")
        before = json.loads(before_path.read_text())
        after = json.loads((output / "runs" / f"{name}.json").read_text())
        if without_times(before) != without_times(after):
            raise ValueError(f"Replay differs: {name}")
        matched.append(name)
    write_json(
        output / "replay-comparison.json",
        {
            "passed": True,
            "matched_records": matched,
            "source_sha256": manifest["source"]["sha256"],
            "ignored_fields": ["seconds", "training_seconds"],
        },
    )
    print(f"All {len(matched)} records reproduced exactly, excluding elapsed times.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(args.snapshot, args.output)
