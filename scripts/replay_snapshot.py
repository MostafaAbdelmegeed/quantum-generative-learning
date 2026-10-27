"""Rerun an archived experiment with its exact saved package source."""

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

from qgen.audit import audit_experiment
from qgen.experiment import digest, file_digest, write_json
from qgen.scaling import audit_scaling


def without_timings(value):
    if isinstance(value, dict):
        return {
            k: without_timings(v)
            for k, v in value.items()
            if k not in {"seconds", "training_seconds"}
        }
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    snapshot, output = Path(args.snapshot).resolve(), Path(args.output).resolve()
    if output == snapshot or snapshot in output.parents or output in snapshot.parents:
        raise ValueError("Replay output must be separate from the archived experiment")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    index = json.loads((snapshot / "index.json").read_text())
    if manifest["config_sha256"] != digest(manifest["config"]):
        raise ValueError("Archived configuration receipt mismatch")
    if manifest["source"]["sha256"] != digest(manifest["source"]["files"]):
        raise ValueError("Archived source receipt mismatch")
    for package, expected in manifest["dependencies"].items():
        if version(package) != expected:
            raise ValueError(
                f"Replay requires {package}=={expected}; synchronize the locked environment"
            )
    if platform.python_version() != manifest["python"]:
        raise ValueError(f"Replay requires Python {manifest['python']}")
    root = Path(__file__).resolve().parents[1]
    if manifest.get("lockfile_sha256") != file_digest(root / "uv.lock"):
        raise ValueError("The current dependency lock differs from the archived run")
    with TemporaryDirectory(prefix="qgen-replay-") as temporary:
        package = Path(temporary) / "qgen"
        package.mkdir()
        for name, checksum in manifest["source"]["files"].items():
            if (
                Path(name).name != name
                or file_digest(snapshot / "source" / name) != checksum
            ):
                raise ValueError("Archived source snapshot mismatch")
            shutil.copyfile(snapshot / "source" / name, package / name)
        config_path = Path(temporary) / "config.json"
        write_json(config_path, manifest["config"])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(temporary)
        command = "scaling" if manifest["config"]["mode"] == "scaling" else "run"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "qgen.cli",
                command,
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--workers",
                str(args.workers),
            ],
            env=env,
            check=True,
        )
        (output / "source").mkdir(exist_ok=True)
        for name, checksum in manifest["source"]["files"].items():
            destination = output / "source" / name
            if not destination.exists():
                shutil.copyfile(package / name, destination)
            if file_digest(destination) != checksum:
                raise ValueError("Replayed source snapshot mismatch")
    auditor = audit_scaling if command == "scaling" else audit_experiment
    auditor(output)
    matched = []
    for name, checksum in index["completed"].items():
        original = snapshot / "runs" / f"{name}.json"
        if file_digest(original) != checksum:
            raise ValueError(f"Archived record changed: {name}")
        before, after = [
            json.loads(p.read_text())
            for p in [original, output / "runs" / f"{name}.json"]
        ]
        if without_timings(before) != without_timings(after):
            raise ValueError(f"Replay differs from the archived result: {name}")
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
    main()
