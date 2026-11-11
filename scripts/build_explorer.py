"""Build an offline explorer from records that pass their saved audit receipts."""

import json
from pathlib import Path

from qgen.experiment import file_digest

BASELINE_SERIES = {
    "reproduction-v1",
    "reproduction-v2",
    "shots-v1",
    "connectivity-v1",
    "iqp-mc-v1",
    "scaling-v1",
    "scaling-pairs-v1",
}


def main():
    root = Path(__file__).resolve().parents[1]
    records, scale = [], []
    for folder in sorted((root / "results").iterdir()):
        if folder.name not in BASELINE_SERIES:
            continue
        if not (folder / "manifest.json").exists():
            continue
        audit = json.loads((folder / "audit.json").read_text())
        index = json.loads((folder / "index.json").read_text())
        if not audit["passed"] or audit["index_sha256"] != file_digest(
            folder / "index.json"
        ):
            raise ValueError(f"Run needs a fresh audit: {folder}")
        for path in sorted((folder / "runs").glob("*.json")):
            if file_digest(path) != index["completed"][path.stem]:
                raise ValueError(f"Changed result: {path}")
            record = json.loads(path.read_text())
            if "job" not in record:
                scale.append(
                    {
                        "series": folder.name,
                        "n": record["n"],
                        "seed": record["seed"],
                        "metrics": record["metrics"],
                        "samples": record["data"]["samples"]["iqp"][:24],
                    }
                )
                continue
            job = record["job"]
            for fit in record.get("fits", [record.get("fit")]):
                classical = job["kind"] == "classical"
                model = (
                    fit["model"]
                    if classical
                    else "/".join(
                        [record["circuit"]["family"], job["topology"], job["loss"]]
                    )
                )
                if job.get("shots"):
                    model += f"/{job['shots']} shots per call"
                records.append(
                    {
                        "dataset": job["dataset"],
                        "series": "classical" if classical else folder.name,
                        "model": model,
                        "seed": job["seed"],
                        "target": record["dataset"]["target"],
                        "p": fit["probabilities"],
                        "metrics": fit["metrics"],
                        "parameters": fit["n_parameters"]
                        if classical
                        else record["circuit"]["n_parameters"],
                        "seconds": fit["training_seconds"],
                        "compilation": {
                            k: record.get("compilation", {}).get(k)
                            for k in ["cx_count", "depth"]
                        },
                        "deployments": [
                            {
                                "error": d["cx_depolarizing"],
                                "p": d["probabilities"],
                                "metrics": d["metrics"],
                            }
                            for d in record.get("deployments", [])
                        ],
                    }
                )
    payload = json.dumps(
        {"records": records, "scaling": scale}, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    template = (root / "scripts" / "explorer.template.html").read_text(encoding="utf-8")
    (root / "docs" / "explorer.html").write_text(
        template.replace("@@DATA@@", payload), encoding="utf-8"
    )
    print(
        f"Built docs/explorer.html with {len(records)} small-model results and {len(scale)} scaling runs"
    )


if __name__ == "__main__":
    main()
