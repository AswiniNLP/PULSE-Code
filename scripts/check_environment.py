#!/usr/bin/env python3
"""Inspect local dependencies, GPU availability and optional CSV inputs."""

import argparse
import csv
from importlib import metadata
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--check-ml", action="store_true", help="Actually import the model stack")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    report = {"python": sys.version.split()[0], "packages": {}, "errors": []}
    for package in ("torch", "transformers", "peft", "huggingface-hub", "accelerate",
                    "tokenizers", "safetensors", "numpy", "bitsandbytes"):
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
            if package != "bitsandbytes":
                report["errors"].append(f"Missing core package: {package}")
    if args.check_ml or args.require_cuda:
        try:
            import torch
            import transformers
            import peft

            report["ml_imports"] = {"transformers": transformers.__version__, "peft": peft.__version__}
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if args.require_cuda and not report["cuda_available"]:
                report["errors"].append("CUDA is required but unavailable")
        except Exception as exc:
            report["errors"].append(f"Model stack import failed: {exc}")
    if args.train_csv:
        try:
            with args.train_csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"Hatespeech", "Style", "Target", "Counterspeech"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Missing CSV columns: {sorted(missing)}")
                rows = list(reader)
            if not rows:
                raise ValueError("Training CSV is empty")
            for index, row in enumerate(rows, start=2):
                if any(not (row.get(key) or "").strip() for key in required):
                    raise ValueError(f"Blank required value on CSV line {index}")
                if row["Style"] not in {"Gandhi", "Mandela"}:
                    raise ValueError(f"Unsupported Style on CSV line {index}: {row['Style']!r}")
            ids = [row["id"] for row in rows] if "id" in rows[0] else None
            if ids is not None and (any(not key.strip() for key in ids) or len(set(ids)) != len(ids)):
                raise ValueError("CSV id values must be nonempty and unique")
            report["training_data"] = {"rows": len(rows), "styles": sorted({r["Style"] for r in rows}),
                                       "targets": sorted({r["Target"] for r in rows})}
        except (OSError, ValueError, csv.Error) as exc:
            report["errors"].append(str(exc))
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
