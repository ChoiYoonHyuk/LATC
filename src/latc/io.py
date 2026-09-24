"""Portable experiment outputs and provenance."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=json_default, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=json_default) if isinstance(v, (list, tuple, dict)) else v
                             for k, v in row.items()})


def write_jsonl_gzip(path: str | Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")


def manifest(config: dict, config_path: Path) -> dict:
    import scipy
    import matplotlib
    from . import __version__
    return dict(package_version=__version__, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, config=config,
                thread_environment={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                randomness="PCG64/SeedSequence; explicit stream map in docs/REPRODUCIBILITY.md",
                integration="exact finite outcome sums use float64; risk certificates use directed Decimal intervals")
