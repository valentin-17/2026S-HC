from __future__ import annotations

import csv
import json
from pathlib import Path
from time import strftime
from typing import Iterable

RESULTS_ROOT = Path("hc02/results")


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in items]


def make_run_dir(run_id: str | None = None) -> Path:
    output_dir = RESULTS_ROOT / (run_id or strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def result_path(name: str) -> Path:
    timestamp = strftime("%Y%m%d-%H%M%S")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT / f"{name}-{timestamp}.csv"


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No benchmark rows were produced.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def seconds_from_ms(milliseconds: float) -> float:
    return milliseconds / 1000.0
