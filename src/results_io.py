"""CSV schemas and helpers for durable pipeline outputs."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

MATCH_COLUMNS = [
    "pole_oid",
    "pole_lat",
    "pole_lon",
    "image_name",
    "distance_m",
    "match_source",
]
FAILURE_COLUMNS = ["image_name", "error_type", "error_message"]


def default_matches_csv_path(output_path: str | Path) -> Path:
    """Derive a filesystem CSV path beside a shapefile or geodatabase."""
    raw = str(output_path)
    lower = raw.lower()
    gdb_end = lower.rfind(".gdb")
    if gdb_end >= 0:
        gdb_path = Path(raw[: gdb_end + 4])
        feature_name = raw[gdb_end + 4 :].strip("/\\") or "matches"
        feature_name = feature_name.replace("/", "_").replace("\\", "_")
        return gdb_path.parent / f"{gdb_path.stem}_{feature_name}_matches.csv"

    path = Path(output_path)
    return path.with_name(f"{path.stem}_matches.csv")


def failure_csv_path(matches_csv_path: str | Path) -> Path:
    path = Path(matches_csv_path)
    return path.with_name(f"{path.stem}_failures.csv")


def stage_csv_path(final_path: str | Path) -> Path:
    """Reserve a temporary CSV in the destination directory for atomic replace."""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final.stem}.",
        suffix=".tmp",
        dir=final.parent,
    )
    os.close(descriptor)
    return Path(name)


def write_match_csv(path: str | Path, rows: Iterable[dict]) -> int:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["pole_oid"]),
            float(row["distance_m"]),
            str(row["image_name"]),
        ),
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "pole_oid": row["pole_oid"],
                    "pole_lat": f"{float(row['pole_lat']):.8f}",
                    "pole_lon": f"{float(row['pole_lon']):.8f}",
                    "image_name": row["image_name"],
                    "distance_m": f"{float(row['distance_m']):.2f}",
                    "match_source": row["match_source"],
                }
            )
    return len(ordered)


def write_failure_csv(path: str | Path, failures: Iterable[dict]) -> int:
    ordered = sorted(failures, key=lambda row: str(row["image_name"]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    return len(ordered)


def read_match_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MATCH_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Match CSV is missing required column(s): {', '.join(missing)}.")
        rows = list(reader)

    for line_number, row in enumerate(rows, start=2):
        if not row["image_name"].strip():
            raise ValueError(f"Match CSV line {line_number} has an empty image_name.")
        try:
            row["distance_m"] = float(row["distance_m"])
            row["pole_lat"] = float(row["pole_lat"])
            row["pole_lon"] = float(row["pole_lon"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Match CSV line {line_number} contains a non-numeric coordinate or distance."
            ) from exc
    return rows


def read_failure_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FAILURE_COLUMNS:
            raise ValueError("Failure CSV columns must be: " + ", ".join(FAILURE_COLUMNS) + ".")
        return list(reader)
