"""Export the closest pole for each image from the normalized match CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.results_io import read_match_csv  # noqa: E402

OUTPUT_COLUMNS = [
    "image_name",
    "pole_oid",
    "distance_m",
    "pole_lat",
    "pole_lon",
    "pole_attributes_json",
]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _read_poles_geopandas(data_path: str) -> dict[str, dict[str, Any]]:
    import geopandas as gpd  # type: ignore

    gdf = gpd.read_file(data_path).to_crs(epsg=4326)
    columns = list(gdf.columns)
    lookup = {column.lower(): column for column in columns}
    oid_field = next(
        (lookup[name] for name in ("fid", "objectid", "oid", "oid_") if name in lookup),
        None,
    )
    attribute_fields = [column for column in columns if column not in {"geometry", oid_field}]
    result: dict[str, dict[str, Any]] = {}
    for index, row in gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type != "Point":
            raise ValueError(f"Expected point geometry, found {geometry.geom_type!r}.")
        oid = row[oid_field] if oid_field is not None else index
        result[str(oid)] = {
            "pole_oid": oid,
            "pole_lon": geometry.x,
            "pole_lat": geometry.y,
            "attributes": {field: _json_safe(row[field]) for field in attribute_fields},
        }
    return result


def read_poles(data_path: str) -> dict[str, dict[str, Any]]:
    try:
        from src.arcgis_io import read_pole_attributes

        return read_pole_attributes(data_path)
    except ImportError:
        print("[warn] arcpy not found; trying geopandas ...", file=sys.stderr)
        return _read_poles_geopandas(data_path)


def build_closest_mapping(
    match_rows: list[dict],
    poles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    closest: dict[str, dict[str, Any]] = {}
    for row in match_rows:
        image_name = str(row["image_name"])
        pole_key = str(row["pole_oid"])
        pole = poles.get(pole_key)
        if pole is None:
            raise ValueError(
                f"Match CSV references pole OID {row['pole_oid']!r}, "
                "which is absent from the dataset."
            )
        candidate = {
            "image_name": image_name,
            "pole_oid": pole["pole_oid"],
            "distance_m": float(row["distance_m"]),
            "pole_lat": pole["pole_lat"],
            "pole_lon": pole["pole_lon"],
            "pole_attributes": pole["attributes"],
        }
        current = closest.get(image_name)
        try:
            candidate_oid_key = (0, int(candidate["pole_oid"]))
        except (TypeError, ValueError):
            candidate_oid_key = (1, str(candidate["pole_oid"]))
        try:
            current_oid_key = (0, int(current["pole_oid"])) if current else None
        except (TypeError, ValueError):
            current_oid_key = (1, str(current["pole_oid"])) if current else None
        candidate_key = (candidate["distance_m"], candidate_oid_key)
        if current is None or candidate_key < (
            current["distance_m"],
            current_oid_key,
        ):
            closest[image_name] = candidate
    return dict(sorted(closest.items()))


def write_closest_csv(path: str | Path, closest: dict[str, dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in closest.values():
            writer.writerow(
                {
                    "image_name": record["image_name"],
                    "pole_oid": record["pole_oid"],
                    "distance_m": f"{record['distance_m']:.2f}",
                    "pole_lat": f"{record['pole_lat']:.8f}",
                    "pole_lon": f"{record['pole_lon']:.8f}",
                    "pole_attributes_json": json.dumps(
                        record["pole_attributes"],
                        sort_keys=True,
                        default=_json_safe,
                    ),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches_csv", required=True, help="Normalized CSV written by main.py.")
    parser.add_argument("--data_path", required=True, help="Delivered pole dataset.")
    parser.add_argument("--output", required=True, help="Closest image-to-pole CSV path.")
    args = parser.parse_args(argv)
    try:
        matches = read_match_csv(args.matches_csv)
        poles = read_poles(args.data_path)
        closest = build_closest_mapping(matches, poles)
        write_closest_csv(args.output, closest)
    except (OSError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Match rows read  : {len(matches)}")
    print(f"CSV rows written : {len(closest)}")
    print(f"Output           : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
