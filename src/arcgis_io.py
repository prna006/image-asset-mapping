"""ArcPy-backed dataset I/O with no import-time ArcGIS dependency."""

from __future__ import annotations

import importlib
import math
from typing import Any

SHAPEFILE_TEXT_LIMIT = 254
DEFAULT_COUNT_FIELD = "match_cnt"


def _arcpy():
    """Import ArcPy only when an ArcGIS operation is requested."""
    return importlib.import_module("arcpy")


def dataset_exists(data_path: str) -> bool:
    return bool(_arcpy().Exists(data_path))


def delete_dataset(data_path: str) -> None:
    arcpy = _arcpy()
    if arcpy.Exists(data_path):
        arcpy.management.Delete(data_path)


def copy_features(src_path: str, out_path: str) -> str:
    arcpy = _arcpy()
    previous = arcpy.env.overwriteOutput
    try:
        arcpy.env.overwriteOutput = True
        arcpy.management.CopyFeatures(src_path, out_path)
    finally:
        arcpy.env.overwriteOutput = previous
    print(f"  [arcgis_io] Copied '{src_path}' -> '{out_path}'.")
    return out_path


def validate_pole_dataset(data_path: str) -> int:
    """Validate the pole input contract and return its usable point count."""
    arcpy = _arcpy()
    if not arcpy.Exists(data_path):
        raise ValueError(f"Pole dataset does not exist: {data_path}")
    description = arcpy.Describe(data_path)
    if str(getattr(description, "shapeType", "")).lower() != "point":
        raise ValueError(
            f"Pole dataset must contain point geometry; found "
            f"{getattr(description, 'shapeType', 'unknown')!r}."
        )
    spatial_reference = getattr(description, "spatialReference", None)
    if (
        spatial_reference is None
        or str(getattr(spatial_reference, "name", "Unknown")).lower() == "unknown"
    ):
        raise ValueError("Pole dataset must have a defined spatial reference.")
    if not getattr(description, "OIDFieldName", None):
        raise ValueError("Pole dataset must have an object ID field.")

    wgs84 = arcpy.SpatialReference(4326)
    count = 0
    with arcpy.da.SearchCursor(data_path, ["OID@", "SHAPE@XY"], spatial_reference=wgs84) as cursor:
        for oid, xy in cursor:
            if xy is None or len(xy) != 2:
                raise ValueError(f"Pole OID {oid} has empty geometry.")
            lon, lat = xy
            if not all(math.isfinite(float(value)) for value in (lon, lat)):
                raise ValueError(f"Pole OID {oid} has non-finite coordinates.")
            if not -180 <= float(lon) <= 180 or not -90 <= float(lat) <= 90:
                raise ValueError(
                    f"Pole OID {oid} could not be projected to valid WGS84 coordinates."
                )
            count += 1
    if count == 0:
        raise ValueError("Pole dataset contains no usable point features.")
    return count


def load_poles(
    data_path: str,
    height_field: str = "HEIGHT",
    default_height_m: float = 15.0,
) -> list[dict]:
    """Load pole OIDs, WGS84 coordinates, and heights in metres."""
    arcpy = _arcpy()
    wgs84 = arcpy.SpatialReference(4326)
    feet_to_metres = 0.3048

    fields_by_lower = {field.name.lower(): field.name for field in arcpy.ListFields(data_path)}
    actual_height_field = fields_by_lower.get(height_field.lower())
    if actual_height_field is None:
        print(
            f"  WARN: Height field '{height_field}' not found in {data_path}; "
            f"using default pole height {default_height_m:.1f} m for all poles."
        )

    fields = ["OID@", "SHAPE@XY"]
    if actual_height_field is not None:
        fields.append(actual_height_field)

    poles: list[dict] = []
    with arcpy.da.SearchCursor(data_path, fields, spatial_reference=wgs84) as cursor:
        for row in cursor:
            oid, xy = row[0], row[1]
            if xy is None:
                continue

            height_m = default_height_m
            if actual_height_field is not None:
                try:
                    height_ft = float(row[2])
                    if not math.isfinite(height_ft) or height_ft <= 0:
                        raise ValueError
                    height_m = height_ft * feet_to_metres
                except (TypeError, ValueError):
                    print(
                        f"  WARN: OID {oid}: {actual_height_field} value {row[2]!r} "
                        f"is not a positive height in feet; using {default_height_m:.1f} m."
                    )

            poles.append(
                {
                    "oid": oid,
                    "lon": xy[0],
                    "lat": xy[1],
                    "height_m": height_m,
                }
            )
    return poles


def _field_lookup(data_path: str) -> dict[str, Any]:
    return {field.name.lower(): field for field in _arcpy().ListFields(data_path)}


def _delete_fields_if_present(data_path: str, field_names: list[str]) -> None:
    arcpy = _arcpy()
    lookup = _field_lookup(data_path)
    actual_names = [lookup[name.lower()].name for name in field_names if name.lower() in lookup]
    if actual_names:
        arcpy.management.DeleteField(data_path, actual_names)


def _is_shapefile(data_path: str) -> bool:
    return str(data_path).lower().endswith(".shp")


def prepare_match_fields(
    data_path: str,
    oid_to_paths: dict[int, list[str]],
    oid_to_distances: dict[int, list[str]],
    image_field_name: str,
    distance_field_name: str,
    count_field_name: str = DEFAULT_COUNT_FIELD,
) -> bool:
    """Recreate current-run summary fields and return whether text fields fit."""
    arcpy = _arcpy()
    _delete_fields_if_present(
        data_path,
        [image_field_name, distance_field_name, count_field_name],
    )
    arcpy.management.AddField(data_path, count_field_name, "LONG")

    image_values = [";".join(values) for values in oid_to_paths.values()]
    distance_values = [";".join(values) for values in oid_to_distances.values()]
    image_length = max([1, *(len(value) for value in image_values)])
    distance_length = max([1, *(len(value) for value in distance_values)])

    if _is_shapefile(data_path) and max(image_length, distance_length) > SHAPEFILE_TEXT_LIMIT:
        print(
            "  WARN: Joined match values exceed the shapefile 254-character text limit; "
            "omitting image/distance summary fields. The normalized CSV is authoritative."
        )
        return False

    arcpy.management.AddField(
        data_path,
        image_field_name,
        "TEXT",
        field_length=image_length,
    )
    arcpy.management.AddField(
        data_path,
        distance_field_name,
        "TEXT",
        field_length=distance_length,
    )
    return True


def write_matches(
    data_path: str,
    oid_to_paths: dict[int, list[str]],
    oid_to_distances: dict[int, list[str]],
    field_name: str = "match_imgs",
    distance_field_name: str = "img_dists",
    count_field_name: str = DEFAULT_COUNT_FIELD,
    include_text_fields: bool = True,
) -> int:
    """Write current-run values for every pole, clearing all unmatched rows."""
    arcpy = _arcpy()
    cursor_fields = ["OID@", count_field_name]
    if include_text_fields:
        cursor_fields.extend([field_name, distance_field_name])

    updated = 0
    with arcpy.da.UpdateCursor(data_path, cursor_fields) as cursor:
        for row in cursor:
            oid = row[0]
            paths = oid_to_paths.get(oid, [])
            distances = oid_to_distances.get(oid, [])
            if len(paths) != len(distances):
                raise ValueError(
                    f"OID {oid} has {len(paths)} image(s) but {len(distances)} distance value(s)."
                )

            new_values: list[Any] = [oid, len(paths)]
            if include_text_fields:
                new_values.extend([";".join(paths), ";".join(distances)])

            if list(row) == new_values:
                continue
            for index, value in enumerate(new_values):
                row[index] = value
            cursor.updateRow(row)
            updated += 1
    return updated


def validate_match_output(
    data_path: str,
    oid_to_paths: dict[int, list[str]],
    oid_to_distances: dict[int, list[str]],
    field_name: str = "match_imgs",
    distance_field_name: str = "img_dists",
    count_field_name: str = DEFAULT_COUNT_FIELD,
    include_text_fields: bool = True,
) -> int:
    """Verify every staged GIS row contains the current run's values."""
    arcpy = _arcpy()
    fields = ["OID@", count_field_name]
    if include_text_fields:
        fields.extend([field_name, distance_field_name])

    row_count = 0
    with arcpy.da.SearchCursor(data_path, fields) as cursor:
        for row in cursor:
            row_count += 1
            oid = row[0]
            paths = oid_to_paths.get(oid, [])
            distances = oid_to_distances.get(oid, [])
            expected: list[Any] = [oid, len(paths)]
            if include_text_fields:
                expected.extend([";".join(paths), ";".join(distances)])
            actual = list(row)
            if include_text_fields:
                actual[2] = actual[2] or ""
                actual[3] = actual[3] or ""
            if actual != expected:
                raise RuntimeError(f"Staged GIS validation failed for pole OID {oid}.")
    return row_count


def read_pole_attributes(data_path: str) -> dict[str, dict[str, Any]]:
    """Read WGS84 pole coordinates and JSON-ready non-geometry attributes."""
    arcpy = _arcpy()
    wgs84 = arcpy.SpatialReference(4326)
    fields = list(arcpy.ListFields(data_path))
    attribute_fields = [field.name for field in fields if field.type not in {"Geometry", "OID"}]
    cursor_fields = ["OID@", "SHAPE@XY", *attribute_fields]
    result: dict[str, dict[str, Any]] = {}
    with arcpy.da.SearchCursor(
        data_path,
        cursor_fields,
        spatial_reference=wgs84,
    ) as cursor:
        for row in cursor:
            if row[1] is None:
                continue
            result[str(row[0])] = {
                "pole_oid": row[0],
                "pole_lon": row[1][0],
                "pole_lat": row[1][1],
                "attributes": dict(zip(attribute_fields, row[2:], strict=True)),
            }
    return result
