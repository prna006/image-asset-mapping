"""Rules-based image-to-pole matching pipeline for ArcGIS Pro."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
import traceback
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_config
from src.results_io import (
    default_matches_csv_path,
    failure_csv_path,
    read_failure_csv,
    read_match_csv,
    stage_csv_path,
    write_failure_csv,
    write_match_csv,
)

DEFAULT_CONFIG = Path(__file__).with_name("pole_tagging.toml")
DEFAULT_VENDOR = "generic"
DEFAULT_IMG_FIELD = "match_imgs"
DEFAULT_DIST_FIELD = "img_dists"
DEFAULT_HEIGHT_FIELD = "HEIGHT"
DEFAULT_POLE_HEIGHT_M = 15.0
DEFAULT_DEBUG_KML = False
DEFAULT_DEBUG_OUTPUT = "debug_poles.kml"
DEFAULT_FALLBACK_ENABLED = False
DEFAULT_FALLBACK_MAX_CAMERA_TO_POLE_M = 20.0
DEFAULT_FALLBACK_MAX_SNAP_DISTANCE_M = 12.0
DEFAULT_FALLBACK_MIN_OVERLAP_IMAGES = 2
DEFAULT_FALLBACK_SINGLETON_CONFIDENCE_CAP = 0.45
DEFAULT_FALLBACK_REGION_RELAXATION_M = 0.0
DEFAULT_FALLBACK_WEIGHT_CLIP = 3.0
VALID_VENDORS = {"dji", "skydio", "generic"}
SUPPORTED_ARCGIS_VERSIONS = {(3, 4), (3, 5), (3, 6), (3, 7)}
PINNED_DEPENDENCIES = {
    "cv2": ("OpenCV", "opencv-python", "4.10.0.84"),
    "exiftool": ("pyexiftool", "pyexiftool", "0.5.6"),
    "geopy": ("Geopy", "geopy", "2.4.1"),
    "pygeomag": ("pygeomag", "pygeomag", "1.1.0"),
    "shapely": ("Shapely", "shapely", "2.1.2"),
    "tqdm": ("tqdm", "tqdm", "4.67.1"),
}
ESRI_MANAGED_DEPENDENCIES = {
    "numpy": "NumPy",
    "PIL": "Pillow",
    "scipy": "SciPy",
}


class PipelineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


class IncompleteRunError(RuntimeError):
    """Raised when strict mode encounters one or more failed images."""


@dataclass(frozen=True)
class PipelineResult:
    processed_images: int
    failed_images: int
    unmatched_images: int
    total_matches: int
    updated_poles: int
    output_path: str
    matches_csv_path: Path
    debug_kml_failed: bool = False

    @property
    def exit_code(self) -> int:
        return 2 if self.failed_images or self.debug_kml_failed else 0


def _resolve_option(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _same_dataset(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
        os.path.abspath(str(right))
    )


def _temporary_dataset_path(final_path: str, label: str) -> str:
    path = Path(final_path)
    token = uuid.uuid4().hex[:8]
    return str(path.with_name(f"_{path.stem}_{label}_{token}{path.suffix}"))


def _json_error(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expandvars(str(path))))


def _read_pyvenv_config(prefix: str | Path) -> dict[str, str]:
    config_path = Path(prefix) / "pyvenv.cfg"
    try:
        lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    config: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            config[key.strip().lower()] = value.strip()
    return config


def _classify_arcgis_runtime(install: dict[str, Any]) -> tuple[bool, str]:
    """Identify an initialized Conda runtime or a validated ArcGIS uv wrapper."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and _normalized_path(conda_prefix) == _normalized_path(sys.prefix):
        return True, "Conda OK"

    config = _read_pyvenv_config(sys.prefix)
    include_system = config.get("include-system-site-packages", "").lower() == "true"
    uv_wrapper = "uv" in config
    home = config.get("home")
    install_dir = install.get("InstallDir")
    home_matches_arcgis = False
    if home and install_dir:
        expected_home = Path(str(install_dir)) / "bin" / "Python" / "envs" / "arcgispro-py3"
        home_matches_arcgis = _normalized_path(home) == _normalized_path(expected_home)

    if uv_wrapper and include_system and home_matches_arcgis:
        return (
            True,
            "uv wrapper over ArcGIS Python OK WITH WARNING "
            "(recreate and retest after ArcGIS upgrades)",
        )

    if uv_wrapper:
        reasons = []
        if not include_system:
            reasons.append("system site packages are disabled")
        if not home_matches_arcgis:
            reasons.append("base interpreter does not match this ArcGIS installation")
        return False, f"uv wrapper UNSUPPORTED ({'; '.join(reasons)})"

    return False, "UNSUPPORTED (use an initialized ArcGIS Conda environment)"


def check_environment() -> bool:
    """Print the supported ArcGIS runtime checks and return whether all pass."""
    ok = True
    python_pair = sys.version_info[:2]
    python_ok = python_pair in {(3, 11), (3, 13)}
    print(f"Python        : {sys.version.split()[0]} {'OK' if python_ok else 'UNSUPPORTED'}")
    ok &= python_ok

    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arcpy = importlib.import_module("arcpy")
        install = arcpy.GetInstallInfo()
        version_text = str(install.get("Version", "unknown"))
        parts = version_text.split(".")
        version_pair = (int(parts[0]), int(parts[1]))
        version_ok = version_pair in SUPPORTED_ARCGIS_VERSIONS
        print(f"ArcGIS Pro    : {version_text} {'OK' if version_ok else 'UNSUPPORTED'}")
        ok &= version_ok
        if version_ok:
            expected_python = (3, 11) if version_pair in {(3, 4), (3, 5)} else (3, 13)
            pair_ok = python_pair == expected_python
            status = "OK" if pair_ok else f"EXPECTED {expected_python[0]}.{expected_python[1]}"
            print(
                f"ArcGIS/Python : {version_pair[0]}.{version_pair[1]} + "
                f"{python_pair[0]}.{python_pair[1]} {status}"
            )
            ok &= pair_ok

        spatial_reference = arcpy.SpatialReference(4326)
        if getattr(spatial_reference, "factoryCode", None) != 4326:
            raise RuntimeError("ArcPy could not construct WGS84 SpatialReference(4326).")
        runtime_ok, runtime_status = _classify_arcgis_runtime(install)
        print(f"ArcPy runtime : {runtime_status}")
        ok &= runtime_ok
    except Exception as exc:
        print(f"ArcGIS Pro    : FAIL ({exc})")
        ok = False

    for module_name, (label, distribution, expected_version) in PINNED_DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            version_ok = version == expected_version
            status = "OK" if version_ok else f"EXPECTED {expected_version}"
            print(f"{label:<14}: {version} {status}")
            ok &= version_ok
        except Exception as exc:
            print(f"{label:<14}: FAIL ({exc})")
            ok = False

    for module_name, label in ESRI_MANAGED_DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"{label:<14}: {version} OK (Esri managed)")
        except Exception as exc:
            print(f"{label:<14}: FAIL ({exc})")
            ok = False

    exiftool_path = os.environ.get("EXIFTOOL_PATH", "exiftool")
    resolved_exiftool = shutil.which(exiftool_path)
    if resolved_exiftool is None:
        print("ExifTool      : NOT FOUND")
        ok = False
    else:
        try:
            completed = subprocess.run(
                [resolved_exiftool, "-ver"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = completed.stdout.strip()
            if not version:
                raise RuntimeError("version output was empty")
            print(f"ExifTool      : {version} OK ({resolved_exiftool})")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            print(f"ExifTool      : FAIL ({exc})")
            ok = False
    return ok


def _image_paths(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"image_dir does not exist: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".jpg")
    if not images:
        raise ValueError(f"No .JPG files found in {directory}")
    return images


def _validate_output_parents(output_path: str, matches_csv_path: Path) -> None:
    raw_output = str(output_path)
    gdb_end = raw_output.lower().rfind(".gdb")
    if gdb_end >= 0:
        gdb_path = Path(raw_output[: gdb_end + 4]).expanduser().resolve()
        if not gdb_path.is_dir():
            raise ValueError(f"GIS output geodatabase does not exist: {gdb_path}")
        internal_parts = [
            part for part in raw_output[gdb_end + 4 :].replace("\\", "/").split("/") if part
        ]
        if len(internal_parts) > 1:
            from src.arcgis_io import dataset_exists

            feature_dataset = str(gdb_path.joinpath(*internal_parts[:-1]))
            if not dataset_exists(feature_dataset):
                raise ValueError(f"GIS output feature dataset does not exist: {feature_dataset}")
        gis_parent = gdb_path
    else:
        gis_parent = Path(output_path).expanduser().resolve().parent

    for label, parent in (
        ("GIS output", gis_parent),
        ("match CSV", matches_csv_path.expanduser().resolve().parent),
    ):
        if not parent.is_dir():
            raise ValueError(f"{label} parent directory does not exist: {parent}")
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"{label} parent directory is not writable: {parent}")


def _validate_destinations(
    data_path: str,
    output_path: str,
    matches_csv_path: Path,
    overwrite: bool,
) -> None:
    from src.arcgis_io import dataset_exists

    if _same_dataset(data_path, output_path):
        raise ValueError(
            "--output must be different from --data_path; source data is never modified."
        )
    _validate_output_parents(output_path, matches_csv_path)
    if dataset_exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output dataset already exists: {output_path}. Use --overwrite to replace it."
        )

    failures_path = failure_csv_path(matches_csv_path)
    for csv_path in (matches_csv_path, failures_path):
        if csv_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {csv_path}. Use --overwrite to replace it."
            )


def _commit_outputs(
    staged_dataset: str,
    output_path: str,
    staged_matches_csv: Path,
    matches_csv_path: Path,
    staged_failures_csv: Path | None,
    failures_csv_path: Path,
) -> None:
    """Promote staged outputs and restore prior destinations if promotion fails."""
    from src.arcgis_io import copy_features, dataset_exists, delete_dataset

    dataset_backup = _temporary_dataset_path(output_path, "backup")
    had_dataset = dataset_exists(output_path)
    match_backup = stage_csv_path(matches_csv_path) if matches_csv_path.exists() else None
    failure_backup = stage_csv_path(failures_csv_path) if failures_csv_path.exists() else None

    # stage_csv_path creates the file to reserve a collision-free name. A
    # backup must not appear to exist until the original is actually moved;
    # otherwise an earlier failure could restore an empty placeholder.
    if match_backup is not None:
        match_backup.unlink(missing_ok=True)
    if failure_backup is not None:
        failure_backup.unlink(missing_ok=True)

    dataset_promotion_started = False
    matches_promotion_started = False
    failures_promotion_started = False
    cleanup_backups = True
    try:
        if had_dataset:
            copy_features(output_path, dataset_backup)
        if match_backup is not None:
            os.replace(matches_csv_path, match_backup)
        if failure_backup is not None:
            os.replace(failures_csv_path, failure_backup)

        dataset_promotion_started = True
        copy_features(staged_dataset, output_path)
        matches_promotion_started = True
        os.replace(staged_matches_csv, matches_csv_path)
        if staged_failures_csv is not None:
            failures_promotion_started = True
            os.replace(staged_failures_csv, failures_csv_path)
        else:
            failures_csv_path.unlink(missing_ok=True)
    except Exception:
        try:
            if dataset_promotion_started:
                if had_dataset and dataset_exists(dataset_backup):
                    copy_features(dataset_backup, output_path)
                elif dataset_exists(output_path):
                    delete_dataset(output_path)

            if match_backup is not None and match_backup.exists():
                matches_csv_path.unlink(missing_ok=True)
                os.replace(match_backup, matches_csv_path)
            elif match_backup is None and matches_promotion_started:
                matches_csv_path.unlink(missing_ok=True)
            if failure_backup is not None and failure_backup.exists():
                failures_csv_path.unlink(missing_ok=True)
                os.replace(failure_backup, failures_csv_path)
            elif failure_backup is None and failures_promotion_started:
                failures_csv_path.unlink(missing_ok=True)
        except Exception as rollback_error:
            cleanup_backups = False
            raise RuntimeError(
                "Output promotion and rollback both failed. Recovery backups "
                f"were retained at GIS={dataset_backup!r}, "
                f"matches={str(match_backup)!r}, failures={str(failure_backup)!r}."
            ) from rollback_error
        raise
    finally:
        delete_dataset(staged_dataset)
        if cleanup_backups and dataset_exists(dataset_backup):
            delete_dataset(dataset_backup)
        staged_matches_csv.unlink(missing_ok=True)
        if staged_failures_csv is not None:
            staged_failures_csv.unlink(missing_ok=True)
        if cleanup_backups and match_backup is not None:
            match_backup.unlink(missing_ok=True)
        if cleanup_backups and failure_backup is not None:
            failure_backup.unlink(missing_ok=True)


def _write_outputs(
    staged_dataset: str,
    output_path: str,
    matches_csv_path: Path,
    match_rows: list[dict],
    failures: list[dict],
    oid_to_paths: dict[int, list[str]],
    oid_to_distances: dict[int, list[str]],
    img_field: str,
    dist_field: str,
) -> int:
    from src.arcgis_io import (
        delete_dataset,
        prepare_match_fields,
        validate_match_output,
        write_matches,
    )

    staged_matches = stage_csv_path(matches_csv_path)
    failures_path = failure_csv_path(matches_csv_path)
    staged_failures = stage_csv_path(failures_path) if failures else None

    try:
        include_text = prepare_match_fields(
            staged_dataset,
            oid_to_paths,
            oid_to_distances,
            img_field,
            dist_field,
        )
        updated = write_matches(
            staged_dataset,
            oid_to_paths,
            oid_to_distances,
            img_field,
            dist_field,
            include_text_fields=include_text,
        )
        write_match_csv(staged_matches, match_rows)
        if len(read_match_csv(staged_matches)) != len(match_rows):
            raise RuntimeError("Staged match CSV validation failed.")
        if staged_failures is not None:
            write_failure_csv(staged_failures, failures)
            if len(read_failure_csv(staged_failures)) != len(failures):
                raise RuntimeError("Staged failure CSV validation failed.")
        validate_match_output(
            staged_dataset,
            oid_to_paths,
            oid_to_distances,
            img_field,
            dist_field,
            include_text_fields=include_text,
        )
        _commit_outputs(
            staged_dataset,
            output_path,
            staged_matches,
            matches_csv_path,
            staged_failures,
            failures_path,
        )
        return updated
    except Exception:
        delete_dataset(staged_dataset)
        staged_matches.unlink(missing_ok=True)
        if staged_failures is not None:
            staged_failures.unlink(missing_ok=True)
        raise


def _run_pipeline_with_stage(
    staged_dataset: str,
    image_dir: str | Path,
    vendor: str,
    data_path: str,
    output_path: str,
    matches_csv_path: str | Path,
    img_field: str,
    dist_field: str,
    height_field: str,
    overwrite: bool = False,
    require_complete: bool = False,
    verbose: bool = False,
    default_pole_height_m: float = DEFAULT_POLE_HEIGHT_M,
    dem_path: str | None = None,
    debug_kml: bool = DEFAULT_DEBUG_KML,
    debug_output_path: str | None = None,
    fallback_config: Any = None,
) -> PipelineResult:
    from tqdm import tqdm

    from src.arcgis_io import dataset_exists, load_poles
    from src.camera import load_camera, load_dem_from_file, prefetch_dem_for_images
    from src.debug_kml_export import (
        build_debug_poles,
        build_footprint_records,
        build_image_record,
        build_kml,
        write_kml,
    )
    from src.fallback import resolve_fallback_matches
    from src.frustum import get_footprint, ground_distance_m
    from src.matcher import find_poles_in_frustum

    image_dir = Path(image_dir).resolve()
    matches_csv_path = Path(matches_csv_path).resolve()
    images = _image_paths(image_dir)
    if not dataset_exists(data_path):
        raise ValueError(f"Pole dataset does not exist: {data_path}")
    _validate_destinations(data_path, output_path, matches_csv_path, overwrite)

    print(f"Found {len(images)} image(s) in {image_dir}")
    print(f"Loading poles from staged output {staged_dataset} ...")
    poles = load_poles(staged_dataset, height_field, default_pole_height_m)
    poles_by_oid = {pole["oid"]: pole for pole in poles}
    print(f"  Loaded {len(poles)} pole(s).")

    if dem_path:
        load_dem_from_file(dem_path)
    else:
        prefetch_dem_for_images(images, vendor)

    oid_to_matches: dict[int, list[tuple[str, float, str]]] = defaultdict(list)
    debug_images: list[dict] = []
    debug_images_by_name: dict[str, dict] = {}
    debug_footprints: list[dict] = []
    unmatched_images: list[dict] = []
    failures: list[dict] = []
    n_primary_unmatched = 0

    with tqdm(
        images,
        desc="Processing images",
        unit="image",
        dynamic_ncols=True,
        disable=None,
    ) as image_progress:
        for index, img_path in enumerate(image_progress, start=1):
            try:
                camera = load_camera(img_path, vendor=vendor)
                footprint = get_footprint(camera)
                matched = find_poles_in_frustum(poles, footprint, camera)

                cam_lat, cam_lon = camera.gps_coords
                for oid in matched:
                    pole = poles_by_oid[oid]
                    distance_m = ground_distance_m(
                        cam_lat,
                        cam_lon,
                        pole["lat"],
                        pole["lon"],
                    )
                    oid_to_matches[oid].append((img_path.name, distance_m, "primary"))

                if debug_kml:
                    record = build_image_record(img_path, camera)
                    record["match_source"] = "primary" if matched else "unmatched"
                    record["matched_pole_count"] = len(matched)
                    debug_images.append(record)
                    debug_images_by_name[img_path.name] = record
                    debug_footprints.extend(
                        build_footprint_records(
                            img_path,
                            footprint,
                            camera.pitch_deg,
                        )
                    )

                if fallback_config is not None and fallback_config.enabled and not matched:
                    unmatched_images.append(
                        {
                            "image_name": img_path.name,
                            "camera": camera,
                            "footprint": footprint,
                        }
                    )
                if not matched:
                    n_primary_unmatched += 1
                image_progress.set_postfix_str(
                    f"{img_path.name}: {len(matched)} matched",
                    refresh=False,
                )
                if verbose:
                    image_progress.write(
                        f"  [{index}/{len(images)}] {img_path.name} -> "
                        f"{len(matched)} pole(s) matched"
                    )
            except Exception as exc:
                failure = {"image_name": img_path.name, **_json_error(exc)}
                failures.append(failure)
                image_progress.write(
                    f"  WARN: {img_path.name} failed -- {exc}",
                    file=sys.stderr,
                )
                if verbose:
                    traceback.print_exc()

    if failures and require_complete:
        raise IncompleteRunError(
            f"{len(failures)} image(s) failed and --require-complete was requested."
        )
    if failures and len(failures) == len(images):
        raise IncompleteRunError("All images failed; no usable partial output was produced.")

    fallback_estimates: list[dict] = []
    fallback_regions: list[dict] = []
    fallback_matches = 0
    if fallback_config is not None and fallback_config.enabled and unmatched_images:
        assignments, support_regions, estimate_regions = resolve_fallback_matches(
            unmatched_images,
            poles,
            fallback_config,
        )
        fallback_estimates = assignments
        fallback_regions = [*support_regions, *estimate_regions]
        camera_by_image = {image["image_name"]: image["camera"] for image in unmatched_images}
        for assignment in assignments:
            pole_oid = assignment["pole_oid"]
            for image_name in assignment["image_names"]:
                record = debug_images_by_name.get(image_name)
                if record is not None:
                    record["match_source"] = (
                        "fallback" if pole_oid is not None else "fallback-unresolved"
                    )
                    record["fallback_confidence"] = assignment["confidence"]
                    record["fallback_pole_oid"] = pole_oid
                    record["fallback_mode"] = assignment["mode"]
                    record["fallback_reason"] = assignment["reason"]
                if pole_oid is None:
                    continue
                camera = camera_by_image[image_name]
                pole = poles_by_oid[pole_oid]
                distance_m = ground_distance_m(
                    camera.gps_coords[0],
                    camera.gps_coords[1],
                    pole["lat"],
                    pole["lon"],
                )
                oid_to_matches[pole_oid].append((image_name, distance_m, "fallback"))
                fallback_matches += 1

    sorted_matches = {
        oid: sorted(values, key=lambda item: (item[1], item[0], item[2]))
        for oid, values in oid_to_matches.items()
    }
    oid_to_paths = {
        oid: [image_name for image_name, _, _ in values] for oid, values in sorted_matches.items()
    }
    oid_to_distances = {
        oid: [f"{distance:.2f}" for _, distance, _ in values]
        for oid, values in sorted_matches.items()
    }
    match_rows = [
        {
            "pole_oid": oid,
            "pole_lat": poles_by_oid[oid]["lat"],
            "pole_lon": poles_by_oid[oid]["lon"],
            "image_name": image_name,
            "distance_m": distance,
            "match_source": source,
        }
        for oid, values in sorted_matches.items()
        for image_name, distance, source in values
    ]

    updated = _write_outputs(
        staged_dataset,
        output_path,
        matches_csv_path,
        match_rows,
        failures,
        oid_to_paths,
        oid_to_distances,
        img_field,
        dist_field,
    )

    debug_kml_failed = False
    if debug_kml:
        try:
            debug_poles = build_debug_poles(poles, oid_to_paths)
            debug_doc = build_kml(
                debug_poles,
                debug_images,
                debug_footprints,
                fallback_estimates,
                fallback_regions,
                img_field,
                image_dir,
            )
            write_kml(debug_doc, Path(debug_output_path or DEFAULT_DEBUG_OUTPUT))
        except Exception as exc:
            debug_kml_failed = True
            print(
                f"  WARN: GIS and CSV outputs were committed, but debug KML failed -- {exc}",
                file=sys.stderr,
            )
            if verbose:
                traceback.print_exc()

    n_processed = len(images) - len(failures)
    n_unmatched = n_primary_unmatched - fallback_matches
    total_matches = len(match_rows)
    print("\n--- Pipeline complete ---")
    print(f"  Images processed : {n_processed}")
    print(f"  Images failed    : {len(failures)}")
    print(f"  Images unmatched : {n_unmatched}")
    print(f"  Total matches    : {total_matches}")
    print(f"  Poles updated    : {updated}")
    print(f"  Output           : {output_path}")
    print(f"  Match CSV        : {matches_csv_path}")
    if failures:
        print(f"  Failure CSV      : {failure_csv_path(matches_csv_path)}")
    if debug_kml_failed:
        print("  Debug KML        : FAILED (primary outputs are usable)")

    return PipelineResult(
        processed_images=n_processed,
        failed_images=len(failures),
        unmatched_images=n_unmatched,
        total_matches=total_matches,
        updated_poles=updated,
        output_path=output_path,
        matches_csv_path=matches_csv_path,
        debug_kml_failed=debug_kml_failed,
    )


def run_pipeline(
    image_dir: str | Path,
    vendor: str,
    data_path: str,
    output_path: str,
    matches_csv_path: str | Path,
    img_field: str,
    dist_field: str,
    height_field: str,
    overwrite: bool = False,
    require_complete: bool = False,
    verbose: bool = False,
    default_pole_height_m: float = DEFAULT_POLE_HEIGHT_M,
    dem_path: str | None = None,
    debug_kml: bool = DEFAULT_DEBUG_KML,
    debug_output_path: str | None = None,
    fallback_config: Any = None,
) -> PipelineResult:
    """Run against a disposable copy so source OIDs and data remain untouched."""
    from src.arcgis_io import copy_features, dataset_exists, delete_dataset, validate_pole_dataset

    resolved_matches_csv = Path(matches_csv_path).resolve()
    if not dataset_exists(data_path):
        raise ValueError(f"Pole dataset does not exist: {data_path}")
    validate_pole_dataset(data_path)
    _validate_destinations(data_path, output_path, resolved_matches_csv, overwrite)

    staged_dataset = _temporary_dataset_path(output_path, "stage")
    copy_features(data_path, staged_dataset)
    try:
        return _run_pipeline_with_stage(
            staged_dataset=staged_dataset,
            image_dir=image_dir,
            vendor=vendor,
            data_path=data_path,
            output_path=output_path,
            matches_csv_path=resolved_matches_csv,
            img_field=img_field,
            dist_field=dist_field,
            height_field=height_field,
            overwrite=overwrite,
            require_complete=require_complete,
            verbose=verbose,
            default_pole_height_m=default_pole_height_m,
            dem_path=dem_path,
            debug_kml=debug_kml,
            debug_output_path=debug_output_path,
            fallback_config=fallback_config,
        )
    finally:
        delete_dataset(staged_dataset)


def check_inputs(
    image_dir: str | Path,
    data_path: str,
    height_field: str = DEFAULT_HEIGHT_FIELD,
    default_pole_height_m: float = DEFAULT_POLE_HEIGHT_M,
    dem_path: str | None = None,
) -> bool:
    """Validate inputs without creating or modifying any outputs."""
    from src.arcgis_io import load_poles, validate_pole_dataset
    from src.camera import (
        camera_from_metadata,
        extract_metadata,
        has_usable_relative_altitude,
        load_dem_from_file,
    )

    try:
        images = _image_paths(image_dir)
        pole_count = validate_pole_dataset(data_path)
        # Exercise height conversion and warnings without writing to the dataset.
        load_poles(data_path, height_field, default_pole_height_m)
        if dem_path:
            load_dem_from_file(dem_path)

        errors: list[tuple[str, Exception]] = []
        all_have_relative_altitude = True
        for image in images:
            try:
                metadata = extract_metadata(image)
                all_have_relative_altitude &= has_usable_relative_altitude(metadata)
                camera_from_metadata(metadata, image)
            except Exception as exc:
                errors.append((image.name, exc))
        if errors:
            for name, exc in errors:
                print(f"Input image FAIL: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"Input check failed: {len(errors)} of {len(images)} image(s) are invalid.")
            return False

        if not dem_path and all_have_relative_altitude:
            print(
                "  WARN: Every image has relative altitude; USGS 3DEP connectivity "
                "was not required or tested."
            )
        terrain = f"local DEM {dem_path}" if dem_path else "relative altitude / USGS 3DEP fallback"
        print("\n--- Input check complete ---")
        print(f"  Images validated : {len(images)}")
        print(f"  Poles validated  : {pole_count}")
        print(f"  Terrain source   : {terrain}")
        print("  Outputs created  : none")
        return True
    except Exception as exc:
        print(f"Input check FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = PipelineArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a TOML config file.")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check the ArcGIS runtime and exit.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate the environment and all inputs without creating outputs.",
    )
    parser.add_argument("--image_dir", help="Directory containing .JPG drone images.")
    parser.add_argument(
        "--vendor",
        choices=sorted(VALID_VENDORS),
        help="Informational vendor hint.",
    )
    parser.add_argument("--data_path", help="Source pole shapefile or feature class.")
    parser.add_argument(
        "--output",
        help="Required derived output dataset; the source is never modified.",
    )
    parser.add_argument("--matches_csv", help="Authoritative normalized match CSV path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived outputs.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write nothing if any image fails; partial output is the default.",
    )
    parser.add_argument(
        "--img_field",
        help=f"Optional joined image field (default: {DEFAULT_IMG_FIELD}).",
    )
    parser.add_argument(
        "--dist_field",
        help=f"Optional joined distance field (default: {DEFAULT_DIST_FIELD}).",
    )
    parser.add_argument(
        "--height_field",
        help=f"Pole height field in feet (default: {DEFAULT_HEIGHT_FIELD}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image details and tracebacks in addition to progress bars.",
    )
    parser.add_argument("--dem", metavar="PATH", help="Local WGS84 DEM; skips USGS download.")
    parser.add_argument("--pole_height", type=float, help="Default pole height in metres.")
    parser.add_argument("--debug_kml", action="store_true", default=None)
    parser.add_argument("--debug_output")
    parser.add_argument("--fallback", action="store_true", default=None)
    parser.add_argument("--fallback_max_camera_to_pole_m", type=float)
    parser.add_argument("--fallback_max_snap_distance_m", type=float)
    parser.add_argument("--fallback_min_overlap_images", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.check_env:
        return 0 if check_environment() else 1

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if args.config and not config_path.exists():
        parser.error(f"Config file does not exist: {config_path}")
    try:
        config = load_config(config_path if config_path.exists() else None)
    except (OSError, ConfigError) as exc:
        parser.error(f"Could not load config '{config_path}': {exc}")

    if not args.image_dir:
        parser.error("--image_dir is required.")
    data_path = _resolve_option(args.data_path, config.data_path)
    if not data_path:
        parser.error("--data_path or defaults.data_path is required.")
    output_path = _resolve_option(args.output, config.output_path)
    if not output_path and not args.check_inputs:
        parser.error("--output or defaults.output_path is required; source data is never modified.")
    vendor = _resolve_option(args.vendor, config.vendor, DEFAULT_VENDOR)
    img_field = _resolve_option(args.img_field, config.img_field, DEFAULT_IMG_FIELD)
    dist_field = _resolve_option(args.dist_field, config.dist_field, DEFAULT_DIST_FIELD)
    height_field = _resolve_option(args.height_field, config.height_field, DEFAULT_HEIGHT_FIELD)
    if len({img_field.lower(), dist_field.lower(), "match_cnt"}) != 3:
        parser.error("img_field, dist_field, and match_cnt must have distinct names.")
    default_pole_height_m = _resolve_option(
        args.pole_height,
        config.default_pole_height_m,
        DEFAULT_POLE_HEIGHT_M,
    )
    if not math.isfinite(default_pole_height_m) or default_pole_height_m <= 0:
        parser.error("default pole height must be a positive finite number of metres.")
    dem_path = _resolve_option(args.dem, config.dem_path)

    if args.check_inputs:
        if not check_environment():
            return 1
        return (
            0
            if check_inputs(
                args.image_dir,
                data_path,
                height_field=height_field,
                default_pole_height_m=default_pole_height_m,
                dem_path=dem_path,
            )
            else 1
        )

    matches_csv_path = Path(
        _resolve_option(
            args.matches_csv,
            config.matches_csv_path,
            default_matches_csv_path(output_path),
        )
    )

    try:
        from src.fallback import FallbackConfig, validate_fallback_config
        from src.frustum import configure_projection

        fallback_config = FallbackConfig(
            enabled=_resolve_option(
                args.fallback,
                config.fallback_enabled,
                DEFAULT_FALLBACK_ENABLED,
            ),
            max_camera_to_pole_m=_resolve_option(
                args.fallback_max_camera_to_pole_m,
                config.fallback_max_camera_to_pole_m,
                DEFAULT_FALLBACK_MAX_CAMERA_TO_POLE_M,
            ),
            max_snap_distance_m=_resolve_option(
                args.fallback_max_snap_distance_m,
                config.fallback_max_snap_distance_m,
                DEFAULT_FALLBACK_MAX_SNAP_DISTANCE_M,
            ),
            min_overlap_images=_resolve_option(
                args.fallback_min_overlap_images,
                config.fallback_min_overlap_images,
                DEFAULT_FALLBACK_MIN_OVERLAP_IMAGES,
            ),
            singleton_confidence_cap=_resolve_option(
                config.fallback_singleton_confidence_cap,
                DEFAULT_FALLBACK_SINGLETON_CONFIDENCE_CAP,
            ),
            region_relaxation_m=_resolve_option(
                config.fallback_region_relaxation_m,
                DEFAULT_FALLBACK_REGION_RELAXATION_M,
            ),
            weight_clip=_resolve_option(config.fallback_weight_clip, DEFAULT_FALLBACK_WEIGHT_CLIP),
        )
        configure_projection(config.max_range_m, config.nadir_threshold_deg, config.margin_px)
        validate_fallback_config(fallback_config)
    except (ImportError, ValueError) as exc:
        print(f"ERROR: invalid runtime or configuration: {exc}", file=sys.stderr)
        return 1

    if not check_environment():
        return 1

    try:
        result = run_pipeline(
            image_dir=args.image_dir,
            vendor=vendor,
            data_path=data_path,
            output_path=output_path,
            matches_csv_path=matches_csv_path,
            img_field=img_field,
            dist_field=dist_field,
            height_field=height_field,
            overwrite=args.overwrite,
            require_complete=args.require_complete,
            verbose=args.verbose,
            default_pole_height_m=default_pole_height_m,
            dem_path=dem_path,
            debug_kml=_resolve_option(args.debug_kml, config.debug_kml, DEFAULT_DEBUG_KML),
            debug_output_path=_resolve_option(args.debug_output, config.debug_output_path),
            fallback_config=fallback_config,
        )
        return result.exit_code
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
