"""Materialize and run the deterministic ArcGIS/ExifTool smoke fixture."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import arcpy
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _run(command: list[str], expected_codes: tuple[int, ...] = (0,)) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode not in expected_codes:
        raise RuntimeError(
            f"Command exited {completed.returncode}, expected {sorted(expected_codes)}: "
            + " ".join(command)
        )


def _create_fixture(work: Path, definition: dict) -> tuple[Path, Path, Path]:
    images = work / "images"
    images.mkdir()
    camera = definition["camera"]
    image_path = images / camera["image_name"]
    Image.new("RGB", (400, 300), color=(96, 128, 160)).save(image_path, "JPEG")

    exiftool = os.environ.get("EXIFTOOL_PATH", "exiftool")
    _run(
        [
            exiftool,
            "-config",
            str(FIXTURE_DIR / "skydio.config"),
            "-overwrite_original",
            "-EXIF:Make=Skydio",
            f"-EXIF:GPSLatitude={camera['latitude']}",
            "-EXIF:GPSLatitudeRef=N",
            f"-EXIF:GPSLongitude={abs(camera['longitude'])}",
            "-EXIF:GPSLongitudeRef=W",
            f"-EXIF:GPSAltitude={camera['absolute_altitude_m']}",
            f"-EXIF:FocalLength={camera['focal_length_mm']}",
            f"-EXIF:FocalLengthIn35mmFormat={camera['focal_length_35mm']}",
            "-XMP-skydio:CalibratedFocalLengthX=250",
            "-XMP-skydio:CalibratedFocalLengthY=250",
            "-XMP-skydio:CalibratedOpticalCenterX=200",
            "-XMP-skydio:CalibratedOpticalCenterY=150",
            f"-XMP-skydio:CameraOrientationNEDPitch={camera['pitch_deg']}",
            f"-XMP-skydio:CameraOrientationNEDRoll={camera['roll_deg']}",
            f"-XMP-skydio:CameraOrientationNEDYaw={camera['yaw_deg']}",
            f"-XMP-skydio:CameraPositionFLUZ=0,0,{camera['relative_altitude_m']}",
            f"-XMP-skydio:GpsMslHeight={camera['absolute_altitude_m']}",
            str(image_path),
        ]
    )

    poles = work / "poles.shp"
    arcpy.management.CreateFeatureclass(work, poles.name, "POINT", spatial_reference=4326)
    arcpy.management.AddField(poles, "HEIGHT", "DOUBLE")
    pole = definition["pole"]
    with arcpy.da.InsertCursor(poles, ["SHAPE@XY", "HEIGHT"]) as cursor:
        cursor.insertRow(((pole["longitude"], pole["latitude"]), pole["height_ft"]))

    dem_definition = definition["dem"]
    dem = work / "flat_dem.tif"
    size = int(dem_definition["size"])
    cell = float(dem_definition["cell_size_deg"])
    values = np.full((size, size), dem_definition["elevation_m"], dtype=np.float32)
    lower_left = arcpy.Point(
        camera["longitude"] - size * cell / 2,
        camera["latitude"] - size * cell / 2,
    )
    arcpy.NumPyArrayToRaster(values, lower_left, cell, cell).save(str(dem))
    arcpy.management.DefineProjection(dem, arcpy.SpatialReference(4326))
    return images, poles, dem


def _assert_results(output: Path, matches: Path) -> None:
    expected_path = FIXTURE_DIR / "expected_matches.csv"
    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    with matches.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    if actual != expected:
        raise AssertionError(f"Match CSV mismatch. Expected {expected!r}; got {actual!r}.")

    fields = {field.name.lower() for field in arcpy.ListFields(output)}
    if not {"match_cnt", "match_imgs", "img_dists"}.issubset(fields):
        raise AssertionError(f"Output fields are incomplete: {sorted(fields)}")
    with arcpy.da.SearchCursor(output, ["match_cnt", "match_imgs", "img_dists"]) as cursor:
        row = next(iter(cursor))
    if row[0] != 1 or row[1] != "synthetic_skydio.jpg" or row[2] != "0.00":
        raise AssertionError(f"Unexpected GIS match values: {row!r}")


def main() -> int:
    definition = json.loads((FIXTURE_DIR / "fixture.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pole-tagging-smoke-") as directory:
        work = Path(directory)
        images, poles, dem = _create_fixture(work, definition)
        source_files = [path for path in work.iterdir() if path.stem == poles.stem]
        before = _hash_files(source_files)
        output = work / "poles_tagged.shp"
        matches = work / "poles_tagged_matches.csv"

        common = [
            "--image_dir",
            str(images),
            "--data_path",
            str(poles),
            "--dem",
            str(dem),
        ]
        _run([sys.executable, "main.py", "--check-inputs", *common])
        _run(
            [
                sys.executable,
                "main.py",
                *common,
                "--output",
                str(output),
                "--matches_csv",
                str(matches),
                "--require-complete",
            ]
        )
        _assert_results(output, matches)
        after = _hash_files(source_files)
        if before != after:
            raise AssertionError("The source pole fixture changed during the pipeline run.")

    print(
        "Smoke test PASS: preflight, offline pipeline, outputs, and source immutability verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
