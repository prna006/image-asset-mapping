from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class CliValidationTests(unittest.TestCase):
    def test_output_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main.main(["--image_dir", "images", "--data_path", "source.shp"])
        self.assertEqual(raised.exception.code, 1)

    def test_source_and_output_must_differ(self):
        fake_arcgis = types.SimpleNamespace(dataset_exists=lambda _path: False)
        with patch.dict(sys.modules, {"src.arcgis_io": fake_arcgis}):
            with self.assertRaises(ValueError):
                main._validate_destinations(
                    "poles.shp",
                    "./poles.shp",
                    Path("matches.csv"),
                    overwrite=False,
                )

    def test_existing_output_requires_overwrite(self):
        fake_arcgis = types.SimpleNamespace(dataset_exists=lambda _path: True)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"src.arcgis_io": fake_arcgis}),
        ):
            matches = Path(directory) / "matches.csv"
            with self.assertRaises(FileExistsError):
                main._validate_destinations(
                    "source.shp",
                    "output.shp",
                    matches,
                    overwrite=False,
                )
            main._validate_destinations(
                "source.shp",
                "output.shp",
                matches,
                overwrite=True,
            )

    def test_check_inputs_does_not_require_or_create_output(self):
        with (
            patch.object(main, "check_environment", return_value=True),
            patch.object(main, "check_inputs", return_value=True) as check_inputs,
        ):
            result = main.main(
                ["--check-inputs", "--image_dir", "images", "--data_path", "source.shp"]
            )
        self.assertEqual(result, 0)
        check_inputs.assert_called_once()

    def test_check_inputs_stops_when_environment_is_invalid(self):
        with (
            patch.object(main, "check_environment", return_value=False),
            patch.object(main, "check_inputs") as check_inputs,
        ):
            result = main.main(
                ["--check-inputs", "--image_dir", "images", "--data_path", "source.shp"]
            )
        self.assertEqual(result, 1)
        check_inputs.assert_not_called()

    def test_input_check_validates_without_writing(self):
        fake_arcgis = types.SimpleNamespace(
            validate_pole_dataset=lambda _path: 1,
            load_poles=lambda *_args: [{"oid": 1, "lat": 37.0, "lon": -77.0, "height_m": 12.0}],
        )
        fake_camera = types.SimpleNamespace(
            extract_metadata=lambda _path: {"relative": 30},
            has_usable_relative_altitude=lambda _metadata: True,
            camera_from_metadata=lambda *_args: object(),
            load_dem_from_file=lambda _path: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").touch()
            before = sorted(root.iterdir())
            with patch.dict(
                sys.modules,
                {"src.arcgis_io": fake_arcgis, "src.camera": fake_camera},
            ):
                result = main.check_inputs(root, "poles.shp")
            after = sorted(root.iterdir())
        self.assertTrue(result)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
