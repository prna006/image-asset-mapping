from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class Camera:
    gps_coords = (37.0, -77.0)
    pitch_deg = -90.0


class MetadataValidationError(ValueError):
    pass


class FakeProgress:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix_str(self, *_args, **_kwargs):
        return None

    def write(self, *_args, **_kwargs):
        return None


class PipelineFailureTests(unittest.TestCase):
    def _modules(self, kml_build_error=None, kml_write_error=None):
        def load_camera(path, vendor):
            del vendor
            if path.name.startswith("bad"):
                raise MetadataValidationError("missing metadata")
            return Camera()

        def build_kml(*_args):
            if kml_build_error is not None:
                raise kml_build_error
            return None

        def write_kml(*_args):
            if kml_write_error is not None:
                raise kml_write_error

        return {
            "src.arcgis_io": types.SimpleNamespace(
                dataset_exists=lambda _path: True,
                load_poles=lambda *_args: [{"oid": 1, "lat": 37.0, "lon": -77.0, "height_m": 15.0}],
            ),
            "src.camera": types.SimpleNamespace(
                load_camera=load_camera,
                load_dem_from_file=lambda _path: None,
                prefetch_dem_for_images=lambda *_args: None,
            ),
            "src.debug_kml_export": types.SimpleNamespace(
                build_debug_poles=lambda *_args: [],
                build_footprint_records=lambda *_args: [],
                build_image_record=lambda *_args: {},
                build_kml=build_kml,
                write_kml=write_kml,
            ),
            "src.fallback": types.SimpleNamespace(
                resolve_fallback_matches=lambda *_args: ([], [], [])
            ),
            "src.frustum": types.SimpleNamespace(
                get_footprint=lambda _camera: object(),
                ground_distance_m=lambda *_args: 4.25,
            ),
            "src.matcher": types.SimpleNamespace(find_poles_in_frustum=lambda *_args: [1]),
            "tqdm": types.SimpleNamespace(tqdm=FakeProgress),
        }

    def _run(self, directory: str, require_complete: bool, debug_kml: bool = False):
        return main._run_pipeline_with_stage(
            staged_dataset="stage.shp",
            image_dir=directory,
            vendor="generic",
            data_path="source.shp",
            output_path="output.shp",
            matches_csv_path=Path(directory) / "matches.csv",
            img_field="match_imgs",
            dist_field="img_dists",
            height_field="HEIGHT",
            require_complete=require_complete,
            debug_kml=debug_kml,
        )

    def test_partial_is_written_and_returns_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "good.jpg").touch()
            Path(directory, "bad.jpg").touch()
            with (
                patch.dict(sys.modules, self._modules()),
                patch.object(main, "_validate_destinations"),
                patch.object(main, "_write_outputs", return_value=1) as write_outputs,
            ):
                result = self._run(directory, require_complete=False)

        self.assertEqual(result.exit_code, 2)
        failures = write_outputs.call_args.args[4]
        self.assertEqual(failures[0]["image_name"], "bad.jpg")
        self.assertEqual(failures[0]["error_type"], "MetadataValidationError")

    def test_strict_mode_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "good.jpg").touch()
            Path(directory, "bad.jpg").touch()
            with (
                patch.dict(sys.modules, self._modules()),
                patch.object(main, "_validate_destinations"),
                patch.object(main, "_write_outputs") as write_outputs,
            ):
                with self.assertRaises(main.IncompleteRunError):
                    self._run(directory, require_complete=True)
        write_outputs.assert_not_called()

    def test_all_failed_is_fatal_even_in_partial_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "bad.jpg").touch()
            with (
                patch.dict(sys.modules, self._modules()),
                patch.object(main, "_validate_destinations"),
                patch.object(main, "_write_outputs") as write_outputs,
            ):
                with self.assertRaises(main.IncompleteRunError):
                    self._run(directory, require_complete=False)
        write_outputs.assert_not_called()

    def test_debug_kml_failure_keeps_outputs_and_returns_exit_two(self):
        error_cases = (
            {"kml_build_error": RuntimeError("KML build failed")},
            {"kml_write_error": OSError("KML write failed")},
        )
        for errors in error_cases:
            with self.subTest(errors=errors), tempfile.TemporaryDirectory() as directory:
                Path(directory, "good.jpg").touch()
                with (
                    patch.dict(sys.modules, self._modules(**errors)),
                    patch.object(main, "_validate_destinations"),
                    patch.object(main, "_write_outputs", return_value=1) as write_outputs,
                ):
                    result = self._run(
                        directory,
                        require_complete=True,
                        debug_kml=True,
                    )

                self.assertEqual(result.exit_code, 2)
                self.assertTrue(result.debug_kml_failed)
                self.assertEqual(write_outputs.call_args.args[4], [])

    def test_successful_debug_kml_keeps_complete_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "good.jpg").touch()
            with (
                patch.dict(sys.modules, self._modules()),
                patch.object(main, "_validate_destinations"),
                patch.object(main, "_write_outputs", return_value=1),
            ):
                result = self._run(
                    directory,
                    require_complete=True,
                    debug_kml=True,
                )

        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.debug_kml_failed)


if __name__ == "__main__":
    unittest.main()
