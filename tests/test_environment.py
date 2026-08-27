from __future__ import annotations

import contextlib
import io
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class EnvironmentTests(unittest.TestCase):
    def _arcpy(self, version="3.7.0", install_dir="C:/ArcGIS/Pro", spatial_code=4326):
        return types.SimpleNamespace(
            GetInstallInfo=lambda: {
                "Version": version,
                "InstallDir": install_dir,
            },
            SpatialReference=lambda _code: types.SimpleNamespace(factoryCode=spatial_code),
        )

    def _run_check(
        self,
        arcpy,
        prefix,
        environment,
        versions=None,
        exiftool_error=None,
        python_pair=(3, 13),
    ):
        versions = versions or {
            distribution: expected
            for _, distribution, expected in main.PINNED_DEPENDENCIES.values()
        }

        def import_module(name):
            if name == "arcpy":
                return arcpy
            return types.SimpleNamespace(__version__="managed")

        output = io.StringIO()
        with (
            patch.object(main.importlib, "import_module", side_effect=import_module),
            patch.object(main.importlib.metadata, "version", side_effect=versions.__getitem__),
            patch.object(main.shutil, "which", return_value="C:/tools/exiftool.exe"),
            (
                patch.object(
                    main.subprocess,
                    "run",
                    side_effect=exiftool_error,
                    return_value=types.SimpleNamespace(stdout="13.30\n"),
                )
            ),
            patch.object(main.sys, "prefix", str(prefix)),
            patch.object(main.sys, "version_info", python_pair),
            patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stdout(output),
        ):
            result = main.check_environment()
        return result, output.getvalue()

    def test_supported_arcgis_versions_are_accepted_in_conda(self):
        with tempfile.TemporaryDirectory() as directory:
            for version in ("3.4.0", "3.5.1", "3.6.2", "3.7.0"):
                with self.subTest(version=version):
                    result, output = self._run_check(
                        self._arcpy(version=version, install_dir=directory),
                        directory,
                        {"CONDA_PREFIX": directory},
                        python_pair=(3, 11) if version.startswith(("3.4", "3.5")) else (3, 13),
                    )
                    self.assertTrue(result)
                    self.assertIn(f"ArcGIS Pro    : {version} OK", output)
                    self.assertIn("ArcPy runtime : Conda OK", output)

    def test_valid_arcgis_uv_wrapper_is_accepted_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install_dir = root / "ArcGIS" / "Pro"
            prefix = root / "project" / ".venv"
            prefix.mkdir(parents=True)
            home = install_dir / "bin" / "Python" / "envs" / "arcgispro-py3"
            (prefix / "pyvenv.cfg").write_text(
                f"home = {home}\nuv = 0.10.5\ninclude-system-site-packages = true\n",
                encoding="utf-8",
            )
            result, output = self._run_check(
                self._arcpy(install_dir=str(install_dir)),
                prefix,
                {"VIRTUAL_ENV": str(prefix)},
            )
        self.assertTrue(result)
        self.assertIn("OK WITH WARNING", output)

    def test_uv_wrapper_requires_system_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install_dir = root / "ArcGIS" / "Pro"
            prefix = root / ".venv"
            prefix.mkdir()
            home = install_dir / "bin" / "Python" / "envs" / "arcgispro-py3"
            (prefix / "pyvenv.cfg").write_text(
                f"home = {home}\nuv = 0.10.5\ninclude-system-site-packages = false\n",
                encoding="utf-8",
            )
            result, output = self._run_check(
                self._arcpy(install_dir=str(install_dir)),
                prefix,
                {"VIRTUAL_ENV": str(prefix)},
            )
        self.assertFalse(result)
        self.assertIn("system site packages are disabled", output)

    def test_uv_wrapper_requires_matching_arcgis_home(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / ".venv"
            prefix.mkdir()
            (prefix / "pyvenv.cfg").write_text(
                "home = C:/unrelated/python\nuv = 0.10.5\ninclude-system-site-packages = true\n",
                encoding="utf-8",
            )
            result, output = self._run_check(
                self._arcpy(install_dir=str(root / "ArcGIS" / "Pro")),
                prefix,
                {"VIRTUAL_ENV": str(prefix)},
            )
        self.assertFalse(result)
        self.assertIn("base interpreter does not match", output)

    def test_ordinary_virtual_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(self._arcpy(), directory, {})
        self.assertFalse(result)
        self.assertIn("use an initialized ArcGIS Conda environment", output)

    def test_unsupported_arcgis_version_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(
                self._arcpy(version="3.8.0", install_dir=directory),
                directory,
                {"CONDA_PREFIX": directory},
            )
        self.assertFalse(result)
        self.assertIn("3.8.0 UNSUPPORTED", output)

    def test_arcgis_python_generation_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(
                self._arcpy(version="3.5.0", install_dir=directory),
                directory,
                {"CONDA_PREFIX": directory},
                python_pair=(3, 13),
            )
        self.assertFalse(result)
        self.assertIn("EXPECTED 3.11", output)

    def test_arcpy_spatial_reference_smoke_failure_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(
                self._arcpy(install_dir=directory, spatial_code=0),
                directory,
                {"CONDA_PREFIX": directory},
            )
        self.assertFalse(result)
        self.assertIn("could not construct WGS84", output)

    def test_wrong_pinned_dependency_version_fails(self):
        versions = {
            distribution: expected
            for _, distribution, expected in main.PINNED_DEPENDENCIES.values()
        }
        versions["tqdm"] = "0.0.0"
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(
                self._arcpy(install_dir=directory),
                directory,
                {"CONDA_PREFIX": directory},
                versions,
            )
        self.assertFalse(result)
        self.assertIn("EXPECTED 4.67.1", output)

    def test_exiftool_execution_failure_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self._run_check(
                self._arcpy(install_dir=directory),
                directory,
                {"CONDA_PREFIX": directory},
                exiftool_error=main.subprocess.TimeoutExpired("exiftool", 10),
            )
        self.assertFalse(result)
        self.assertIn("ExifTool      : FAIL", output)


if __name__ == "__main__":
    unittest.main()
