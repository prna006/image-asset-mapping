from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class OutputCommitTests(unittest.TestCase):
    def _arcgis_module(self, datasets, fail_backup=False):
        def copy_features(source, destination):
            if fail_backup and "_output_backup_" in destination:
                raise RuntimeError("backup failed")
            datasets[destination] = datasets[source]

        return types.SimpleNamespace(
            copy_features=copy_features,
            dataset_exists=lambda path: path in datasets,
            delete_dataset=lambda path: datasets.pop(path, None),
        )

    def test_csv_promotion_failure_restores_existing_dataset_and_csv(self):
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_csv = root / "staged.csv"
            final_csv = root / "matches.csv"
            failure_csv = root / "matches_failures.csv"
            staged_csv.write_text("new csv", encoding="utf-8")
            final_csv.write_text("old csv", encoding="utf-8")
            datasets = {"stage.shp": "new dataset", "output.shp": "old dataset"}

            def fail_new_csv(source, destination):
                if Path(source) == staged_csv and Path(destination) == final_csv:
                    raise OSError("promotion failed")
                real_replace(source, destination)

            with (
                patch.dict(sys.modules, {"src.arcgis_io": self._arcgis_module(datasets)}),
                patch.object(main.os, "replace", side_effect=fail_new_csv),
            ):
                with self.assertRaises(OSError):
                    main._commit_outputs(
                        "stage.shp",
                        "output.shp",
                        staged_csv,
                        final_csv,
                        None,
                        failure_csv,
                    )

            self.assertEqual(datasets["output.shp"], "old dataset")
            self.assertEqual(final_csv.read_text(encoding="utf-8"), "old csv")

    def test_backup_failure_leaves_existing_outputs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_csv = root / "staged.csv"
            final_csv = root / "matches.csv"
            failure_csv = root / "matches_failures.csv"
            staged_csv.write_text("new csv", encoding="utf-8")
            final_csv.write_text("old csv", encoding="utf-8")
            datasets = {"stage.shp": "new dataset", "output.shp": "old dataset"}

            with patch.dict(
                sys.modules,
                {"src.arcgis_io": self._arcgis_module(datasets, fail_backup=True)},
            ):
                with self.assertRaises(RuntimeError):
                    main._commit_outputs(
                        "stage.shp",
                        "output.shp",
                        staged_csv,
                        final_csv,
                        None,
                        failure_csv,
                    )

            self.assertEqual(datasets["output.shp"], "old dataset")
            self.assertEqual(final_csv.read_text(encoding="utf-8"), "old csv")


if __name__ == "__main__":
    unittest.main()
