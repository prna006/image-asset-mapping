from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src import arcgis_io


@dataclass
class FakeField:
    name: str
    type: str = "String"
    length: int = 0


class FakeUpdateCursor:
    def __init__(self, rows):
        self.rows = rows
        self.updated = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.rows)

    def updateRow(self, row):
        self.updated.append(list(row))


class ArcgisIoTests(unittest.TestCase):
    def _fake_arcpy(self, rows=None, fields=None):
        cursor = FakeUpdateCursor(rows or [])
        field_list = list(fields or [])

        def add_field(_path, name, field_type, **kwargs):
            field_list.append(FakeField(name, field_type, kwargs.get("field_length", 0)))

        def delete_field(_path, names):
            lowered = {name.lower() for name in names}
            field_list[:] = [field for field in field_list if field.name.lower() not in lowered]

        return (
            types.SimpleNamespace(
                ListFields=lambda _path: field_list,
                management=types.SimpleNamespace(AddField=add_field, DeleteField=delete_field),
                da=types.SimpleNamespace(
                    SearchCursor=lambda _path, _fields: cursor,
                    UpdateCursor=lambda _path, _fields: cursor,
                ),
            ),
            cursor,
            field_list,
        )

    def test_write_matches_clears_unmatched_rows(self):
        fake, cursor, _ = self._fake_arcpy(
            rows=[[1, 99, "stale.jpg", "1.00"], [2, 1, "old.jpg", "2.00"]]
        )
        with patch.dict(sys.modules, {"arcpy": fake}):
            updated = arcgis_io.write_matches(
                "poles.shp",
                {2: ["new.jpg"]},
                {2: ["3.00"]},
            )
        self.assertEqual(updated, 2)
        self.assertEqual(cursor.updated[0], [1, 0, "", ""])
        self.assertEqual(cursor.updated[1], [2, 1, "new.jpg", "3.00"])

    def test_shapefile_overflow_omits_text_fields(self):
        fake, _, fields = self._fake_arcpy(
            fields=[FakeField("match_imgs"), FakeField("img_dists"), FakeField("match_cnt")]
        )
        with patch.dict(sys.modules, {"arcpy": fake}):
            include_text = arcgis_io.prepare_match_fields(
                "poles.shp",
                {1: ["x" * 255]},
                {1: ["1.00"]},
                "match_imgs",
                "img_dists",
            )
        self.assertFalse(include_text)
        self.assertEqual([field.name for field in fields], ["match_cnt"])

    def test_geodatabase_accepts_long_summary_fields(self):
        fake, _, fields = self._fake_arcpy()
        with patch.dict(sys.modules, {"arcpy": fake}):
            include_text = arcgis_io.prepare_match_fields(
                "results.gdb/poles",
                {1: ["x" * 400]},
                {1: ["1.00"]},
                "match_imgs",
                "img_dists",
            )
        self.assertTrue(include_text)
        lengths = {field.name: field.length for field in fields}
        self.assertEqual(lengths["match_imgs"], 400)

    def test_staged_output_validation_accepts_null_empty_text(self):
        fake, _, _ = self._fake_arcpy(rows=[[1, 0, None, None], [2, 1, "image.jpg", "3.00"]])
        with patch.dict(sys.modules, {"arcpy": fake}):
            row_count = arcgis_io.validate_match_output(
                "poles.shp",
                {2: ["image.jpg"]},
                {2: ["3.00"]},
            )
        self.assertEqual(row_count, 2)

    def test_validate_pole_dataset_accepts_defined_nonempty_points(self):
        cursor = FakeUpdateCursor([[1, (-77.0, 37.0)]])
        fake = types.SimpleNamespace(
            Exists=lambda _path: True,
            Describe=lambda _path: types.SimpleNamespace(
                shapeType="Point",
                spatialReference=types.SimpleNamespace(name="NAD 1983"),
                OIDFieldName="FID",
            ),
            SpatialReference=lambda code: types.SimpleNamespace(factoryCode=code),
            da=types.SimpleNamespace(SearchCursor=lambda *_args, **_kwargs: cursor),
        )
        with patch.dict(sys.modules, {"arcpy": fake}):
            self.assertEqual(arcgis_io.validate_pole_dataset("poles.shp"), 1)

    def test_validate_pole_dataset_rejects_wrong_geometry(self):
        fake = types.SimpleNamespace(
            Exists=lambda _path: True,
            Describe=lambda _path: types.SimpleNamespace(
                shapeType="Polyline",
                spatialReference=types.SimpleNamespace(name="WGS 1984"),
                OIDFieldName="FID",
            ),
        )
        with (
            patch.dict(sys.modules, {"arcpy": fake}),
            self.assertRaisesRegex(ValueError, "point geometry"),
        ):
            arcgis_io.validate_pole_dataset("lines.shp")

    def test_nonpositive_height_uses_default(self):
        cursor = FakeUpdateCursor([[1, (-77.0, 37.0), 0]])
        fake = types.SimpleNamespace(
            ListFields=lambda _path: [FakeField("HEIGHT")],
            SpatialReference=lambda code: types.SimpleNamespace(factoryCode=code),
            da=types.SimpleNamespace(SearchCursor=lambda *_args, **_kwargs: cursor),
        )
        with patch.dict(sys.modules, {"arcpy": fake}):
            poles = arcgis_io.load_poles("poles.shp", default_height_m=12.0)
        self.assertEqual(poles[0]["height_m"], 12.0)


if __name__ == "__main__":
    unittest.main()
