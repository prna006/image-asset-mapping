from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.results_io import (
    default_matches_csv_path,
    failure_csv_path,
    read_match_csv,
    write_failure_csv,
    write_match_csv,
)


class ResultsIoTests(unittest.TestCase):
    def test_default_csv_paths_for_shapefile_and_geodatabase(self):
        self.assertEqual(
            default_matches_csv_path("C:/work/poles.shp").name,
            "poles_matches.csv",
        )
        self.assertEqual(
            default_matches_csv_path("C:/work/results.gdb/network/poles").name,
            "results_network_poles_matches.csv",
        )

    def test_match_csv_is_complete_and_deterministic(self):
        rows = [
            {
                "pole_oid": 2,
                "pole_lat": 38,
                "pole_lon": -77,
                "image_name": "b.jpg",
                "distance_m": 9.126,
                "match_source": "fallback",
            },
            {
                "pole_oid": 10,
                "pole_lat": 39,
                "pole_lon": -78,
                "image_name": "c.jpg",
                "distance_m": 1,
                "match_source": "primary",
            },
            {
                "pole_oid": 1,
                "pole_lat": 37,
                "pole_lon": -76,
                "image_name": "a.jpg",
                "distance_m": 2,
                "match_source": "primary",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matches.csv"
            self.assertEqual(write_match_csv(output, rows), 3)
            loaded = read_match_csv(output)

        self.assertEqual([row["pole_oid"] for row in loaded], ["1", "2", "10"])
        self.assertEqual(loaded[1]["distance_m"], 9.13)

    def test_failure_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matches.csv"
            failures = failure_csv_path(output)
            write_failure_csv(
                failures,
                [{"image_name": "bad.jpg", "error_type": "ValueError", "error_message": "bad"}],
            )
            with failures.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
