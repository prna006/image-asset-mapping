from __future__ import annotations

import unittest

from tools.export_image_to_pole_csv import build_closest_mapping


class ExporterTests(unittest.TestCase):
    def test_closest_pole_wins_with_oid_tie_break(self):
        rows = [
            {"image_name": "a.jpg", "pole_oid": "2", "distance_m": 5.0},
            {"image_name": "a.jpg", "pole_oid": "1", "distance_m": 5.0},
            {"image_name": "b.jpg", "pole_oid": "2", "distance_m": 3.0},
            {"image_name": "c.jpg", "pole_oid": "10", "distance_m": 5.0},
            {"image_name": "c.jpg", "pole_oid": "2", "distance_m": 5.0},
        ]
        poles = {
            "1": {"pole_oid": 1, "pole_lat": 1.0, "pole_lon": 2.0, "attributes": {"kind": "A"}},
            "2": {"pole_oid": 2, "pole_lat": 3.0, "pole_lon": 4.0, "attributes": {"kind": "B"}},
            "10": {"pole_oid": 10, "pole_lat": 5.0, "pole_lon": 6.0, "attributes": {}},
        }
        closest = build_closest_mapping(rows, poles)
        self.assertEqual(closest["a.jpg"]["pole_oid"], 1)
        self.assertEqual(closest["b.jpg"]["pole_attributes"], {"kind": "B"})
        self.assertEqual(closest["c.jpg"]["pole_oid"], 2)


if __name__ == "__main__":
    unittest.main()
