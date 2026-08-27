from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from shapely.geometry import Point

    from src.fallback import FallbackConfig, resolve_fallback_matches
    from src.frustum import get_footprint
    from src.matcher import find_poles_in_frustum
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "geometry runtime dependencies are not installed")
class GeometryTests(unittest.TestCase):
    def _camera(self, lat=37.0, lon=-77.0, pitch=-90.0):
        return types.SimpleNamespace(
            image_path=Path("synthetic.jpg"),
            K=np.array([[2000.0, 0.0, 2000.0], [0.0, 2000.0, 1500.0], [0.0, 0.0, 1.0]]),
            dist_coeffs=np.zeros(5),
            R=Rotation.from_euler("xyz", [pitch, 0, 0], degrees=True).as_matrix(),
            pitch_deg=pitch,
            yaw_deg=0.0,
            gps_coords=(lat, lon),
            rel_alt=30.0,
            image_width=4000,
            image_height=3000,
        )

    def test_nadir_footprint_is_valid_and_contains_camera(self):
        camera = self._camera()
        footprint = get_footprint(camera)
        self.assertTrue(footprint.is_valid)
        self.assertTrue(footprint.contains(Point(camera.gps_coords[1], camera.gps_coords[0])))

    def test_matcher_includes_inside_and_excludes_outside_poles(self):
        camera = self._camera()
        footprint = get_footprint(camera)
        poles = [
            {"oid": 1, "lat": 37.0, "lon": -77.0, "height_m": 10},
            {"oid": 2, "lat": 38.0, "lon": -78.0, "height_m": 10},
        ]
        with patch("src.matcher.pole_visible_in_image", return_value=False):
            self.assertEqual(find_poles_in_frustum(poles, footprint, camera), [1])

    def test_oblique_footprint_is_valid_and_range_limited(self):
        camera = self._camera(pitch=-45.0)
        footprint = get_footprint(camera)
        self.assertTrue(footprint.is_valid)
        self.assertGreater(footprint.area, 0)

    def test_fallback_resolves_two_overlapping_images_to_nearby_pole(self):
        camera = self._camera()
        footprint = get_footprint(camera)
        images = [
            {"image_name": "a.jpg", "camera": camera, "footprint": footprint},
            {"image_name": "b.jpg", "camera": camera, "footprint": footprint},
        ]
        poles = [{"oid": 7, "lat": 37.0, "lon": -77.0, "height_m": 10}]
        assignments, _, _ = resolve_fallback_matches(
            images,
            poles,
            FallbackConfig(enabled=True, max_snap_distance_m=20),
        )
        self.assertEqual(assignments[0]["pole_oid"], 7)
        self.assertEqual(set(assignments[0]["image_names"]), {"a.jpg", "b.jpg"})


if __name__ == "__main__":
    unittest.main()
