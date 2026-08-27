from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from src.camera import MetadataValidationError, camera_from_metadata, load_dem_from_file
except ModuleNotFoundError:
    MetadataValidationError = ValueError
    camera_from_metadata = None
    load_dem_from_file = None


@unittest.skipIf(camera_from_metadata is None, "camera runtime dependencies are not installed")
class CameraMetadataTests(unittest.TestCase):
    def _base(self, make="Skydio"):
        metadata = {
            "EXIF:Make": make,
            "EXIF:ImageWidth": 4000,
            "EXIF:ImageHeight": 3000,
            "EXIF:GPSLatitude": 0.0,
            "EXIF:GPSLatitudeRef": "N",
            "EXIF:GPSLongitude": 0.0,
            "EXIF:GPSLongitudeRef": "E",
        }
        if make == "Skydio":
            metadata.update(
                {
                    "XMP:CalibratedFocalLengthX": 2500,
                    "XMP:CalibratedFocalLengthY": 2500,
                }
            )
        return metadata

    def test_skydio_accepts_real_zero_coordinates_and_angles(self):
        meta = {
            **self._base(),
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": 0,
            "XMP:CameraOrientationNEDYaw": 0,
            "XMP:CameraPositionFLUZ": "0,0,30",
        }
        camera = camera_from_metadata(meta, Path("synthetic.jpg"))
        self.assertEqual(camera.gps_coords, (0.0, 0.0))
        self.assertEqual(camera.rel_alt, 30.0)

    def test_dji_calibrated_camera_is_valid(self):
        meta = {
            **self._base("DJI"),
            "XMP:CalibratedFocalLength": 2800,
            "XMP:GimbalRollDegree": 0,
            "XMP:GimbalPitchDegree": -90,
            "XMP:GimbalYawDegree": 0,
            "XMP:RelativeAltitude": 40,
        }
        with patch("src.camera._compute_mag_declination", return_value=0):
            camera = camera_from_metadata(meta, "synthetic.jpg")
        self.assertEqual(camera.vendor, "dji")
        self.assertEqual(camera.rel_alt, 40)

    def test_generic_requires_exif_focal_lengths(self):
        meta = {
            **self._base("Other"),
            "XMP:GimbalRollDegree": 0,
            "XMP:GimbalPitchDegree": -90,
            "XMP:GimbalYawDegree": 0,
            "XMP:RelativeAltitude": 25,
        }
        with self.assertRaisesRegex(MetadataValidationError, "FocalLengthIn35mmFormat"):
            camera_from_metadata(meta, "synthetic.jpg")

    def test_missing_orientation_is_not_defaulted_to_zero(self):
        meta = {
            **self._base(),
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": -90,
            "XMP:CameraPositionFLUZ": "0,0,30",
        }
        with self.assertRaisesRegex(MetadataValidationError, "CameraOrientationNEDYaw"):
            camera_from_metadata(meta, "synthetic.jpg")

    def test_invalid_coordinate_and_nonfinite_value_are_rejected(self):
        meta = {
            **self._base(),
            "EXIF:GPSLatitude": 91,
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": -90,
            "XMP:CameraOrientationNEDYaw": float("nan"),
            "XMP:CameraPositionFLUZ": "0,0,30",
        }
        with self.assertRaises(MetadataValidationError):
            camera_from_metadata(meta, "synthetic.jpg")

    def test_absolute_altitude_requires_terrain_when_relative_is_missing(self):
        meta = {
            **self._base(),
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": -90,
            "XMP:CameraOrientationNEDYaw": 0,
            "XMP:GpsMslHeight": 100,
        }
        with (
            patch("src.camera._get_terrain_elevation", return_value=None),
            self.assertRaises(MetadataValidationError),
        ):
            camera_from_metadata(meta, "synthetic.jpg")

    def test_safe_optical_center_and_distortion_defaults(self):
        meta = {
            **self._base(),
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": -90,
            "XMP:CameraOrientationNEDYaw": 0,
            "XMP:CameraPositionFLUZ": "0,0,30",
        }
        camera = camera_from_metadata(meta, "synthetic.jpg")
        self.assertEqual(camera.K[0, 2], 2000)
        self.assertEqual(camera.K[1, 2], 1500)
        self.assertTrue((camera.dist_coeffs == 0).all())

    def test_malformed_skydio_relative_altitude_is_rejected(self):
        meta = {
            **self._base(),
            "XMP:CalibratedFocalLengthX": 2500,
            "XMP:CameraOrientationNEDRoll": 0,
            "XMP:CameraOrientationNEDPitch": -90,
            "XMP:CameraOrientationNEDYaw": 0,
            "XMP:CameraPositionFLUZ": "not-a-vector",
        }
        with self.assertRaisesRegex(MetadataValidationError, "CameraPositionFLUZ"):
            camera_from_metadata(meta, "synthetic.jpg")

    def test_projected_dem_is_rejected(self):
        fake_arcpy = types.SimpleNamespace(
            sa=types.SimpleNamespace(Raster=lambda _path: object()),
            Describe=lambda _path: types.SimpleNamespace(
                extent=object(),
                spatialReference=types.SimpleNamespace(
                    name="NAD 1983 UTM Zone 18N", type="Projected", factoryCode=26918
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            dem = Path(directory) / "projected.tif"
            dem.touch()
            with (
                patch.dict(sys.modules, {"arcpy": fake_arcpy}),
                self.assertRaisesRegex(ValueError, "EPSG:4326"),
            ):
                load_dem_from_file(dem)


if __name__ == "__main__":
    unittest.main()
