"""
matcher.py
----------
Spatial matching: find which poles (as GPS points) fall within a
projected image footprint (shapely Polygon).
"""

from __future__ import annotations

from shapely.geometry import Point, Polygon

from src.camera import Camera
from src.frustum import pole_visible_in_image


def find_poles_in_frustum(
    poles: list[dict],
    footprint: Polygon,
    camera: Camera,
) -> list[int]:
    """
    Return the OIDs of poles visible in the image associated with *camera*.

    Two complementary checks are applied; a pole matches if *either* is true:

    1. **Ground footprint** – the pole's GPS coordinate (at ground level) falls
       inside the 2-D ground-plane polygon returned by ``get_footprint()``.
       Reliable for nadir imagery and for poles well within the oblique field
       of view.

    2. **Pole-top back-projection** – the pole's 3-D tip (at ``height_m``
       above the ground) projects into the pixel bounds of the image.
       Catches poles near the camera-side (near) edge of an oblique shot
       whose *base* lies outside the ground footprint but whose *top* is
       genuinely visible in the frame.

    Parameters
    ----------
    poles     : list of dicts with keys "oid", "lat", "lon", "height_m"
    footprint : shapely Polygon in (lon, lat) coordinates
    camera    : Camera instance for the image being processed

    Returns
    -------
    List of OID integers for poles visible in the image.
    """
    matched_oids: list[int] = []
    for pole in poles:
        pt = Point(pole["lon"], pole["lat"])  # (x=lon, y=lat) — matches footprint convention
        if footprint.contains(pt) or pole_visible_in_image(pole, camera):
            matched_oids.append(pole["oid"])
    return matched_oids
