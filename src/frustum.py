"""
frustum.py
----------
Projects the image boundary through the camera model onto the ground plane
and returns the resulting GPS footprint as a shapely Polygon.

Handles three cases:
  - Nadir (pitch â‰ˆ -90Â°): all 4 corners hit the ground â†’ rectangular footprint.
  - Oblique (pitch ~-30Â° to -60Â°): top corners above horizon â†’ binary-search
    the horizon pixel on each vertical image edge, clamp to MAX_RANGE_M, giving
    a trapezoidal footprint.
  - Near-horizontal (pitch â‰ˆ 0Â°): all corners above horizon â†’ all four edges
    are horizon-clamped at MAX_RANGE_M, producing a far-field footprint.

Ray-cast algorithm (from oblique_pipeline.py / base.py in reference repo):
  1. Undistort pixel coordinates: cv2.undistortPoints
  2. Lift to normalised camera ray: Kâ»Â¹ Â· [u, v, 1]áµ€
  3. Rotate into world frame:       R Â· camera_ray
  4. Rayâ€“ground intersection:       y = 0 plane (y-down convention,
                                    camera sits at [0, -rel_alt, 0])
  5. Metric offsets (vert, horz) â†’ GPS via geopy.geodesic bearing
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import MultiPoint, Polygon

from src.camera import Camera

# Maximum slant range (metres) used when a ray is at or above the horizon.
# Increase if images are taken from high altitude with a very oblique angle.
MAX_RANGE_M = 100.0
NADIR_THRESHOLD_DEG = 85.0
IMAGE_MARGIN_PX = 50.0


def configure_projection(
    max_range_m: float | None = None,
    nadir_threshold_deg: float | None = None,
    margin_px: float | None = None,
) -> None:
    """Override projection tuning parameters at runtime."""
    global MAX_RANGE_M, NADIR_THRESHOLD_DEG, IMAGE_MARGIN_PX

    if max_range_m is not None:
        if max_range_m <= 0:
            raise ValueError(f"max_range_m must be positive, got {max_range_m}.")
        MAX_RANGE_M = float(max_range_m)

    if nadir_threshold_deg is not None:
        if not 0 < nadir_threshold_deg <= 90:
            raise ValueError(f"nadir_threshold_deg must be in (0, 90], got {nadir_threshold_deg}.")
        NADIR_THRESHOLD_DEG = float(nadir_threshold_deg)

    if margin_px is not None:
        if margin_px < 0:
            raise ValueError(f"margin_px must be non-negative, got {margin_px}.")
        IMAGE_MARGIN_PX = float(margin_px)


# ---------------------------------------------------------------------------
# Core ray-cast helpers
# ---------------------------------------------------------------------------


def _world_ray(
    u: float,
    v: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Return the unit-direction world-frame ray for pixel (u, v)."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    k = cv2.undistortPoints(pts, K, dist_coeffs)
    nx, ny = float(k[0, 0, 0]), float(k[0, 0, 1])
    return R @ np.array([nx, ny, 1.0])


def _ray_to_ground(
    world_ray: np.ndarray,
    rel_alt: float,
    max_range_m: float | None = None,
) -> tuple[float, float]:
    """
    Intersect a world-frame ray with the ground plane (y = 0).

    Two-stage clamp:
    1. If world_ray[1] (downward component) is too small to reach the ground
       within max_range_m, clamp it to rel_alt / max_range_m so the ray lands
       at the horizon rather than shooting off to infinity.
    2. After projection, if the horizontal ground distance still exceeds
       max_range_m (possible for highly oblique horizontal view), scale both
       components back proportionally so the point sits exactly on the
       max_range_m circle around the drone.

    This ensures left/right corners of a near-horizontal image project to
    genuinely different GPS points (no degenerate LineString from convex_hull).

    Returns (vert_dist, horz_dist) in metres.
    """
    if max_range_m is None:
        max_range_m = MAX_RANGE_M

    wr_y = world_ray[1]
    min_wr_y = rel_alt / max_range_m  # y-component that gives exactly max_range_m

    if wr_y < min_wr_y:
        wr_y = min_wr_y

    vert_dist = rel_alt * world_ray[2] / wr_y
    horz_dist = rel_alt * world_ray[0] / wr_y

    # Hard clamp on total horizontal range
    horiz_range = math.sqrt(vert_dist**2 + horz_dist**2)
    if horiz_range > max_range_m:
        scale = max_range_m / horiz_range
        vert_dist *= scale
        horz_dist *= scale

    return float(vert_dist), float(horz_dist)


def _horizon_v(
    u: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    R: np.ndarray,
    img_height: int,
) -> float:
    """
    Binary-search for the image row v (at column u) where world_ray[1] = 0
    (the camera horizon). Returns a v in [0, img_height].

    If the whole column is below the horizon returns 0.
    If the whole column is above the horizon returns img_height.
    """

    def wr_y(v: float) -> float:
        return float(_world_ray(u, v, K, dist_coeffs, R)[1])

    y_top = wr_y(0.0)
    y_bot = wr_y(float(img_height))

    if y_top >= 0:  # whole column below horizon
        return 0.0
    if y_bot <= 0:  # whole column above horizon
        return float(img_height)

    # Bisect: find v where wr_y transitions from < 0 to >= 0
    lo, hi = 0.0, float(img_height)
    for _ in range(40):
        mid = (lo + hi) * 0.5
        if wr_y(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# GPS conversion
# ---------------------------------------------------------------------------


def _metric_to_gps(
    origin_lat: float,
    origin_lon: float,
    vert_dist: float,
    horz_dist: float,
) -> tuple[float, float]:
    """
    Convert metric ground offsets (vert, horz) to an absolute GPS coordinate.

    vert_dist and horz_dist are already in the fixed geographic frame
    (North and East respectively). The camera yaw has already been normalized
    to the same true-north frame when the Camera model was loaded.
    """
    # Base bearings for positive offsets (North and East).
    # If an offset is negative the bearing is reversed by 180° so that
    # geodesic() is always called with a positive distance (negative distances
    # are undefined behaviour in geopy and give wrong results for nadir imagery
    # where South/West offsets are common).
    bearing_north = 0.0
    bearing_east = 90.0

    if vert_dist < 0:
        bearing_north = (bearing_north + 180.0) % 360.0
    if horz_dist < 0:
        bearing_east = (bearing_east + 180.0) % 360.0

    pt = geodesic(meters=abs(vert_dist)).destination(
        (origin_lat, origin_lon), bearing=bearing_north
    )
    pt = geodesic(meters=abs(horz_dist)).destination(
        (pt.latitude, pt.longitude), bearing=bearing_east
    )
    return pt.latitude, pt.longitude


def _corner_to_gps(
    u: float,
    v: float,
    camera: Camera,
) -> tuple[float, float]:
    """Project a single pixel corner to GPS, clamping above-horizon rays."""
    wr = _world_ray(u, v, camera.K, camera.dist_coeffs, camera.R)
    vert_dist, horz_dist = _ray_to_ground(wr, camera.rel_alt)
    return _metric_to_gps(
        camera.gps_coords[0],
        camera.gps_coords[1],
        vert_dist,
        horz_dist,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _gps_to_metric(
    cam_lat: float,
    cam_lon: float,
    pole_lat: float,
    pole_lon: float,
) -> tuple[float, float]:
    """
    Return (north_off, east_off) in metres from the camera to the pole
    using a flat-earth approximation (accurate to sub-metre over <500 m).
    """
    _R_EARTH = 6_371_000.0
    dlat = math.radians(pole_lat - cam_lat)
    dlon = math.radians(pole_lon - cam_lon)
    north_off = _R_EARTH * dlat
    east_off = _R_EARTH * dlon * math.cos(math.radians(cam_lat))
    return north_off, east_off


def ground_distance_m(
    cam_lat: float,
    cam_lon: float,
    pole_lat: float,
    pole_lon: float,
) -> float:
    """Return horizontal ground distance in metres from the camera to the pole."""
    north_off, east_off = _gps_to_metric(cam_lat, cam_lon, pole_lat, pole_lon)
    return math.hypot(north_off, east_off)


def _pole_point_in_image(
    east_off: float,
    north_off: float,
    height_m: float,
    camera: Camera,
    margin_px: float | None = None,
) -> bool:
    """
    Return True if the point at (east_off, north_off, height_m) projects into
    the image frame.

    World convention: x = East, y = Down, z = North.
    Camera sits at [0, -rel_alt, 0]; the sample point at [east, -height_m, north].

    margin_px : number of pixels to expand the image bounds on all sides.
                Absorbs GPS position error in oblique frames where a genuinely
                visible pole may project a few dozen pixels outside the nominal
                image boundary.
    """
    if margin_px is None:
        margin_px = IMAGE_MARGIN_PX

    V = np.array(
        [east_off, camera.rel_alt - height_m, north_off],
        dtype=np.float64,
    )

    # Reject points beyond the same slant-range cap used for the ground footprint.
    if np.linalg.norm(V) > MAX_RANGE_M:
        return False

    # Camera-frame vector
    V_cam = camera.R.T @ V

    # Point is behind the lens — not visible
    if V_cam[2] <= 0.0:
        return False

    # Project to pixel coordinates (rvec/tvec = 0 → point is already in camera frame)
    pts_2d, _ = cv2.projectPoints(
        V_cam.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera.K,
        camera.dist_coeffs,
    )
    u = float(pts_2d[0, 0, 0])
    v = float(pts_2d[0, 0, 1])

    return (-margin_px <= u <= camera.image_width + margin_px) and (
        -margin_px <= v <= camera.image_height + margin_px
    )


def pole_visible_in_image(pole: dict, camera: Camera) -> bool:
    """
    Return True if *any* sampled point along the pole back-projects into the
    image frame.

    Three points are tested along the pole shaft (top, upper-third, lower-third)
    so that oblique shots where only part of the pole is visible are matched.

    Algorithm
    ---------
    1. Compute metric offsets (north, east) from camera GPS to pole GPS.
    2. For each sample height, build the world-frame vector and delegate to
       _pole_point_in_image.
    3. Return True on the first hit; False if all samples miss.
    """
    cam_lat, cam_lon = camera.gps_coords
    north_off, east_off = _gps_to_metric(cam_lat, cam_lon, pole["lat"], pole["lon"])

    total_height = pole["height_m"]
    # Six evenly-spaced samples from top to near-base.
    # Dense coverage ensures oblique shots that clip only part of the shaft
    # are still matched. The 0.1× sample catches near-horizontal views where
    # only the base stub is visible. Ground level (0) is already handled by
    # the footprint containment check in matcher.py.
    sample_fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    for frac in sample_fractions:
        if _pole_point_in_image(east_off, north_off, total_height * frac, camera):
            return True

    return False


# ---------------------------------------------------------------------------
# Nadir footprint (pitch <= -85°)
# ---------------------------------------------------------------------------


def _nadir_footprint(camera: Camera) -> Polygon:
    """
    Return a yaw-rotated rectangular GPS Polygon for a near-nadir camera.

    Ground half-extents derived directly from the camera intrinsics:
      half_width_m  = rel_alt × (image_width  / 2) / fx
      half_height_m = rel_alt × (image_height / 2) / fy

    The rectangle is then rotated by yaw_deg (clockwise from North) so it
    aligns with the drone heading, and each corner is projected to GPS.
    """
    fx = float(camera.K[0, 0])
    fy = float(camera.K[1, 1])
    hw = camera.rel_alt * (camera.image_width / 2.0) / fx  # East half-extent  (m)
    hh = camera.rel_alt * (camera.image_height / 2.0) / fy  # North half-extent (m)

    # Un-rotated corners: (east_off, north_off)
    # image +u → East,  image +v → South (north negated for +v)
    raw_corners = [
        (-hw, +hh),  # top-left
        (+hw, +hh),  # top-right
        (+hw, -hh),  # bottom-right
        (-hw, -hh),  # bottom-left
    ]

    # Rotate by yaw clockwise from North:
    #   East'  = E·cos θ − N·sin θ
    #   North' = E·sin θ + N·cos θ
    yaw_rad = math.radians(camera.yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    cam_lat, cam_lon = camera.gps_coords

    pts_gps = []
    for east, north in raw_corners:
        east_r = east * cos_y - north * sin_y
        north_r = east * sin_y + north * cos_y
        lat, lon = _metric_to_gps(
            cam_lat,
            cam_lon,
            north_r,
            east_r,
        )
        pts_gps.append((lon, lat))

    return Polygon(pts_gps)


def get_footprint(camera: Camera):
    """
    Project the image boundary onto the ground and return a shapely Polygon.

    For each image corner:
      - If it hits the ground within MAX_RANGE_M, project it directly.
      - If it is above the horizon, find the horizon pixel on the same image
        edge (binary search) and project that point clamped to MAX_RANGE_M.

    This produces a correct trapezoidal or rectangular footprint for nadir,
    oblique, and side-view images alike.

    Returns
    -------
    shapely.geometry.Polygon  in (lon, lat) coordinates.
    """
    if camera.rel_alt <= 0:
        raise ValueError(
            f"Camera altitude must be positive (got rel_alt={camera.rel_alt:.2f} m) "
            f"for {camera.image_path.name}. Check EXIF altitude tags."
        )

    # Near-nadir shortcut: only for pitch in [-90°, -85°].
    # Ray-casting is numerically unreliable when the camera points straight
    # down; use a direct focal-length calculation instead.
    if camera.pitch_deg <= -NADIR_THRESHOLD_DEG:
        return _nadir_footprint(camera)

    W, H = camera.image_width, camera.image_height

    # ---- bottom corners (almost always below horizon) ----
    pts_gps: list[tuple[float, float]] = []  # (lon, lat)

    for u, v in [(0, H), (W, H), (W, 0), (0, 0)]:
        wr = _world_ray(u, v, camera.K, camera.dist_coeffs, camera.R)

        if wr[1] >= camera.rel_alt / MAX_RANGE_M:
            # Below horizon (or close enough): project normally
            lat, lon = _corner_to_gps(u, v, camera)
        else:
            # Above horizon: find horizon pixel on this vertical/horizontal edge
            # and project at MAX_RANGE_M
            if v == 0:
                # top edge corner â†’ find horizon on the vertical edge (same u)
                v_h = _horizon_v(u, camera.K, camera.dist_coeffs, camera.R, H)
            else:
                # bottom edge corner above horizon would be bizarre; use v as-is
                v_h = v

            wr_h = _world_ray(u, v_h, camera.K, camera.dist_coeffs, camera.R)
            vert_dist, horz_dist = _ray_to_ground(wr_h, camera.rel_alt)
            lat, lon = _metric_to_gps(
                camera.gps_coords[0],
                camera.gps_coords[1],
                vert_dist,
                horz_dist,
            )

        pts_gps.append((lon, lat))

    # Anchor the footprint to the camera's own ground position.
    # Without this, oblique footprints start some distance ahead of the drone,
    # leaving a blind spot for poles that are close to (or directly below) the
    # flight path.  Adding the camera GPS as a vertex pulls the near edge back
    # to the drone's ground track so those poles fall inside the polygon.
    cam_lat, cam_lon = camera.gps_coords
    pts_gps.append((cam_lon, cam_lat))

    hull = MultiPoint(pts_gps).convex_hull
    if not isinstance(hull, Polygon):
        raise ValueError(
            f"Footprint for image at {camera.gps_coords} is degenerate "
            f"(hull type: {type(hull).__name__}). Check camera intrinsics and orientation."
        )
    return hull
