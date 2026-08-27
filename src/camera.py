"""
camera.py
---------
Parses EXIF/XMP metadata from a drone image and builds the camera model
(intrinsics, distortion, rotation, GPS, altitude) needed for frustum projection.

Vendor support: "skydio", "dji", "generic"
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import exiftool
import numpy as np
from pygeomag import GeoMag
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signed_gps(value: float, ref: str) -> float:
    """Convert unsigned degrees + cardinal ref to signed float."""
    if ref in ("S", "W"):
        return -abs(value)
    return abs(value)


def _decimal_year(dt: datetime) -> float:
    """Convert a datetime to a decimal year (needed by pygeomag)."""
    start = datetime(dt.year, 1, 1)
    next_year = datetime(dt.year + 1, 1, 1)
    fraction = (dt - start) / (next_year - start)
    return dt.year + fraction


def _compute_mag_declination(
    vendor: str,
    gps_coords: tuple[float, float],
    abs_alt: float,
) -> float:
    """Return the magnetic declination in degrees for a DJI image."""
    if vendor != "dji":
        return 0.0
    try:
        lat, lon = gps_coords
        gm = GeoMag()
        result = gm.calculate(
            glat=lat,
            glon=lon,
            alt=abs_alt / 1000.0,
            time=_decimal_year(datetime.now()),
        )
        return result.d
    except Exception as exc:
        print(f"  WARN: mag declination lookup failed for {gps_coords}: {exc}; using 0.0")
        return 0.0


def _sensor_dims_from_35mm(flen_35mm: float, flen: float, img_w: int, img_h: int):
    """Return (sensor_width_mm, sensor_height_mm) from 35mm-equivalent focal length."""
    if flen <= 0 or flen_35mm <= 0 or img_h <= 0 or img_w <= 0:
        raise ValueError(
            f"Cannot derive sensor size: flen_35mm={flen_35mm}, flen={flen}, "
            f"img_w={img_w}, img_h={img_h} — all values must be positive."
        )
    # 35mm full-frame diagonal = 43.2669 mm
    full_frame_diag = 43.2669
    crop_factor = flen_35mm / flen
    sensor_diag = full_frame_diag / crop_factor
    aspect = img_w / img_h
    sensor_w = math.sqrt(sensor_diag**2 / (1 + 1 / aspect**2))
    sensor_h = sensor_w / aspect
    return sensor_w, sensor_h


def _parse_distortion_dewarp(dewarp_str: str) -> np.ndarray:
    """
    Parse Skydio DewarpData into an OpenCV (k1, k2, p1, p2, k3) vector.
    DewarpData is a comma-separated list of values; Skydio stores radial
    coefficients in k1, k2, k3 order (p1=p2=0 when tangential terms absent).
    """
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate(dewarp_str.split(",")[:5]):
        try:
            coeffs[i] = float(v.strip())
        except ValueError:
            pass  # malformed token — keep 0.0 for that coefficient
    return np.array(coeffs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Vendor detection from EXIF:Make
# ---------------------------------------------------------------------------

# Maps the lowercase EXIF:Make string to the internal vendor routing key.
# Extend this table if additional drone manufacturers need explicit support.
_MAKE_TO_VENDOR: dict[str, str] = {
    "dji": "dji",
    "skydio": "skydio",
}


class MetadataValidationError(ValueError):
    """Raised when image metadata cannot produce a trustworthy camera model."""


def _number(meta: dict, tag: str, *, positive: bool = False) -> float:
    """Read a required, finite numeric metadata value without treating zero as absent."""
    if tag not in meta or meta[tag] in (None, ""):
        raise MetadataValidationError(f"Missing required metadata tag {tag}.")
    try:
        value = float(meta[tag])
    except (TypeError, ValueError) as exc:
        raise MetadataValidationError(
            f"Metadata tag {tag} must be numeric; got {meta[tag]!r}."
        ) from exc
    if not math.isfinite(value):
        raise MetadataValidationError(f"Metadata tag {tag} must be finite; got {value!r}.")
    if positive and value <= 0:
        raise MetadataValidationError(f"Metadata tag {tag} must be positive; got {value!r}.")
    return value


def _optional_number(meta: dict, *tags: str, positive: bool = False) -> float | None:
    for tag in tags:
        if tag in meta and meta[tag] not in (None, ""):
            return _number(meta, tag, positive=positive)
    return None


def _orientation(meta: dict, tags: tuple[str, str, str]) -> tuple[float, float, float]:
    return tuple(_number(meta, tag) for tag in tags)  # type: ignore[return-value]


def _gps(meta: dict) -> tuple[float, float]:
    lat = _number(meta, "EXIF:GPSLatitude")
    lon = _number(meta, "EXIF:GPSLongitude")
    lat_ref = str(meta.get("EXIF:GPSLatitudeRef", "")).strip().upper()
    lon_ref = str(meta.get("EXIF:GPSLongitudeRef", "")).strip().upper()
    if lat_ref not in {"N", "S"}:
        raise MetadataValidationError("EXIF:GPSLatitudeRef must be N or S.")
    if lon_ref not in {"E", "W"}:
        raise MetadataValidationError("EXIF:GPSLongitudeRef must be E or W.")
    lat = _signed_gps(lat, lat_ref)
    lon = _signed_gps(lon, lon_ref)
    if not -90 <= lat <= 90:
        raise MetadataValidationError(f"GPS latitude is outside [-90, 90]: {lat}.")
    if not -180 <= lon <= 180:
        raise MetadataValidationError(f"GPS longitude is outside [-180, 180]: {lon}.")
    return lat, lon


def _vendor_from_make(meta: dict) -> str:
    """
    Derive the vendor routing string from the ``EXIF:Make`` metadata tag.

    Returns ``"skydio"``, ``"dji"``, or ``"generic"``.
    An absent or unrecognized Make value produces ``"generic"`` and emits a
    console warning so the caller knows which fallback branch was taken.
    """
    raw = str(meta.get("EXIF:Make") or "").strip()
    key = raw.lower()
    if key in _MAKE_TO_VENDOR:
        return _MAKE_TO_VENDOR[key]
    if raw:
        print(f"  WARN: Unrecognized EXIF:Make value {raw!r}; defaulting to 'generic'.")
    return "generic"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    """Fully-parsed camera model for one drone image."""

    image_path: Path

    # Intrinsics
    K: np.ndarray  # 3×3 camera matrix
    dist_coeffs: np.ndarray  # (k1, k2, p1, p2, k3)

    # Extrinsics / orientation
    R: np.ndarray  # 3×3 rotation matrix (camera → NED-like local frame)
    pitch_deg: float
    yaw_deg: float  # true-north referenced for DJI after load-time normalization
    roll_deg: float

    # Position
    gps_coords: tuple[float, float]  # (lat, lon) signed degrees
    # Metres AGL (3DEP-corrected when available, otherwise above takeoff from XMP)
    rel_alt: float
    abs_alt: float  # metres AMSL

    # Image dimensions
    image_width: int
    image_height: int

    # Vendor string (lowercased)
    vendor: str = "skydio"

    # Derived
    mag_declination: float = 0.0

    def __post_init__(self) -> None:
        if self.mag_declination == 0.0:
            self.mag_declination = _compute_mag_declination(
                self.vendor,
                self.gps_coords,
                self.abs_alt,
            )


# ---------------------------------------------------------------------------
# Terrain elevation (3DEP)
# ---------------------------------------------------------------------------

# Module-level DEM tile cache populated by prefetch_dem_for_images() before the
# main processing loop.  When set, _get_terrain_elevation samples from this
# in-memory array instead of issuing a separate HTTP request per camera frame.
_dem_tile: dict | None = None  # keys: array (float32 ndarray), xmin, ymax, cell_w, cell_h


@functools.lru_cache(maxsize=512)
def _get_terrain_elevation(lat: float, lon: float) -> float | None:
    """
    Return terrain elevation (metres AMSL) at (lat, lon) via the USGS
    3DEP ImageServer (1-metre resolution where available, automatically
    falling back to 3 m / 10 m / 30 m for areas without 1 m coverage).

    Inputs must be WGS84 (EPSG:4326) — exactly as stored in drone EXIF GPS.
    Note: the pole shapefile uses NAD83 (EPSG:4269), but those coordinates
    are not involved in this terrain query.

    Inputs are pre-rounded to 4 decimal places (~11 m) by the caller so that
    nearby frames share the same cached result.

    Returns None when the point is outside USGS coverage (non-US), the
    service is unreachable, or any other error occurs — the caller falls back
    to the XMP-derived rel_alt in that case.
    """
    _NO_DATA_SENTINEL = -10_000.0  # treat any value below this as no-data

    # ---- fast path: sample in-memory prefetched tile ----
    tile = _dem_tile
    if tile is not None:
        col = int((lon - tile["xmin"]) / tile["cell_w"])
        row = int((tile["ymax"] - lat) / tile["cell_h"])
        arr = tile["array"]
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            val = float(arr[row, col])
            if not math.isnan(val) and val > _NO_DATA_SENTINEL:
                return val
            return None  # no-data pixel; skip HTTP
        # point falls outside the prefetched tile bbox — fall through to HTTP

    # ---- slow path: single-point HTTP identify ----
    import json
    import urllib.parse
    import urllib.request

    try:
        # 3DEP ImageServer identify — uses the highest-resolution mosaic layer
        # (1 m 3DEP where available; auto-falls back to 3 m / 10 m / 30 m).
        geometry = json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}})
        params = urllib.parse.urlencode(
            {
                "geometry": geometry,
                "geometryType": "esriGeometryPoint",
                "returnGeometry": "false",
                "returnCatalogItems": "false",
                "f": "json",
            }
        )
        url = (
            "https://elevation.nationalmap.gov/arcgis/rest/services"
            f"/3DEPElevation/ImageServer/identify?{params}"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        raw_value = data.get("value", "NoData")
        if raw_value in (None, "NoData", ""):
            return None
        elev = float(raw_value)
        if elev <= _NO_DATA_SENTINEL:
            return None  # outside US coverage or over water
        return elev
    except Exception:
        return None


def _compute_true_agl(
    lat: float,
    lon: float,
    msl_alt: float,
    xmp_rel_alt: float,
) -> float:
    """
    Return the best available AGL estimate for a camera frame.

    Uses a positive vendor-relative altitude when present. Otherwise derives
    AGL as ``msl_alt - terrain_elevation`` using the configured DEM cache or
    USGS 3DEP. Missing terrain is an error when no relative altitude exists.

    Parameters
    ----------
    lat, lon      : WGS84 decimal degrees (EXIF GPS coordinates)
    msl_alt       : camera altitude above MSL in metres
                    (DJI ``XMP:AbsoluteAltitude``, Skydio ``XMP:GpsMslHeight``)
    xmp_rel_alt   : fallback AGL from XMP tag (above takeoff)
    """
    _MIN_AGL = 1.0  # clamp — no credible drone survey flies < 1 m AGL

    if xmp_rel_alt > 0:
        return xmp_rel_alt
    if msl_alt > 0.0:
        terrain_elev = _get_terrain_elevation(round(lat, 4), round(lon, 4))
        if terrain_elev is not None:
            agl = msl_alt - terrain_elev
            if agl >= _MIN_AGL:
                return agl

    raise MetadataValidationError(
        "No usable positive relative altitude and terrain elevation was unavailable "
        "for the absolute/MSL altitude. Supply a WGS84 DEM or valid relative-altitude metadata."
    )


def prefetch_dem_tile(
    coords: list[tuple[float, float]],
    padding_deg: float = 0.02,
) -> None:
    """
    Download 3DEP F32 elevation data covering all (lat, lon) pairs and store it
    in the module-level _dem_tile cache for fast in-memory sampling.

    Uses the USGS 3DEP ImageServer exportImage endpoint, which returns 1 m
    resolution where available and falls back to 3 m / 10 m / 30 m elsewhere.

    To avoid HTTP 504 Gateway Timeout errors the total area is split into a
    grid of sub-tiles, each at most _SUBTILE_PX × _SUBTILE_PX pixels.  The
    sub-tiles are fetched sequentially and stitched into one NumPy array.  For
    large surveys this gracefully reduces resolution so every sub-tile stays
    within the server limit regardless of survey extent.

    Silently no-ops on any error so the per-point HTTP fallback remains intact.
    """
    global _dem_tile
    import io
    import urllib.parse
    import urllib.request

    from tqdm import tqdm

    if not coords:
        return

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    xmin = min(lons) - padding_deg
    xmax = max(lons) + padding_deg
    ymin = min(lats) - padding_deg
    ymax = max(lats) + padding_deg

    # Target ~1 m/px.  1° ≈ 111 320 m; longitude degree shrinks with cos(lat).
    mid_lat = (ymin + ymax) / 2.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
    m_per_deg_lat = 111_320.0
    px_w_full = int((xmax - xmin) * m_per_deg_lon)
    px_h_full = int((ymax - ymin) * m_per_deg_lat)

    # Maximum pixels per sub-tile request.  Keeping this at 1 000 ensures each
    # individual request completes well within the USGS gateway's timeout even
    # on slow connections (~4 MB per sub-tile at F32).
    _SUBTILE_PX = 1_000
    # Overall cap: degrade resolution rather than issuing hundreds of requests.
    _MAX_TOTAL_PX = 5_000
    scale = min(1.0, _MAX_TOTAL_PX / max(px_w_full, px_h_full, 1))
    px_w_full = max(1, int(px_w_full * scale))
    px_h_full = max(1, int(px_h_full * scale))

    # Number of sub-tile columns / rows needed.
    n_cols = max(1, math.ceil(px_w_full / _SUBTILE_PX))
    n_rows = max(1, math.ceil(px_h_full / _SUBTILE_PX))
    n_tiles = n_cols * n_rows

    cell_w = (xmax - xmin) / px_w_full
    cell_h = (ymax - ymin) / px_h_full

    print(
        f"Prefetching 3DEP tile ({px_w_full}\u00d7{px_h_full} px) "
        f"for {len(coords)} image(s) "
        f"[{n_tiles} sub-tile(s) of \u2264{_SUBTILE_PX}\u00d7{_SUBTILE_PX} px] …"
    )

    _CHUNK = 256 * 1024  # streaming chunk size (bytes)
    _TIMEOUT = 60  # per-read socket timeout per sub-tile request

    def _fetch_subtile(
        sx_min: float,
        sy_min: float,
        sx_max: float,
        sy_max: float,
        spx_w: int,
        spx_h: int,
    ) -> np.ndarray | None:
        """Fetch one sub-tile; return float32 ndarray or None on failure."""
        params = urllib.parse.urlencode(
            {
                "bbox": f"{sx_min},{sy_min},{sx_max},{sy_max}",
                "bboxSR": "4326",
                "size": f"{spx_w},{spx_h}",
                "imageSR": "4326",
                "format": "tiff",
                "pixelType": "F32",
                "noData": "-3.40282347e+38",
                "noDataInterpretation": "esriNoDataMatchAny",
                "interpolation": "+RSP_BilinearInterpolation",
                "f": "image",
            }
        )
        url = (
            "https://elevation.nationalmap.gov/arcgis/rest/services"
            f"/3DEPElevation/ImageServer/exportImage?{params}"
        )
        buf = io.BytesIO()
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                buf.write(chunk)
        from PIL import Image

        img = Image.open(buf)
        arr = np.array(img, dtype=np.float32)
        arr[arr < -1e37] = np.nan
        return arr

    # Allocate the full output array (NaN everywhere initially).
    full_arr = np.full((px_h_full, px_w_full), np.nan, dtype=np.float32)

    failed = 0
    tile_idx = 0
    with tqdm(
        total=n_tiles,
        desc="Downloading DEM",
        unit="tile",
        dynamic_ncols=True,
        disable=None,
    ) as dem_progress:
        for row_idx in range(n_rows):
            # Pixel row range for this sub-tile (top-to-bottom in pixel space,
            # but geographic y increases northward).
            row_px_start = row_idx * _SUBTILE_PX
            row_px_end = min(row_px_start + _SUBTILE_PX, px_h_full)
            spx_h = row_px_end - row_px_start
            # Geographic bbox: row 0 is the northernmost strip.
            sy_max = ymax - row_px_start * cell_h
            sy_min = ymax - row_px_end * cell_h

            for col_idx in range(n_cols):
                tile_idx += 1
                col_px_start = col_idx * _SUBTILE_PX
                col_px_end = min(col_px_start + _SUBTILE_PX, px_w_full)
                spx_w = col_px_end - col_px_start
                sx_min = xmin + col_px_start * cell_w
                sx_max = xmin + col_px_end * cell_w

                try:
                    sub = _fetch_subtile(
                        sx_min,
                        sy_min,
                        sx_max,
                        sy_max,
                        spx_w,
                        spx_h,
                    )
                    if sub is not None:
                        # sub shape may differ by ±1 px due to server rounding.
                        actual_h = min(sub.shape[0], spx_h)
                        actual_w = min(sub.shape[1], spx_w)
                        full_arr[
                            row_px_start : row_px_start + actual_h,
                            col_px_start : col_px_start + actual_w,
                        ] = sub[:actual_h, :actual_w]
                except Exception as exc:
                    failed += 1
                    dem_progress.write(
                        f"  WARN: sub-tile {tile_idx}/{n_tiles} failed ({exc}); "
                        "affected area will use per-point HTTP queries."
                    )
                finally:
                    dem_progress.set_postfix(failed=failed, refresh=False)
                    dem_progress.update()

    if failed == n_tiles:
        print("  WARN: all sub-tiles failed; will use per-point HTTP queries.")
        _dem_tile = None
        return

    _dem_tile = {
        "array": full_arr,
        "xmin": xmin,
        "ymax": ymax,
        "cell_w": cell_w,
        "cell_h": cell_h,
    }
    res_lat_m = cell_h * 111_320.0
    res_lon_m = cell_w * 111_320.0 * math.cos(math.radians(mid_lat))
    res_m = (res_lat_m + res_lon_m) / 2.0
    status = f"({failed} sub-tile(s) failed)" if failed else "all sub-tiles OK"
    print(
        f"  3DEP tile loaded: {full_arr.shape[0]}\u00d7{full_arr.shape[1]} px, "
        f"resolution ~{res_m:.1f} m/px "
        f"({res_lon_m:.1f} m E-W \u00d7 {res_lat_m:.1f} m N-S), "
        f"bbox [{xmin:.5f}, {ymin:.5f}, {xmax:.5f}, {ymax:.5f}] – {status}"
    )


def load_dem_from_file(dem_path: str | Path) -> None:
    """
    Load a local DEM raster (GeoTIFF or any arcpy-readable format) into the
    module-level ``_dem_tile`` cache, replacing any previously downloaded tile.

    The raster **must be WGS 84 / EPSG:4326**. Projected and other geographic
    coordinate systems are rejected because the sampling cache stores degrees.

    Resolution order for reading the file:
      1. arcpy (always available in the ArcGIS Pro environment)
      2. rasterio
      3. GDAL (osgeo)

    Raises ``RuntimeError`` if none of the above libraries are importable or
    the file cannot be read.
    """
    global _dem_tile
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    print(f"Loading DEM from file: {dem_path} …")

    # ------------------------------------------------------------------
    # Attempt 1: arcpy (guaranteed in ArcGIS Pro environment)
    # ------------------------------------------------------------------
    try:
        import arcpy  # type: ignore

        ras = arcpy.sa.Raster(str(dem_path))
        desc = arcpy.Describe(str(dem_path))
        ext = desc.extent

        # The cache indexes directly by WGS84 longitude/latitude.
        sr = desc.spatialReference
        if not sr or sr.type != "Geographic" or getattr(sr, "factoryCode", None) != 4326:
            name = getattr(sr, "name", "unknown")
            raise ValueError(f"DEM must use WGS 84 (EPSG:4326); found {name!r}.")

        # Read pixel array; nodata → NaN
        arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype(np.float32)
        # arcpy returns row 0 = north (ymax)
        xmin = float(ext.XMin)
        ymax = float(ext.YMax)
        cell_w = float(desc.meanCellWidth)
        cell_h = float(desc.meanCellHeight)

        _dem_tile = {
            "array": arr,
            "xmin": xmin,
            "ymax": ymax,
            "cell_w": cell_w,
            "cell_h": cell_h,
        }
        print(
            f"  DEM loaded via arcpy: {arr.shape[0]}\u00d7{arr.shape[1]} px, "
            f"cell {cell_w:.6f}\u00b0 \u00d7 {cell_h:.6f}\u00b0, "
            f"bbox [{xmin:.5f}, {ymax - arr.shape[0] * cell_h:.5f}, "
            f"{xmin + arr.shape[1] * cell_w:.5f}, {ymax:.5f}]"
        )
        return
    except ValueError:
        raise
    except Exception as exc:
        print(f"  arcpy load failed ({exc}); trying rasterio …")

    # ------------------------------------------------------------------
    # Attempt 2: rasterio
    # ------------------------------------------------------------------
    try:
        import rasterio  # type: ignore

        with rasterio.open(str(dem_path)) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise ValueError(f"DEM must use WGS 84 (EPSG:4326); found {ds.crs!r}.")
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            t = ds.transform  # affine transform
            xmin = float(t.c)  # west edge
            ymax = float(t.f)  # north edge
            cell_w = float(abs(t.a))  # pixel width  (positive)
            cell_h = float(abs(t.e))  # pixel height (positive)

        _dem_tile = {
            "array": arr,
            "xmin": xmin,
            "ymax": ymax,
            "cell_w": cell_w,
            "cell_h": cell_h,
        }
        print(
            f"  DEM loaded via rasterio: {arr.shape[0]}\u00d7{arr.shape[1]} px, "
            f"cell {cell_w:.6f}\u00b0 \u00d7 {cell_h:.6f}\u00b0, "
            f"bbox [{xmin:.5f}, {ymax - arr.shape[0] * cell_h:.5f}, "
            f"{xmin + arr.shape[1] * cell_w:.5f}, {ymax:.5f}]"
        )
        return
    except ValueError:
        raise
    except Exception as exc:
        print(f"  rasterio load failed ({exc}); trying GDAL …")

    # ------------------------------------------------------------------
    # Attempt 3: GDAL (osgeo)
    # ------------------------------------------------------------------
    try:
        from osgeo import gdal  # type: ignore

        gdal.UseExceptions()
        ds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError("GDAL returned None – unrecognised format.")
        projection = ds.GetProjection()
        from osgeo import osr  # type: ignore

        spatial_reference = osr.SpatialReference(wkt=projection)
        authority = spatial_reference.GetAuthorityCode(None)
        if authority != "4326":
            raise ValueError(
                f"DEM must use WGS 84 (EPSG:4326); found EPSG:{authority or 'unknown'}."
            )
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr[arr == nodata] = np.nan
        gt = ds.GetGeoTransform()  # (xmin, cell_w, 0, ymax, 0, -cell_h)
        xmin = float(gt[0])
        ymax = float(gt[3])
        cell_w = float(abs(gt[1]))
        cell_h = float(abs(gt[5]))
        ds = None  # close

        _dem_tile = {
            "array": arr,
            "xmin": xmin,
            "ymax": ymax,
            "cell_w": cell_w,
            "cell_h": cell_h,
        }
        print(
            f"  DEM loaded via GDAL: {arr.shape[0]}\u00d7{arr.shape[1]} px, "
            f"cell {cell_w:.6f}\u00b0 \u00d7 {cell_h:.6f}\u00b0, "
            f"bbox [{xmin:.5f}, {ymax - arr.shape[0] * cell_h:.5f}, "
            f"{xmin + arr.shape[1] * cell_w:.5f}, {ymax:.5f}]"
        )
        return
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DEM '{dem_path}' with arcpy, rasterio, or GDAL. Last error: {exc}"
        ) from exc


def prefetch_dem_for_images(
    image_paths: list,
    vendor: str = "skydio",
) -> None:
    """
    Read GPS coordinates from all images via a single ExifTool batch call, then
    call prefetch_dem_tile() to download one DEM raster covering the whole flight.

    Call this once before the main per-image processing loop so that every
    subsequent _get_terrain_elevation() call is served from the in-memory tile.
    """
    et_path = _exiftool_path()
    str_paths = [str(p) for p in image_paths]

    try:
        with exiftool.ExifToolHelper(executable=et_path) as et:
            meta_list = et.get_tags(
                str_paths,
                tags=[
                    "EXIF:GPSLatitude",
                    "EXIF:GPSLatitudeRef",
                    "EXIF:GPSLongitude",
                    "EXIF:GPSLongitudeRef",
                ],
            )
    except Exception as exc:
        print(f"  WARN: ExifTool batch GPS read failed ({exc}); skipping DEM prefetch.")
        return

    coords: list[tuple[float, float]] = []
    for meta in meta_list:
        try:
            lat_raw = float(meta.get("EXIF:GPSLatitude", 0.0) or 0.0)
            lon_raw = float(meta.get("EXIF:GPSLongitude", 0.0) or 0.0)
            lat_ref = str(meta.get("EXIF:GPSLatitudeRef", "N"))
            lon_ref = str(meta.get("EXIF:GPSLongitudeRef", "E"))
            lat = _signed_gps(lat_raw, lat_ref)
            lon = _signed_gps(lon_raw, lon_ref)
            if lat != 0.0 or lon != 0.0:
                coords.append((lat, lon))
        except Exception:
            continue

    if not coords:
        print("  WARN: No valid GPS coordinates found in images; skipping DEM prefetch.")
        return

    prefetch_dem_tile(coords)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _exiftool_path() -> str:
    """Return ExifTool binary path, preferring EXIFTOOL_PATH env var."""
    env = os.environ.get("EXIFTOOL_PATH")
    if env:
        return env
    # Common install locations
    candidates = [
        r"C:\Windows\exiftool.exe",
        r"C:\Program Files\ExifTool\exiftool.exe",
        "exiftool",  # on PATH
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "exiftool"


def extract_metadata(image_path: str | Path) -> dict:
    """Read one image's metadata through ExifTool."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    with exiftool.ExifToolHelper(executable=_exiftool_path()) as et:
        meta_list = et.get_metadata([str(image_path)])
    if not meta_list:
        raise MetadataValidationError(f"ExifTool returned no metadata for {image_path}.")
    return meta_list[0]


def has_usable_relative_altitude(meta: dict) -> bool:
    """Return whether metadata contains a positive vendor-relative altitude."""
    if _vendor_from_make(meta) == "skydio":
        raw = meta.get("XMP:CameraPositionFLUZ")
        if raw in (None, ""):
            return False
        try:
            return float(str(raw).split(",")[-1]) > 0
        except (TypeError, ValueError):
            return False
    try:
        return float(meta.get("XMP:RelativeAltitude", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def camera_from_metadata(meta: dict, image_path: str | Path) -> Camera:
    """Validate an ExifTool record and construct a trustworthy camera model."""
    image_path = Path(image_path)
    vendor = _vendor_from_make(meta)

    width = _optional_number(meta, "EXIF:ImageWidth", "File:ImageWidth", positive=True)
    height = _optional_number(meta, "EXIF:ImageHeight", "File:ImageHeight", positive=True)
    if width is None or height is None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise MetadataValidationError(
                f"Could not determine dimensions for {image_path.name}; image tags are missing "
                "and OpenCV could not read the image."
            )
        img_h, img_w = image.shape[:2]
    else:
        img_w, img_h = int(width), int(height)

    lat, lon = _gps(meta)
    abs_alt = _optional_number(meta, "XMP:AbsoluteAltitude", "EXIF:GPSAltitude") or 0.0
    mag_declination = 0.0

    if vendor == "skydio":
        fx = _number(meta, "XMP:CalibratedFocalLengthX", positive=True)
        fy = _number(meta, "XMP:CalibratedFocalLengthY", positive=True)
        roll_deg, pitch_deg, yaw_deg = _orientation(
            meta,
            (
                "XMP:CameraOrientationNEDRoll",
                "XMP:CameraOrientationNEDPitch",
                "XMP:CameraOrientationNEDYaw",
            ),
        )
        flu_z = meta.get("XMP:CameraPositionFLUZ")
        try:
            relative_alt = float(str(flu_z).split(",")[-1]) if flu_z not in (None, "") else 0.0
        except (TypeError, ValueError) as exc:
            raise MetadataValidationError(
                f"XMP:CameraPositionFLUZ must end with a numeric altitude; got {flu_z!r}."
            ) from exc
        msl_alt = _optional_number(meta, "XMP:GpsMslHeight") or abs_alt
        rel_alt = _compute_true_agl(lat, lon, msl_alt, relative_alt)
        dewarp = str(meta.get("XMP:DewarpData") or "")
        dist_coeffs = _parse_distortion_dewarp(dewarp) if dewarp else np.zeros(5)
    else:
        roll_deg, pitch_deg, yaw_deg = _orientation(
            meta,
            ("XMP:GimbalRollDegree", "XMP:GimbalPitchDegree", "XMP:GimbalYawDegree"),
        )
        calibrated = _optional_number(meta, "XMP:CalibratedFocalLength", positive=True)
        if vendor == "dji" and calibrated is not None:
            fx = fy = calibrated
        else:
            flen_35mm = _number(meta, "EXIF:FocalLengthIn35mmFormat", positive=True)
            flen = _number(meta, "EXIF:FocalLength", positive=True)
            sensor_w, sensor_h = _sensor_dims_from_35mm(flen_35mm, flen, img_w, img_h)
            fx = flen * img_w / sensor_w
            fy = flen * img_h / sensor_h
        relative_alt = _optional_number(meta, "XMP:RelativeAltitude") or 0.0
        rel_alt = _compute_true_agl(lat, lon, abs_alt, relative_alt)
        dist_coeffs = np.zeros(5)
        if vendor == "dji":
            mag_declination = _compute_mag_declination(vendor, (lat, lon), abs_alt)
            yaw_deg = (yaw_deg + mag_declination) % 360.0

    cx = _optional_number(meta, "XMP:CalibratedOpticalCenterX")
    cy = _optional_number(meta, "XMP:CalibratedOpticalCenterY")
    cx = img_w / 2 if cx is None else cx
    cy = img_h / 2 if cy is None else cy

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------------------------
    # Rotation matrix: camera → world (NED-like, y-down)
    #
    # Convention: extrinsic "xyz" rotations — first pitch around fixed X,
    # then yaw around fixed Y, then roll around fixed Z.
    # Equivalent matrix: Rz(roll) @ Ry(yaw) @ Rx(pitch).
    # -----------------------------------------------------------------------
    R = Rotation.from_euler(
        "xyz",
        [pitch_deg, yaw_deg, roll_deg],
        degrees=True,
    ).as_matrix()

    return Camera(
        image_path=image_path,
        K=K,
        dist_coeffs=dist_coeffs,
        R=R,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg,
        gps_coords=(lat, lon),
        rel_alt=rel_alt,
        abs_alt=abs_alt,
        image_width=img_w,
        image_height=img_h,
        vendor=vendor,
        mag_declination=mag_declination,
    )


def load_camera(image_path: str | Path, vendor: str = "generic") -> Camera:
    """Read, validate, and parse one image; *vendor* is retained for compatibility."""
    del vendor
    return camera_from_metadata(extract_metadata(image_path), image_path)
