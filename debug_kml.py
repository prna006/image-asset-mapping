"""
debug_kml.py
------------
Debugging utility: reads the matched output shapefile and the source image
directory, then writes a KML file viewable in Google Earth containing:

  * Poles folder  – green pins for matched poles, red for unmatched.
                    Each placemark description lists the image filenames
                    stored in the match_imgs field.

  * Images folder – yellow camera pins at each image's GPS location.
                    Description shows filename, pitch, yaw, and altitude.


  * Footprints folder (--footprints flag) – semi‑transparent polygons
                    showing each image's ground frustum.

Usage
-----
  python debug_kml.py \
      --shp   "C:/path/to/RBB_Poles_matched.shp" \
      --images "C:/path/to/images" \
      --output "debug.kml" \
      [--vendor skydio] \
      [--footprints] \

      [--img_field match_imgs] \
      [--threads N]          # optional: number of worker threads
      [--heatmap]            # add footprint-intersection heatmap as GroundOverlay
      [--hm_resolution N]    # heatmap grid resolution (default 500)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import xml.etree.ElementTree as ET
from pathlib import Path

from src.camera import load_camera
from src.debug_kml_export import (
    build_footprint_records,
    build_image_record,
    build_kml,
    write_kml,
)
from src.frustum import get_footprint

try:
    from tqdm import tqdm

    _tqdm = tqdm
except ImportError:

    def _tqdm(x, **kwargs):
        return x  # fallback: no progress bar


try:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")  # non-interactive backend – safe when arcpy is loaded
    import matplotlib.colors as _mcolors
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as _MPath

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------


def build_density_grid(
    footprints: list[dict],
    resolution: int = 500,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Rasterise footprint polygons onto a (resolution × resolution) grid.

    Returns
    -------
    count : np.ndarray shape (resolution, resolution) float64
        Sum of abs(pitch_deg) weights for footprints overlapping each grid
        cell (row 0 = south).  Cells with no coverage are exactly 0.0.
    extent : (lon_min, lon_max, lat_min, lat_max)
    """
    if not _HAS_MPL:
        raise RuntimeError("numpy/matplotlib are required for heatmap generation.")

    all_lons = [c[0] for fp in footprints for c in fp["coords"]]
    all_lats = [c[1] for fp in footprints for c in fp["coords"]]
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    dlon = (lon_max - lon_min) * 0.05 or 1e-4
    dlat = (lat_max - lat_min) * 0.05 or 1e-4
    lon_min -= dlon
    lon_max += dlon
    lat_min -= dlat
    lat_max += dlat

    lons = np.linspace(lon_min, lon_max, resolution)
    lats = np.linspace(lat_min, lat_max, resolution)
    lon_g, lat_g = np.meshgrid(lons, lats)
    flat_pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])

    count = np.zeros(resolution * resolution, dtype=np.float64)
    for fp in footprints:
        weight = abs(fp.get("pitch", 1.0))
        path = _MPath(np.array(fp["coords"]))
        inside = path.contains_points(flat_pts)
        count += inside * weight

    return count.reshape(resolution, resolution), (lon_min, lon_max, lat_min, lat_max)


def _render_heatmap_png(
    count: np.ndarray,
    out_path: Path,
    cmap: str = "YlOrRd",
    alpha: float = 0.75,
) -> None:
    """
    Save *count* as a georeferenced-ready PNG:
      - Zero cells are fully transparent.
      - Non-zero cells use *cmap* at opacity *alpha*.
      - No axes, borders, or whitespace – pixel 0,0 maps to NW corner
        (north-up, ready for <GroundOverlay>).
    """
    if not _HAS_MPL:
        raise RuntimeError("numpy/matplotlib required for PNG rendering.")

    norm = _mcolors.Normalize(vmin=0, vmax=max(float(count.max()), 1e-9))
    if hasattr(matplotlib, "colormaps"):
        cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    else:
        cmap_obj = plt.get_cmap(cmap)

    # Build RGBA array; row 0 = south, flip vertically for north-up PNG.
    rgba = cmap_obj(norm(count.astype(float)))  # (H, W, 4), float 0-1
    rgba[count == 0, 3] = 0.0  # transparent background
    rgba[count > 0, 3] = alpha
    png_data = (rgba[::-1] * 255).astype(np.uint8)  # flip: row 0 → north
    plt.imsave(out_path, png_data, origin="upper", format="png")


def _add_ground_overlay(
    doc: ET.Element,
    png_filename: str,
    extent: tuple[float, float, float, float],
    max_overlap: float,
) -> None:
    """Append a <GroundOverlay> element to *doc* for the heatmap PNG."""
    lon_min, lon_max, lat_min, lat_max = extent
    go = ET.SubElement(doc, "GroundOverlay")
    ET.SubElement(go, "name").text = f"Footprint Heatmap (max pitch-weight: {max_overlap:.1f}°)"
    ET.SubElement(go, "drawOrder").text = "1"
    icon = ET.SubElement(go, "Icon")
    ET.SubElement(icon, "href").text = png_filename
    box = ET.SubElement(go, "LatLonBox")
    ET.SubElement(box, "north").text = f"{lat_max:.8f}"
    ET.SubElement(box, "south").text = f"{lat_min:.8f}"
    ET.SubElement(box, "east").text = f"{lon_max:.8f}"
    ET.SubElement(box, "west").text = f"{lon_min:.8f}"


# ---------------------------------------------------------------------------
# Read shapefile via arcpy
# ---------------------------------------------------------------------------


def _read_shapefile(shp_path: str, img_field: str) -> list[dict]:
    """
    Return a list of dicts: {oid, lon, lat, matched_imgs: str | None}.
    Falls back to an empty list if the field doesn't exist.
    """
    try:
        import arcpy  # type: ignore
    except ImportError:
        print("[warn] arcpy not found – trying shapefile via geopandas …")
        return _read_shapefile_gpd(shp_path, img_field)

    wgs84 = arcpy.SpatialReference(4326)
    fields = [f.name for f in arcpy.ListFields(shp_path)]
    have_field = img_field in fields
    cursor_fields = ["OID@", "SHAPE@XY"] + ([img_field] if have_field else [])

    poles = []
    with arcpy.da.SearchCursor(shp_path, cursor_fields, spatial_reference=wgs84) as cur:
        for row in cur:
            oid = row[0]
            lon, lat = row[1]
            imgs = row[2] if have_field else None
            poles.append({"oid": oid, "lon": lon, "lat": lat, "matched_imgs": imgs or ""})
    return poles


def _read_shapefile_gpd(shp_path: str, img_field: str) -> list[dict]:
    """Fallback reader using geopandas (no arcpy required)."""
    import geopandas as gpd  # type: ignore

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    poles = []
    for _, row in gdf.iterrows():
        poles.append(
            {
                "oid": row.get("FID", 0),
                "lon": row.geometry.x,
                "lat": row.geometry.y,
                "matched_imgs": str(row.get(img_field, "") or ""),
            }
        )
    return poles


# ---------------------------------------------------------------------------
# Load image/camera data – multithreaded
# ---------------------------------------------------------------------------
def _load_image_data(
    image_dir: Path,
    vendor: str,
    include_footprints: bool = False,
    max_workers: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load camera metadata once per image and optionally compute footprints."""
    image_paths = sorted({p for p in image_dir.glob("*") if p.suffix.lower() == ".jpg"})

    def _process_image(img_path: Path) -> tuple[dict, list[dict]] | None:
        try:
            cam = load_camera(img_path, vendor=vendor)
            image_record = build_image_record(img_path, cam)

            footprint_records: list[dict] = []
            if include_footprints:
                try:
                    fp = get_footprint(cam)
                    footprint_records = build_footprint_records(
                        img_path,
                        fp,
                        cam.pitch_deg,
                    )
                except Exception as exc:
                    print(f"  [warn footprint] {img_path.name}: {exc}")

            return image_record, footprint_records
        except Exception as exc:
            print(f"  [warn] {img_path.name}: {exc}")
            return None

    images: list[dict] = []
    footprints: list[dict] = []
    desc = "Reading EXIF + footprints" if include_footprints else "Reading EXIF"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in _tqdm(
            executor.map(_process_image, image_paths),
            total=len(image_paths),
            desc=desc,
            unit="img",
        ):
            if result is None:
                continue
            image_record, footprint_records = result
            images.append(image_record)
            footprints.extend(footprint_records)

    return images, footprints


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pole matches and image positions to KML for Google Earth."
    )
    parser.add_argument(
        "--shp",
        required=True,
        help="Path to the matched output shapefile.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing source .JPG images.",
    )
    parser.add_argument(
        "--output",
        default="debug_poles.kml",
        help="Output KML file path.",
    )
    parser.add_argument(
        "--vendor",
        default="skydio",
        choices=["skydio", "dji", "generic"],
        help="Drone vendor for EXIF parsing.",
    )
    parser.add_argument(
        "--img_field",
        default="match_imgs",
        help="Shapefile field containing matched image paths.",
    )
    parser.add_argument(
        "--footprints",
        action="store_true",
        help="Include frustum footprint polygons (slower – re‑runs ray‑cast).",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help=(
            "Add a footprint-intersection heatmap as a GroundOverlay "
            "(implies footprint computation)."
        ),
    )
    parser.add_argument(
        "--hm_resolution",
        type=int,
        default=500,
        help="Heatmap grid resolution (pixels per side, default: 500).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads for parallel processing (default: auto).",
    )
    args = parser.parse_args()

    shp_path = args.shp
    image_dir = Path(args.images)
    out_path = Path(args.output)
    vendor = args.vendor
    img_field = args.img_field
    threads = args.threads
    do_heatmap = args.heatmap

    print("Reading shapefile …")
    poles = _read_shapefile(shp_path, img_field)

    include_footprints = args.footprints or do_heatmap
    if include_footprints:
        print("Reading image EXIF and computing frustum footprints …")
    else:
        print("Reading image EXIF …")

    images, footprints = _load_image_data(
        image_dir,
        vendor,
        include_footprints=include_footprints,
        max_workers=threads,
    )

    print("Building KML …")
    kml = build_kml(poles, images, footprints, [], [], img_field, image_dir)

    if do_heatmap:
        if not _HAS_MPL:
            print("[warn] numpy/matplotlib not found – skipping heatmap GroundOverlay.")
        elif not footprints:
            print("[warn] No footprints available – skipping heatmap GroundOverlay.")
        else:
            print(f"Building heatmap ({args.hm_resolution}×{args.hm_resolution}) …")
            count, extent = build_density_grid(footprints, resolution=args.hm_resolution)
            png_path = out_path.with_name(out_path.stem + "_heatmap.png")
            _render_heatmap_png(count, png_path)
            # Find the <Document> element inside the returned <kml> root
            doc_el = kml.find("Document")
            if doc_el is None:
                doc_el = kml  # fallback
            _add_ground_overlay(doc_el, png_path.name, extent, float(count.max()))
            covered = count[count > 0]
            mean_covered = f"{covered.mean():.1f}°" if covered.size else "n/a"
            print(
                f"  Heatmap PNG : {png_path}\n"
                f"  Max pitch-weight : {count.max():.1f}°  "
                f"Mean (covered) : {mean_covered}"
            )

    write_kml(kml, out_path)
    print("Done. Open the KML file in Google Earth.")


if __name__ == "__main__":
    main()
