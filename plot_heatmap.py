"""
plot_heatmap.py
---------------
Plot a heatmap of camera-footprint intersections.

For every pixel in a regular lon/lat grid the script counts how many camera
frustum footprints overlap that location, then renders the result as a colour
heatmap with semi-transparent footprint outlines overlaid.

Footprint geometry is computed exactly as in debug_kml.py (same ray-cast
pipeline), so the result is consistent with the KML export.

Algorithm
---------
1. Load every .JPG in *images* and project its image boundary onto the ground
   plane to get a lon/lat Polygon (same pipeline as debug_kml.py).
2. Build a (resolution × resolution) lon/lat grid over the bounding box of all
   footprints.
3. For each polygon use ``matplotlib.path.Path.contains_points`` to mark which
   grid pixels fall inside it (fully vectorised – no per-pixel Python loop).
4. Accumulate the boolean masks into an integer count array.
5. Render with ``imshow`` + a ``PolyCollection`` overlay.

Usage
-----
  python plot_heatmap.py \\
      --images  "C:/path/to/images" \\
      [--vendor  skydio|dji|generic]  \\
      [--resolution 500]              \\
      [--output  heatmap.png]         \\
      [--no_outlines]                 \\
      [--threads N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# Make the local ``src`` package importable and reuse _load_footprints.
sys.path.insert(0, str(Path(__file__).parent))
from debug_kml import _load_footprints

# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------


def build_density_grid(
    footprints: list[dict],
    resolution: int = 500,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Rasterise footprint polygons onto a (resolution × resolution) grid.

    Parameters
    ----------
    footprints:
        List of ``{"path": Path, "coords": [(lon, lat), …]}`` dicts as
        returned by ``_load_footprints``.
    resolution:
        Number of grid cells per axis.

    Returns
    -------
    count : np.ndarray, shape (resolution, resolution), dtype int32
        Number of footprints that overlap each grid cell.  Row 0 is the
        southernmost latitude (``origin="lower"`` convention).
    extent : (lon_min, lon_max, lat_min, lat_max)
        Geographic extent of the grid, suitable for passing directly to
        ``imshow(extent=…)``.
    """
    from matplotlib.path import Path as MPath

    # ---- bounding box --------------------------------------------------------
    all_lons = [c[0] for fp in footprints for c in fp["coords"]]
    all_lats = [c[1] for fp in footprints for c in fp["coords"]]
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    # 5 % padding so edge footprints aren't clipped
    dlon = (lon_max - lon_min) * 0.05 or 1e-4
    dlat = (lat_max - lat_min) * 0.05 or 1e-4
    lon_min -= dlon
    lon_max += dlon
    lat_min -= dlat
    lat_max += dlat

    # ---- sample-point grid ---------------------------------------------------
    lons = np.linspace(lon_min, lon_max, resolution)
    lats = np.linspace(lat_min, lat_max, resolution)
    lon_g, lat_g = np.meshgrid(lons, lats)  # each (res, res)
    flat_pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])  # (res², 2)

    # ---- rasterise -----------------------------------------------------------
    count = np.zeros(resolution * resolution, dtype=np.int32)
    for fp in footprints:
        path = MPath(np.array(fp["coords"]))  # automatically closed
        inside = path.contains_points(flat_pts)
        count += inside.view(np.uint8)

    return count.reshape(resolution, resolution), (lon_min, lon_max, lat_min, lat_max)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmap(
    count: np.ndarray,
    extent: tuple[float, float, float, float],
    footprints: list[dict],
    show_outlines: bool = True,
    out_path: Path | None = None,
) -> None:
    """Render the density grid and optionally save to *out_path*."""
    lon_min, lon_max, lat_min, lat_max = extent

    fig, ax = plt.subplots(figsize=(13, 10))

    # ---- density heatmap -----------------------------------------------------
    im = ax.imshow(
        count,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="YlOrRd",
        aspect="auto",
        interpolation="bilinear",
        vmin=0,
    )

    # ---- footprint outlines --------------------------------------------------
    if show_outlines and footprints:
        verts = [np.array(fp["coords"]) for fp in footprints]
        col = PolyCollection(
            verts,
            facecolor="none",
            edgecolor="#1a6faf",
            alpha=0.35,
            linewidth=0.7,
        )
        ax.add_collection(col)

    # ---- colour bar ----------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Overlapping footprints (count)", fontsize=11)

    # ---- labels & annotation -------------------------------------------------
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Camera Footprint Intersection Heatmap", fontsize=14, fontweight="bold")

    covered = int((count > 0).sum())
    total = count.size
    pct = 100.0 * covered / total
    stats_txt = (
        f"Images : {len(footprints)}\n"
        f"Max overlap : {int(count.max())}\n"
        f"Mean overlap (covered): {count[count > 0].mean():.1f}\n"
        f"Area covered : {pct:.1f} % of grid"
    )
    ax.text(
        0.015,
        0.985,
        stats_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap of camera footprint intersections."
    )
    parser.add_argument(
        "--images",
        default=r"C:\Users\prna006\datasets\dominion_test_images_poles",
        help="Directory containing source .JPG images.",
    )
    parser.add_argument(
        "--vendor",
        default="skydio",
        choices=["skydio", "dji", "generic"],
        help="Drone vendor for EXIF parsing.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help="Grid resolution (pixels per side).  Default: 500.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the PNG, e.g. heatmap.png.",
    )
    parser.add_argument(
        "--no_outlines",
        action="store_true",
        help="Suppress footprint-outline overlay.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Worker threads for parallel EXIF loading (default: auto).",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    out_path = Path(args.output) if args.output else None

    print("Computing frustum footprints …")
    footprints = _load_footprints(image_dir, args.vendor, max_workers=args.threads)
    if not footprints:
        print("No footprints computed – aborting.")
        return

    print(
        f"Rasterising {len(footprints)} footprint(s) onto a "
        f"{args.resolution}×{args.resolution} grid …"
    )
    count, extent = build_density_grid(footprints, resolution=args.resolution)
    print(
        f"  max overlap : {int(count.max())}   mean (covered cells) : {count[count > 0].mean():.1f}"
    )

    plot_heatmap(
        count,
        extent,
        footprints,
        show_outlines=not args.no_outlines,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
