# Rules-based pole tagging

ArcGIS Pro script pipeline that projects drone-image footprints and associates
visible utility poles with source images. The repository scripts—not a Python
wheel—are the supported delivery: clone the repo (or download a release
archive) and run `python main.py` from the repository root.

The project is licensed under the [BSD 3-Clause License](LICENSE). The
third-party dependencies in `requirements.txt` and the lock files keep their
own licenses; the full dependency list and Esri's [ArcPy library
policy](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/available-python-libraries.htm)
apply to the environments you build.

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input contract and preflight](#input-contract-and-preflight)
- [Run](#run)
- [Verification and testing](#verification-and-testing)
- [Closest pole per image](#closest-pole-per-image)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Windows with licensed ArcGIS Pro 3.4–3.7.
- Python 3.11 (from ArcGIS Pro 3.4/3.5) or Python 3.13 (from 3.6/3.7), via the
  `arcgispro-py3` Conda environment shipped with ArcGIS Pro. The environment's
  bundled `pip` is sufficient; no other Python package tools are required.
- [ExifTool for Windows](https://exiftool.org), either on `PATH`
  or named by the `EXIFTOOL_PATH` environment variable. Keep `exiftool_files`
  beside the executable when using the full Windows distribution.
- An internet connection to PyPI for the one-time dependency installation. If
  you do not pass `--dem`, terrain queries also need HTTPS access to
  `elevation.nationalmap.gov` (USGS 3DEP).

The end-to-end pipeline has been tested with ArcGIS Pro 3.7 and Python 3.13.
Other versions in the supported range have matching runtime contracts and lock
files but should be smoke-tested on the ArcGIS patch used for delivery.

## Installation

### ArcGIS Python Command Prompt (cmd.exe)

```bat
conda create --clone arcgispro-py3 --name pole-tagging --pinned
activate pole-tagging
python -m pip install --no-deps --require-hashes -r requirements-win-py311.lock
python main.py --check-env
```

For ArcGIS Pro 3.6/3.7, use `requirements-win-py313.lock` in the install
command. `--check-env` prints the active ArcGIS, Python, Esri-managed package,
third-party package, and ExifTool versions and exits nonzero on a mismatch.
Both commands above must exit `0` before the first production run.

Notes:

- Run these commands from the repository root so `main.py` and the TOML
  config resolve as documented.
- The lock files intentionally exclude NumPy, SciPy, and Pillow because Esri
  manages those versions per ArcGIS release.

### Setting `EXIFTOOL_PATH`

If ExifTool is not on `PATH` (or a check still says `ExifTool: NOT FOUND`
after an install), point the environment variable at the executable. To make it permanent, use **System Properties → Advanced → Environment
Variables** (or `setx EXIFTOOL_PATH "C:\...\exiftool.exe"`) and then open a
new terminal. Verify with `exiftool -ver` and `python main.py --check-env`.

## Configuration

The bundled `pole_tagging.toml` holds optional project defaults. For a
project, uncomment and edit the entries that apply (paths, the pole height
field, DEM location, debug switches, fallback tuning). Precedence is:

1. Command-line arguments
2. Values in `pole_tagging.toml`
3. Built-in defaults in code

Only `data_path` is required, and it can come from either the CLI or the TOML
file. Key fields:

- `[defaults] data_path`: the pole point feature class or shapefile.
- `[defaults] height_field` (default `HEIGHT`): pole heights are **feet** and
  are converted to metres; missing/nonnumeric/zero/negative heights fall back
  to `default_pole_height_m` with a warning.
- `[defaults] vendor`: **informational only**. Vendor routing is auto-detected
  from each image's `EXIF:Make` tag (`DJI` → dji, `Skydio` → skydio, anything
  else → generic). The option is retained for compatibility and does not
  change parsing.
- `[projection]`: footprint extent (`max_range_m`), nadir threshold, and
  pixel margin.
- `[fallback]`: multi-image consensus matching for images with no primary
  match. The bundled TOML enables it with a 20 m camera/pole limit and 20 m
  snap distance; `enabled = false` disables it. If the bundled TOML is absent,
  conservative code defaults disable fallback and use a 12 m snap distance.

## Input contract and preflight

The image directory is scanned non-recursively for `.jpg` files, case
insensitively. Vendor routing comes from `EXIF:Make`.

- All images require readable dimensions, GPS latitude/longitude plus valid
  N/S and E/W references, finite orientation values, positive focal-length
  inputs, and a positive AGL source.
- DJI requires gimbal roll, pitch, and yaw plus calibrated focal length or both
  EXIF focal length and 35 mm equivalent focal length.
- Skydio requires NED roll, pitch, and yaw plus calibrated X/Y focal length.
- Generic cameras require gimbal roll, pitch, and yaw plus both EXIF focal
  lengths. Camera-specific correctness is the operator's responsibility.
- A positive vendor-relative altitude is preferred. Otherwise a positive
  absolute/MSL altitude must combine with terrain elevation. Missing values
  are errors; they are never silently replaced with zero.

The pole input must be a nonempty ArcGIS point feature class or complete
shapefile with a defined spatial reference, object ID, and valid coordinates.

A local DEM must be readable WGS 84 (EPSG:4326). Projected rasters and other
geographic CRSs are rejected.

Validate everything without writing output:

```powershell
python main.py --check-inputs `
  --image_dir C:/survey/images `
  --data_path C:/survey/poles.shp `
  --dem C:/survey/dem_wgs84.tif
```

Preflight exits `0` only when the environment, dataset, DEM, and every image
are usable. It never creates outputs.

## Run

Create the output and CSV parent directories first. The source dataset is
never modified. Quote arguments when your paths contain spaces:

```powershell
python main.py `
  --image_dir "C:/surveys/fall26/images" `
  --data_path "C:/surveys/fall26/poles.shp" `
  --output "C:/surveys/fall26/results/poles_tagged.shp" `
  --dem "C:/surveys/fall26/dem_wgs84.tif"
```

Use `--overwrite` to replace derived outputs. By default, valid images are
committed and invalid images are recorded as `MetadataValidationError` (or the
specific processing error) in the failure CSV. Use `--require-complete` to
write nothing when any image fails.

Exit codes:

- `0`: every image processed and all requested outputs succeeded.
- `1`: fatal or strict-mode failure; no new primary output is committed.
- `2`: usable degraded output due to image failures or optional debug-KML
  failure.

For `poles_tagged.shp`, outputs are:

- `poles_tagged.shp`, with `match_cnt` and, when format limits permit,
  `match_imgs` and `img_dists`.
- `poles_tagged_matches.csv`, the authoritative one-row-per-match result.
- `poles_tagged_matches_failures.csv`, only for partial runs.

The match CSV columns are `pole_oid`, `pole_lat`, `pole_lon`, `image_name`,
`distance_m`, and `match_source`. Shapefile text fields are omitted if their
joined values exceed 254 characters; the normalized CSV remains complete.
On commit failure, staged outputs and any backup copies of the previous
output are preserved in the output directory so nothing is lost; inspect those
files before rerunning.

## Closest pole per image

```powershell
python tools/export_image_to_pole_csv.py `
  --matches_csv C:/survey/results/poles_tagged_matches.csv `
  --data_path C:/survey/results/poles_tagged.shp `
  --output C:/survey/results/image_to_pole.csv
```

This supported exporter uses ArcPy and writes `image_name`, `pole_oid`,
`distance_m`, `pole_lat`, `pole_lon`, and `pole_attributes_json`.

`debug_kml.py` and `plot_heatmap.py` are retained as unsupported development
utilities. Their extra dependencies, behavior, and outputs are outside the
handoff dependency contract and acceptance tests.

## Troubleshooting

- `ExifTool: NOT FOUND`: run `python main.py --check-env` to see the resolved
  path. If ExifTool is in a non-default location, set `EXIFTOOL_PATH` to the
  executable (see above) and open a new terminal.
- `ExifTool: FAIL`: `exiftool_files` must sit beside the executable for the
  full Windows distribution; run `exiftool -ver` on its own to narrow it down.
- `ArcPy runtime: UNSUPPORTED`: use the ArcGIS Python Command Prompt with the
  pinned cloned environment, not a standalone Python.
- Environment mismatch on `--check-env`: recreate the clone from the matching
  ArcGIS release and reinstall the correct lock file for its Python
  generation (3.11 for Pro 3.4/3.5, 3.13 for 3.6/3.7).
- Metadata failures: run `exiftool -G1 -s image.jpg` and compare the tags
  with the vendor contract above.
- DEM rejection: project the raster to WGS 84 (EPSG:4326); defining the CRS
  without actually projecting pixels is not sufficient.
- `Output parent directory is not writable`: the destination parent must
  exist and be writable by your account (a standard user account cannot
  usually write to `C:\Program Files\...`). Move the output under your
  profile or a shared project drive; check folder permissions/ACLs if the
  path should be writable.
- Output exists from an earlier run: rerun with `--overwrite`, or point
  `--output`/CSV paths at a new location.
- Terrain queries fail without a DEM: allow outbound HTTPS to
  `elevation.nationalmap.gov`, or pass a local WGS 84 DEM with `--dem`.
- Rollback/commit confusion: source pole data is never modified; staged or
  backup outputs left in the output directory after a failed commit are safe
  to inspect, move, or delete.
