# Third-party dependency review

This is a preliminary inventory for release review, not legal advice. Package
metadata can be incomplete or ambiguous; the license text shipped by each
upstream project controls. Record legal approval in `QUALIFICATION.md` before
publishing a release.

## Python overlay

The following versions are emitted by both Windows lock files. License values
were read from installed wheel metadata on 2026-08-26.

| Package | Version | Declared metadata | Review status |
| --- | --- | --- | --- |
| colorama | 0.4.6 | BSD-3-Clause | Verify upstream license file |
| geographiclib | 2.1 | MIT | Review text |
| geopy | 2.4.1 | MIT | Review text |
| opencv-python | 4.10.0.84 | MIT / Apache-2.0 | Review bundled OpenCV notices |
| pyexiftool | 0.5.6 | GPLv3+ / BSD | Resolve applicable alternative and distribution obligations |
| pygeomag | 1.1.0 | MIT | Review text |
| shapely | 2.1.2 | BSD-3-Clause | Review bundled GEOS notices |
| tqdm | 4.67.1 | MPL-2.0 AND MIT | Review dual-component notices |

NumPy, SciPy, Pillow, and ArcPy are supplied by the client's licensed ArcGIS
Pro environment and are deliberately absent from the overlay locks. Their
redistribution terms still matter if a release artifact ever bundles an Esri
environment; current releases must not do so.

## External runtime services and tools

- ArcGIS Pro / ArcPy: proprietary Esri software installed and licensed by the
  client; never redistribute its environment with this repository.
- ExifTool: external executable installed separately by the client. Confirm
  its Perl Artistic/GPL licensing and notice requirements against the exact
  distribution clients are directed to install.
- USGS 3DEP: remote elevation service used when no local DEM is supplied.
  Review service terms and attribution requirements before release.

## Release evidence

Before tagging a release:

1. Review upstream license files for every row above, especially entries with
   missing or alternative metadata.
2. Record reviewer, date, conclusions, and required notices in
   `QUALIFICATION.md`.
3. Re-run the inventory whenever `requirements.txt` or either lock changes.
4. Confirm the release ZIP contains source and lock files only, with no ArcGIS
   environment or ExifTool binaries.