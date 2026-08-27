from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

_KML_NS = "http://www.opengis.net/kml/2.2"


def build_image_record(image_path: Path, camera) -> dict:
    """Return the KML image record for an already-loaded camera."""
    lat, lon = camera.gps_coords
    return {
        "path": image_path,
        "lon": lon,
        "lat": lat,
        "alt": camera.rel_alt,
        "pitch": camera.pitch_deg,
        "yaw": camera.yaw_deg,
    }


def footprint_parts(geometry) -> list[list[tuple[float, float]]]:
    """Return one coordinate ring per polygon part in (lon, lat) order."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [[(lon, lat) for lon, lat in geometry.exterior.coords]]
    if geometry.geom_type == "MultiPolygon":
        return [
            [(lon, lat) for lon, lat in poly.exterior.coords]
            for poly in geometry.geoms
            if not poly.is_empty
        ]
    return []


def build_footprint_records(image_path: Path, footprint, pitch_deg: float) -> list[dict]:
    """Return zero or more KML footprint records from a projected geometry."""
    rings = footprint_parts(footprint)
    records: list[dict] = []
    for index, coords in enumerate(rings, start=1):
        part_name = image_path.name if len(rings) == 1 else f"{image_path.name} (part {index})"
        records.append(
            {
                "path": image_path,
                "name": part_name,
                "coords": coords,
                "pitch": pitch_deg,
            }
        )
    return records


def build_debug_poles(
    poles: list[dict],
    oid_to_paths: dict[int, list[str]],
) -> list[dict]:
    """Return pole records decorated with current-run matched image names."""
    debug_poles: list[dict] = []
    for pole in poles:
        matched_imgs = ";".join(oid_to_paths.get(pole["oid"], []))
        debug_poles.append({**pole, "matched_imgs": matched_imgs})
    return debug_poles


def _kml_root() -> ET.Element:
    return ET.Element("kml", xmlns=_KML_NS)


def _folder(parent: ET.Element, name: str) -> ET.Element:
    folder = ET.SubElement(parent, "Folder")
    ET.SubElement(folder, "name").text = name
    return folder


def _placemark_point(
    parent: ET.Element,
    name: str,
    lon: float,
    lat: float,
    description: str = "",
    style_url: str = "",
    heading: float | None = None,
    cam_matched: bool = False,
) -> None:
    pm = ET.SubElement(parent, "Placemark")
    ET.SubElement(pm, "name").text = name
    if description:
        desc_el = ET.SubElement(pm, "description")
        desc_el.text = description

    if heading is not None and style_url == "#camera_heading":
        color = "ff00ff00" if cam_matched else "ff00ffff"
        style = ET.SubElement(pm, "Style")
        icon_style = ET.SubElement(style, "IconStyle")
        ET.SubElement(icon_style, "color").text = color
        ET.SubElement(icon_style, "scale").text = "1.2"
        ET.SubElement(icon_style, "heading").text = f"{heading:.1f}"
        icon = ET.SubElement(icon_style, "Icon")
        ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/arrow.png"
    elif style_url:
        ET.SubElement(pm, "styleUrl").text = style_url

    pt = ET.SubElement(pm, "Point")
    ET.SubElement(pt, "coordinates").text = f"{lon},{lat},0"


def _placemark_polygon(
    parent: ET.Element,
    name: str,
    coords: list[tuple[float, float]],
    description: str = "",
    style_url: str = "",
) -> None:
    pm = ET.SubElement(parent, "Placemark")
    ET.SubElement(pm, "name").text = name
    if description:
        desc_el = ET.SubElement(pm, "description")
        desc_el.text = description
    if style_url:
        ET.SubElement(pm, "styleUrl").text = style_url
    poly = ET.SubElement(pm, "Polygon")
    outer = ET.SubElement(poly, "outerBoundaryIs")
    ring = ET.SubElement(outer, "LinearRing")
    closed = list(coords) + [coords[0]]
    ET.SubElement(ring, "coordinates").text = " ".join(f"{lon},{lat},0" for lon, lat in closed)


def _add_styles(doc: ET.Element) -> None:
    styles = [
        ("pole_matched", "ff00aa00", "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"),
        ("pole_unmatched", "ff0000ff", "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"),
    ]
    for style_id, color, href in styles:
        style = ET.SubElement(doc, "Style", id=style_id)
        icon_style = ET.SubElement(style, "IconStyle")
        ET.SubElement(icon_style, "color").text = color
        ET.SubElement(icon_style, "scale").text = "1.0"
        icon = ET.SubElement(icon_style, "Icon")
        ET.SubElement(icon, "href").text = href

    style = ET.SubElement(doc, "Style", id="camera_heading")
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "color").text = "ff00ffff"
    ET.SubElement(icon_style, "scale").text = "1.2"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/arrow.png"

    frustum_style = ET.SubElement(doc, "Style", id="frustum")
    line_style = ET.SubElement(frustum_style, "LineStyle")
    ET.SubElement(line_style, "color").text = "ffffffff"
    ET.SubElement(line_style, "width").text = "1"
    poly_style = ET.SubElement(frustum_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = "4dff7700"

    fallback_styles = [
        (
            "fallback_estimate",
            "ff00a5ff",
            "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png",
        ),
        (
            "fallback_estimate_unresolved",
            "ff0066ff",
            "http://maps.google.com/mapfiles/kml/paddle/wht-circle.png",
        ),
    ]
    for style_id, color, href in fallback_styles:
        style = ET.SubElement(doc, "Style", id=style_id)
        icon_style = ET.SubElement(style, "IconStyle")
        ET.SubElement(icon_style, "color").text = color
        ET.SubElement(icon_style, "scale").text = "1.1"
        icon = ET.SubElement(icon_style, "Icon")
        ET.SubElement(icon, "href").text = href

    support_style = ET.SubElement(doc, "Style", id="fallback_support")
    support_line = ET.SubElement(support_style, "LineStyle")
    ET.SubElement(support_line, "color").text = "ff00ffff"
    ET.SubElement(support_line, "width").text = "1"
    support_poly = ET.SubElement(support_style, "PolyStyle")
    ET.SubElement(support_poly, "color").text = "2200ffff"

    estimate_style = ET.SubElement(doc, "Style", id="fallback_region")
    estimate_line = ET.SubElement(estimate_style, "LineStyle")
    ET.SubElement(estimate_line, "color").text = "ff00a5ff"
    ET.SubElement(estimate_line, "width").text = "2"
    estimate_poly = ET.SubElement(estimate_style, "PolyStyle")
    ET.SubElement(estimate_poly, "color").text = "3300a5ff"

    unresolved_style = ET.SubElement(doc, "Style", id="fallback_region_unresolved")
    unresolved_line = ET.SubElement(unresolved_style, "LineStyle")
    ET.SubElement(unresolved_line, "color").text = "ff0066ff"
    ET.SubElement(unresolved_line, "width").text = "2"
    unresolved_poly = ET.SubElement(unresolved_style, "PolyStyle")
    ET.SubElement(unresolved_poly, "color").text = "220066ff"


def build_kml(
    poles: list[dict],
    images: list[dict],
    footprints: list[dict],
    fallback_estimates: list[dict],
    fallback_regions: list[dict],
    img_field: str,
    images_dir: Path,
) -> ET.Element:
    del img_field

    kml = _kml_root()
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "Pole Tagging Debug"
    ET.SubElement(doc, "open").text = "1"

    _add_styles(doc)

    pole_folder = _folder(doc, "Poles")
    matched_count = 0
    for pole in poles:
        has_match = bool(pole["matched_imgs"])
        style = "#pole_matched" if has_match else "#pole_unmatched"
        status = "MATCHED" if has_match else "unmatched"
        desc_lines = [f"<b>OID:</b> {pole['oid']}", f"<b>Status:</b> {status}"]
        if has_match:
            matched_count += 1
            imgs = pole["matched_imgs"].split(";")
            desc_lines.append(f"<b>Images ({len(imgs)}):</b>")
            for img in imgs:
                img_path = images_dir / img.strip()
                file_uri = img_path.as_uri()
                desc_lines.append(f'&nbsp;&nbsp;<a href="{file_uri}">{img_path.name}</a>')
            first_uri = (images_dir / imgs[0].strip()).as_uri()
            desc_lines.append(f'<br/><img src="{first_uri}" width="640"/>')
        desc = "<br/>".join(desc_lines)
        _placemark_point(
            pole_folder,
            name=f"Pole {pole['oid']}" + (" ✓" if has_match else ""),
            lon=pole["lon"],
            lat=pole["lat"],
            description=desc,
            style_url=style,
        )

    matched_image_names: set[str] = set()
    for pole in poles:
        if pole["matched_imgs"]:
            for fname in pole["matched_imgs"].split(";"):
                matched_image_names.add(fname.strip())

    img_folder = _folder(doc, "Image Locations")
    for img in images:
        file_uri = img["path"].as_uri()
        image_name = img["path"].name
        is_matched = image_name in matched_image_names
        match_source = img.get("match_source", "primary" if is_matched else "unmatched")
        if match_source == "fallback":
            match_label = "FALLBACK MATCHED"
        elif match_source == "fallback-unresolved":
            match_label = "FALLBACK UNRESOLVED"
        elif is_matched:
            match_label = "MATCHED"
        else:
            match_label = "unmatched"
        desc = (
            f"<b>File:</b> {img['path'].name}<br/>"
            f"<b>Match:</b> {match_label}<br/>"
            f"<b>Pitch:</b> {img['pitch']:.1f}°<br/>"
            f"<b>Yaw (true):</b> {img['yaw']:.1f}°<br/>"
            f"<b>Rel Alt:</b> {img['alt']:.1f} m<br/><br/>"
            f'<a href="{file_uri}">Open full image</a><br/><br/>'
            f'<img src="{file_uri}" width="640"/>'
        )
        if match_source.startswith("fallback"):
            desc = (
                f"<b>File:</b> {img['path'].name}<br/>"
                f"<b>Match:</b> {match_label}<br/>"
                f"<b>Fallback mode:</b> {img.get('fallback_mode', 'n/a')}<br/>"
                f"<b>Fallback confidence:</b> {img.get('fallback_confidence', 0.0):.2f}<br/>"
                f"<b>Fallback pole:</b> {img.get('fallback_pole_oid', 'none')}<br/>"
                f"<b>Fallback note:</b> {img.get('fallback_reason', 'n/a')}<br/>"
                f"<b>Pitch:</b> {img['pitch']:.1f}°<br/>"
                f"<b>Yaw (true):</b> {img['yaw']:.1f}°<br/>"
                f"<b>Rel Alt:</b> {img['alt']:.1f} m<br/><br/>"
                f'<a href="{file_uri}">Open full image</a><br/><br/>'
                f'<img src="{file_uri}" width="640"/>'
            )
        heading = (img["yaw"] + 180) % 360
        _placemark_point(
            img_folder,
            name=img["path"].name + (" ✓" if is_matched else ""),
            lon=img["lon"],
            lat=img["lat"],
            description=desc,
            style_url="#camera_heading",
            heading=heading,
            cam_matched=is_matched,
        )

    if footprints:
        fp_folder = _folder(doc, "Frustum Footprints")
        for fp in footprints:
            _placemark_polygon(
                fp_folder,
                name=fp.get("name", fp["path"].name),
                coords=fp["coords"],
                style_url="#frustum",
            )

    if fallback_regions:
        region_folder = _folder(doc, "Fallback Regions")
        for region in fallback_regions:
            if not region.get("coords"):
                continue
            kind = region.get("kind", "support")
            style_url = "#fallback_support"
            if kind == "estimate":
                style_url = "#fallback_region"
            elif kind == "estimate-unresolved":
                style_url = "#fallback_region_unresolved"
            desc_lines = []
            if region.get("image_names"):
                desc_lines.append(f"<b>Images:</b> {', '.join(region['image_names'])}")
            if region.get("description"):
                desc_lines.append(region["description"])
            _placemark_polygon(
                region_folder,
                name=region.get("name", "Fallback region"),
                coords=region["coords"],
                description="<br/>".join(desc_lines),
                style_url=style_url,
            )

    if fallback_estimates:
        estimate_folder = _folder(doc, "Fallback Estimates")
        for estimate in fallback_estimates:
            pole_oid = estimate.get("pole_oid")
            style_url = (
                "#fallback_estimate" if pole_oid is not None else "#fallback_estimate_unresolved"
            )
            desc_lines = [
                f"<b>Mode:</b> {estimate.get('mode', 'n/a')}",
                f"<b>Confidence:</b> {estimate.get('confidence', 0.0):.2f}",
                f"<b>Support images:</b> {estimate.get('support_count', 0)}",
                f"<b>Nearest distance:</b> {estimate.get('nearest_distance_m', 0.0):.2f} m",
            ]
            second_distance = estimate.get("second_distance_m")
            if second_distance is not None:
                desc_lines.append(f"<b>Second distance:</b> {second_distance:.2f} m")
            assigned_pole = pole_oid if pole_oid is not None else "none"
            desc_lines.append(f"<b>Assigned pole:</b> {assigned_pole}")
            desc_lines.append(f"<b>Images:</b> {', '.join(estimate.get('image_names', []))}")
            desc_lines.append(f"<b>Note:</b> {estimate.get('reason', 'n/a')}")
            _placemark_point(
                estimate_folder,
                name=f"Fallback estimate ({estimate.get('mode', 'n/a')})",
                lon=estimate["estimate_lon"],
                lat=estimate["estimate_lat"],
                description="<br/>".join(desc_lines),
                style_url=style_url,
            )

    print(f"  Poles     : {len(poles)} total, {matched_count} matched")
    print(f"  Images    : {len(images)}")
    print(f"  Footprints: {len(footprints)}")
    return kml


def write_kml(kml: ET.Element, out_path: Path) -> None:
    ET.indent(kml, space="  ")
    raw = ET.tostring(kml, encoding="unicode", xml_declaration=False)

    def _cdata_wrap(match: re.Match) -> str:
        inner = match.group(1)
        inner = inner.replace("&lt;", "<").replace("&gt;", ">")
        inner = inner.replace("&amp;", "&").replace("&quot;", '"')
        return f"<description><![CDATA[{inner}]]></description>"

    raw = re.sub(r"<description>(.*?)</description>", _cdata_wrap, raw, flags=re.DOTALL)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write(raw)
    print(f"  Written   : {out_path}")
