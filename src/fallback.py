from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.ops import polygonize, transform, unary_union

_R_EARTH_M = 6_371_000.0


@dataclass(frozen=True)
class FallbackConfig:
    enabled: bool = False
    max_camera_to_pole_m: float = 20.0
    max_snap_distance_m: float = 12.0
    min_overlap_images: int = 2
    singleton_confidence_cap: float = 0.45
    region_relaxation_m: float = 0.0
    weight_clip: float = 3.0


@dataclass
class _LocalFrame:
    origin_lat: float
    origin_lon: float

    def __post_init__(self) -> None:
        cos_lat = math.cos(math.radians(self.origin_lat))
        self._cos_lat = cos_lat if abs(cos_lat) > 1e-9 else 1e-9

    def lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        east = math.radians(lon - self.origin_lon) * _R_EARTH_M * self._cos_lat
        north = math.radians(lat - self.origin_lat) * _R_EARTH_M
        return east, north

    def xy_to_lonlat(self, east: float, north: float) -> tuple[float, float]:
        lat = self.origin_lat + math.degrees(north / _R_EARTH_M)
        lon = self.origin_lon + math.degrees(east / (_R_EARTH_M * self._cos_lat))
        return lon, lat

    def geom_to_xy(self, geometry):
        return transform(self._geom_to_xy_coords, geometry)

    def geom_to_lonlat(self, geometry):
        return transform(self._geom_to_lonlat_coords, geometry)

    def _geom_to_xy_coords(self, lon, lat, z=None):
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
        east = np.radians(lon_arr - self.origin_lon) * _R_EARTH_M * self._cos_lat
        north = np.radians(lat_arr - self.origin_lat) * _R_EARTH_M
        return east, north

    def _geom_to_lonlat_coords(self, east, north, z=None):
        east_arr = np.asarray(east, dtype=np.float64)
        north_arr = np.asarray(north, dtype=np.float64)
        lat = self.origin_lat + np.degrees(north_arr / _R_EARTH_M)
        lon = self.origin_lon + np.degrees(east_arr / (_R_EARTH_M * self._cos_lat))
        return lon, lat


def validate_fallback_config(config: FallbackConfig) -> None:
    if config.max_camera_to_pole_m <= 0:
        raise ValueError(
            f"fallback_max_camera_to_pole_m must be positive, got {config.max_camera_to_pole_m}."
        )
    if config.max_snap_distance_m <= 0:
        raise ValueError(
            f"fallback_max_snap_distance_m must be positive, got {config.max_snap_distance_m}."
        )
    if config.min_overlap_images < 2:
        raise ValueError(
            f"fallback_min_overlap_images must be at least 2, got {config.min_overlap_images}."
        )
    if not 0 <= config.singleton_confidence_cap <= 1:
        raise ValueError(
            "fallback_singleton_confidence_cap must be in [0, 1], "
            f"got {config.singleton_confidence_cap}."
        )
    if config.region_relaxation_m < 0:
        raise ValueError(
            f"fallback_region_relaxation_m must be non-negative, got {config.region_relaxation_m}."
        )
    if config.weight_clip < 1:
        raise ValueError(f"fallback_weight_clip must be at least 1, got {config.weight_clip}.")


def resolve_fallback_matches(
    unmatched_images: list[dict],
    poles: list[dict],
    config: FallbackConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not config.enabled or not unmatched_images:
        return [], [], []

    frame = _build_local_frame(unmatched_images)
    pole_points = [_build_pole_point(pole, frame) for pole in poles]
    image_entries = [_build_image_entry(image, frame, config) for image in unmatched_images]

    assignments: list[dict] = []
    estimate_regions: list[dict] = []
    support_regions = [_support_region_record(entry, frame) for entry in image_entries]

    for component in _connected_components(image_entries, config.region_relaxation_m):
        remaining = component[:]

        while len(remaining) >= config.min_overlap_images:
            candidate = _best_overlap_candidate(remaining, config)
            if candidate is None:
                break

            support_entries = [
                entry
                for entry in remaining
                if _covers(entry["effective_region_xy"], candidate["point_xy"])
            ]
            if len(support_entries) < config.min_overlap_images:
                break

            assignment = _build_assignment(
                support_entries,
                candidate["point_xy"],
                candidate["cell_xy"],
                frame,
                pole_points,
                config,
                mode="overlap",
            )
            assignments.append(assignment)
            estimate_regions.append(_estimate_region_record(assignment))
            supported_names = set(assignment["image_names"])
            remaining = [entry for entry in remaining if entry["image_name"] not in supported_names]

        for entry in remaining:
            camera_point_xy = Point(entry["camera_xy"])
            assignment = _build_assignment(
                [entry],
                camera_point_xy,
                entry["region_xy"],
                frame,
                pole_points,
                config,
                mode="singleton",
            )
            assignments.append(assignment)
            estimate_regions.append(_estimate_region_record(assignment))

    return assignments, support_regions, estimate_regions


def _build_local_frame(unmatched_images: list[dict]) -> _LocalFrame:
    image_count = len(unmatched_images)
    mean_lat = sum(image["camera"].gps_coords[0] for image in unmatched_images) / image_count
    mean_lon = sum(image["camera"].gps_coords[1] for image in unmatched_images) / image_count
    return _LocalFrame(mean_lat, mean_lon)


def _build_pole_point(pole: dict, frame: _LocalFrame) -> dict:
    east, north = frame.lonlat_to_xy(pole["lon"], pole["lat"])
    return {**pole, "x": east, "y": north}


def _build_image_entry(image: dict, frame: _LocalFrame, config: FallbackConfig) -> dict:
    camera = image["camera"]
    cam_lat, cam_lon = camera.gps_coords
    footprint_xy = frame.geom_to_xy(image["footprint"])
    camera_xy = frame.lonlat_to_xy(cam_lon, cam_lat)
    trusted_region = Point(camera_xy).buffer(config.max_camera_to_pole_m)
    region_xy = footprint_xy.intersection(trusted_region)
    if region_xy.is_empty:
        region_xy = trusted_region

    effective_region_xy = region_xy
    if config.region_relaxation_m > 0:
        effective_region_xy = region_xy.buffer(config.region_relaxation_m)

    return {
        "image_name": image["image_name"],
        "camera": camera,
        "camera_xy": camera_xy,
        "region_xy": region_xy,
        "effective_region_xy": effective_region_xy,
        "weight": _region_weight(region_xy, config),
    }


def _region_weight(region_xy, config: FallbackConfig) -> float:
    area_m2 = max(float(region_xy.area), 1.0)
    precision_radius = math.sqrt(area_m2 / math.pi)
    raw_weight = config.max_camera_to_pole_m / max(precision_radius, 1.0)
    return min(config.weight_clip, max(1.0, raw_weight))


def _connected_components(entries: list[dict], relaxation_m: float) -> list[list[dict]]:
    components: list[list[dict]] = []
    visited: set[int] = set()

    for index in range(len(entries)):
        if index in visited:
            continue

        stack = [index]
        component_indices: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_indices.append(current)

            for other in range(len(entries)):
                if other in visited or other == current:
                    continue
                if _regions_connected(entries[current], entries[other], relaxation_m):
                    stack.append(other)

        components.append([entries[i] for i in component_indices])

    return components


def _regions_connected(left: dict, right: dict, relaxation_m: float) -> bool:
    left_geom = left["effective_region_xy"]
    right_geom = right["effective_region_xy"]
    if left_geom.intersects(right_geom):
        return True
    if relaxation_m <= 0:
        return False
    return left["region_xy"].distance(right["region_xy"]) <= relaxation_m


def _best_overlap_candidate(entries: list[dict], config: FallbackConfig) -> dict | None:
    cells = _arrangement_cells([entry["effective_region_xy"] for entry in entries])
    best: dict | None = None

    for cell_xy in cells:
        point_xy = cell_xy.representative_point()
        supporters = [entry for entry in entries if _covers(entry["effective_region_xy"], point_xy)]
        support_count = len(supporters)
        if support_count == 0:
            continue
        score = sum(entry["weight"] for entry in supporters)
        candidate = {
            "cell_xy": cell_xy,
            "point_xy": point_xy,
            "support_count": support_count,
            "score": score,
        }
        if _better_candidate(candidate, best, config.min_overlap_images):
            best = candidate

    if best is None or best["support_count"] < config.min_overlap_images:
        return None
    return best


def _arrangement_cells(regions: list) -> list:
    non_empty_regions = [region.buffer(0) for region in regions if not region.is_empty]
    if not non_empty_regions:
        return []
    if len(non_empty_regions) == 1:
        return [non_empty_regions[0]]

    boundaries = unary_union([region.boundary for region in non_empty_regions])
    cells = [
        cell
        for cell in polygonize(boundaries)
        if any(_covers(region, cell.representative_point()) for region in non_empty_regions)
    ]
    if cells:
        return cells
    return [unary_union(non_empty_regions).buffer(0)]


def _better_candidate(candidate: dict, current: dict | None, min_overlap_images: int) -> bool:
    if current is None:
        return True

    candidate_key = (
        int(candidate["support_count"] >= min_overlap_images),
        candidate["support_count"],
        round(candidate["score"], 6),
        -float(candidate["cell_xy"].area),
    )
    current_key = (
        int(current["support_count"] >= min_overlap_images),
        current["support_count"],
        round(current["score"], 6),
        -float(current["cell_xy"].area),
    )
    return candidate_key > current_key


def _covers(geometry, point) -> bool:
    return geometry.covers(point)


def _build_assignment(
    entries: list[dict],
    estimate_point_xy,
    region_xy,
    frame: _LocalFrame,
    pole_points: list[dict],
    config: FallbackConfig,
    mode: str,
) -> dict:
    estimate_x = float(estimate_point_xy.x)
    estimate_y = float(estimate_point_xy.y)
    estimate_lon, estimate_lat = frame.xy_to_lonlat(estimate_x, estimate_y)
    nearest, second = _nearest_two_poles(estimate_x, estimate_y, pole_points)
    nearest_distance = nearest[0] if nearest is not None else math.inf
    second_distance = second[0] if second is not None else math.inf
    snapped = nearest is not None and nearest_distance <= config.max_snap_distance_m

    confidence = _confidence(
        nearest_distance,
        second_distance,
        len(entries),
        float(region_xy.area),
        config,
        mode,
    )
    if mode == "singleton":
        confidence = min(confidence, config.singleton_confidence_cap)
    if not snapped:
        confidence = 0.0

    if snapped:
        reason = (
            f"snapped to pole {nearest[1]['oid']} at {nearest_distance:.1f} m "
            f"with support from {len(entries)} image(s)"
        )
    elif nearest is None:
        reason = "no candidate poles available"
    else:
        reason = (
            f"nearest pole {nearest[1]['oid']} is {nearest_distance:.1f} m away, "
            f"beyond the snap limit of {config.max_snap_distance_m:.1f} m"
        )

    return {
        "image_names": [entry["image_name"] for entry in entries],
        "pole_oid": nearest[1]["oid"] if snapped and nearest is not None else None,
        "confidence": confidence,
        "mode": mode,
        "estimate_lon": estimate_lon,
        "estimate_lat": estimate_lat,
        "nearest_distance_m": nearest_distance,
        "second_distance_m": second_distance,
        "support_count": len(entries),
        "region_coords": _geometry_coords(frame.geom_to_lonlat(region_xy)),
        "reason": reason,
    }


def _nearest_two_poles(
    x: float,
    y: float,
    pole_points: list[dict],
) -> tuple[tuple[float, dict] | None, tuple[float, dict] | None]:
    ranked = sorted(
        ((math.hypot(pole["x"] - x, pole["y"] - y), pole) for pole in pole_points),
        key=lambda item: item[0],
    )
    nearest = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    return nearest, second


def _confidence(
    nearest_distance: float,
    second_distance: float,
    support_count: int,
    region_area_m2: float,
    config: FallbackConfig,
    mode: str,
) -> float:
    proximity = _clamp01(1.0 - (nearest_distance / config.max_snap_distance_m))
    if math.isinf(second_distance):
        separation = 0.5
    else:
        separation = _clamp01((second_distance - nearest_distance) / max(second_distance, 1.0))

    if mode == "singleton":
        return 0.75 * proximity + 0.25 * separation

    support_strength = _clamp01(support_count / 3.0)
    compactness = 1.0 / (1.0 + region_area_m2 / (math.pi * config.max_snap_distance_m**2))
    return 0.45 * proximity + 0.25 * separation + 0.20 * support_strength + 0.10 * compactness


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _support_region_record(entry: dict, frame: _LocalFrame) -> dict:
    return {
        "name": f"Fallback support - {entry['image_name']}",
        "coords": _geometry_coords(frame.geom_to_lonlat(entry["region_xy"])),
        "kind": "support",
        "image_names": [entry["image_name"]],
    }


def _estimate_region_record(assignment: dict) -> dict:
    return {
        "name": f"Fallback estimate - {', '.join(assignment['image_names'])}",
        "coords": assignment["region_coords"],
        "kind": "estimate" if assignment["pole_oid"] is not None else "estimate-unresolved",
        "image_names": assignment["image_names"],
        "description": assignment["reason"],
    }


def _geometry_coords(geometry) -> list[tuple[float, float]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [(lon, lat) for lon, lat in geometry.exterior.coords]
    if geometry.geom_type == "MultiPolygon":
        largest = max(geometry.geoms, key=lambda poly: poly.area, default=None)
        if largest is not None:
            return [(lon, lat) for lon, lat in largest.exterior.coords]
    return []
