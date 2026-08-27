from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the runtime config file is malformed."""


@dataclass(frozen=True)
class AppConfig:
    data_path: str | None = None
    output_path: str | None = None
    matches_csv_path: str | None = None
    debug_output_path: str | None = None
    vendor: str | None = None
    img_field: str | None = None
    dist_field: str | None = None
    height_field: str | None = None
    default_pole_height_m: float | None = None
    dem_path: str | None = None
    debug_kml: bool | None = None
    max_range_m: float | None = None
    nadir_threshold_deg: float | None = None
    margin_px: float | None = None
    fallback_enabled: bool | None = None
    fallback_max_camera_to_pole_m: float | None = None
    fallback_max_snap_distance_m: float | None = None
    fallback_min_overlap_images: int | None = None
    fallback_singleton_confidence_cap: float | None = None
    fallback_region_relaxation_m: float | None = None
    fallback_weight_clip: float | None = None


def _optional_str(section: dict, key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Config key '{key}' must be a string.")
    return value


def _optional_float(section: dict, key: str) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Config key '{key}' must be a number.")
    return float(value)


def _optional_bool(section: dict, key: str) -> bool | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"Config key '{key}' must be a boolean.")
    return value


def _optional_int(section: dict, key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Config key '{key}' must be an integer.")
    return value


def _optional_table(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' must be a TOML table.")
    return value


def load_config(config_path: Path | None) -> AppConfig:
    """Load the optional repo-level TOML config file."""
    if config_path is None:
        return AppConfig()

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must parse to a TOML table.")

    defaults = _optional_table(raw, "defaults")
    projection = _optional_table(raw, "projection")
    fallback = _optional_table(raw, "fallback")

    return AppConfig(
        data_path=_optional_str(defaults, "data_path"),
        output_path=_optional_str(defaults, "output_path"),
        matches_csv_path=_optional_str(defaults, "matches_csv_path"),
        debug_output_path=_optional_str(defaults, "debug_output_path"),
        vendor=_optional_str(defaults, "vendor"),
        img_field=_optional_str(defaults, "img_field"),
        dist_field=_optional_str(defaults, "dist_field"),
        height_field=_optional_str(defaults, "height_field"),
        default_pole_height_m=_optional_float(defaults, "default_pole_height_m"),
        dem_path=_optional_str(defaults, "dem_path"),
        debug_kml=_optional_bool(defaults, "debug_kml"),
        max_range_m=_optional_float(projection, "max_range_m"),
        nadir_threshold_deg=_optional_float(projection, "nadir_threshold_deg"),
        margin_px=_optional_float(projection, "margin_px"),
        fallback_enabled=_optional_bool(fallback, "enabled"),
        fallback_max_camera_to_pole_m=_optional_float(fallback, "max_camera_to_pole_m"),
        fallback_max_snap_distance_m=_optional_float(fallback, "max_snap_distance_m"),
        fallback_min_overlap_images=_optional_int(fallback, "min_overlap_images"),
        fallback_singleton_confidence_cap=_optional_float(fallback, "singleton_confidence_cap"),
        fallback_region_relaxation_m=_optional_float(fallback, "region_relaxation_m"),
        fallback_weight_clip=_optional_float(fallback, "weight_clip"),
    )
