"""Earth Engine access: init, per-city raw layers, and the SAR flood-mask detector.

The SAR change detection runs server-side in Earth Engine and we download only the
small binary flood mask — uniform and fast across cities.
"""
from __future__ import annotations

from pathlib import Path

import ee
import geemap

from .config import CityConfig


def init_ee(project: str) -> None:
    """Connect to Earth Engine; only fall back to the browser sign-in if needed."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def aoi(cfg: CityConfig) -> ee.Geometry:
    return ee.Geometry.Rectangle([cfg.west, cfg.south, cfg.east, cfg.north])


def _download(img: ee.Image, path: Path, cfg: CityConfig, scale: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    geemap.download_ee_image(img, str(path), scale=scale or cfg.scale,
                             region=aoi(cfg), crs="EPSG:4326")
    return path


def _s1_composite(region, start, end, orbit):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region).filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .select("VV")
    )
    return col, col.median()


def build_flood_mask(cfg: CityConfig, out: Path) -> tuple[Path, dict]:
    """Server-side Sentinel-1 change detection -> binary flood mask GeoTIFF."""
    region = aoi(cfg)
    pre_col, pre = _s1_composite(region, cfg.sar.pre_start, cfg.sar.pre_end, cfg.sar.orbit)
    post_col, post = _s1_composite(region, cfg.sar.post_start, cfg.sar.post_end, cfg.sar.orbit)
    counts = {"pre": pre_col.size().getInfo(), "post": post_col.size().getInfo()}

    # light speckle smoothing, then dB change
    pre_f = pre.focal_median(40, "circle", "meters")
    post_f = post.focal_median(40, "circle", "meters")
    diff = post_f.subtract(pre_f)

    flood = diff.lt(cfg.threshold_db)
    # drop permanent water (JRC) and anything already dark (river channel / sea)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    flood = flood.And(jrc.gt(50).Not()).And(pre_f.gt(cfg.perm_water_db))
    flood = flood.rename("flood").toByte().clip(region)

    _download(flood, out, cfg)
    return out, counts


def fetch_static_layers(cfg: CityConfig, data_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download the global static layers (DEM, built-up, MERIT-Hydro drainage bands).

    Skips layers that already exist (so a city with copied rasters only fetches what's missing).
    """
    region = aoi(cfg)
    paths = {}

    def grab(img, name, scale=None):
        p = data_dir / name
        if force or not p.exists():
            _download(img, p, cfg, scale=scale)
        paths[name.replace(".tif", "")] = p

    grab(ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().clip(region),
         "dem.tif", scale=30)
    grab(ee.ImageCollection("ESA/WorldCover/v200").first().clip(region).eq(50).rename("builtup"),
         "builtup.tif")
    # MERIT Hydro — global hydrography: hnd=HAND, upa=upstream area
    grab(ee.Image("MERIT/Hydro/v1_0_1").select(["hnd", "upa"]).clip(region),
         "merit.tif", scale=90)

    return paths
