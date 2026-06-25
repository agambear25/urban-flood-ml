"""Earth Engine access: init, per-city raw layers, and the SAR flood-mask detector.

The SAR change detection runs server-side in Earth Engine and we download only the
small binary flood mask — uniform and fast across cities.
"""
from __future__ import annotations

from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio

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


def _event_flood(cfg: CityConfig, region, w):
    """Binary flood image for ONE event (server-side change detection)."""
    pre_col, pre = _s1_composite(region, w.pre_start, w.pre_end, w.orbit)
    post_col, post = _s1_composite(region, w.post_start, w.post_end, w.orbit)
    pre_f = pre.focal_median(40, "circle", "meters")
    post_f = post.focal_median(40, "circle", "meters")
    diff = post_f.subtract(pre_f)
    flood = diff.lt(cfg.threshold_db)
    # drop permanent (and, for coastal cities, tidal) water + anything already dark
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    perm_cut = 20 if cfg.coastal else 50
    flood = flood.And(jrc.gt(perm_cut).Not()).And(pre_f.gt(cfg.perm_water_db))
    return flood.unmask(0), {"pre": pre_col.size().getInfo(), "post": post_col.size().getInfo()}


def build_flood_mask(cfg: CityConfig, out: Path) -> tuple[Path, dict]:
    """Multi-event Sentinel-1 change detection.

    Writes two GeoTIFFs: `flood_mask` (flooded in ANY event, 1/0) and `flood_count`
    (frequency-weighted event count) used as training sample weights.
    """
    region = aoi(cfg)
    per_event = {}
    arrays, weights, prof = [], [], None
    # Download each event mask separately (light per-tile compute), then combine LOCALLY —
    # combining all events server-side overruns GEE's per-tile compute limit on big AOIs.
    for i, w in enumerate(cfg.events):
        flood, c = _event_flood(cfg, region, w)
        per_event[w.name] = c
        ep = out.parent / f"_event_{i}.tif"
        _download(flood.toByte().clip(region), ep, cfg)
        with rasterio.open(ep) as s:
            arrays.append(s.read(1)); prof = s.profile
        weights.append(float(w.weight))
        ep.unlink(missing_ok=True)

    h = min(a.shape[0] for a in arrays); wd = min(a.shape[1] for a in arrays)
    count = np.zeros((h, wd), dtype="float32")
    for a, wt in zip(arrays, weights, strict=False):
        count += (a[:h, :wd] == 1).astype("float32") * wt
    flood_any = (count > 0).astype("uint8")

    prof.update(count=1, height=h, width=wd, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(flood_any, 1)
    cprof = prof.copy(); cprof.update(dtype="float32", nodata=None)
    with rasterio.open(out.parent / "flood_count.tif", "w", **cprof) as dst:
        dst.write(count, 1)
    return out, per_event


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
