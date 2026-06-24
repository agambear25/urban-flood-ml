"""Feature engineering: terrain + a uniform drainage backbone, aligned to the flood-mask grid.

Drainage features (computed identically for every city, so the per-city models stay comparable):
  dist_drain     distance to the nearest OSM drain/canal/culvert
  drain_density  drain length density in a moving window
  upstream_area  MERIT-Hydro upstream contributing area (log-scaled)
  sink_depth     local depression depth (underpass / waterlogging low-points)
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy.ndimage import (
    distance_transform_edt,
    grey_closing,
    laplace,
    minimum_filter,
    uniform_filter,
)

from . import FEATURES
from .config import CityConfig


def _ref(data_dir: Path):
    with rasterio.open(data_dir / "flood_mask.tif") as s:
        return s.profile, s.transform, (s.height, s.width), s.crs


def _align(path: Path, ref_xr, band: int = 1, resampling=Resampling.bilinear):
    r = rxr.open_rasterio(path)
    rb = r.sel(band=band) if "band" in r.dims else r.squeeze()
    return rb.rio.reproject_match(ref_xr, resampling=resampling).values.astype("float32")


def _osm_drains(cfg: CityConfig, crs):
    tags = {"waterway": ["drain", "ditch", "canal", "stream", "river"], "tunnel": ["culvert"]}
    try:
        g = ox.features_from_bbox(bbox=(cfg.west, cfg.south, cfg.east, cfg.north), tags=tags)
        g = g[g.geom_type.isin(["LineString", "MultiLineString"])].to_crs(crs)
        return g
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _rasterize_lines(gdf, shape, transform):
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, "uint8")
    return rasterize([(g, 1) for g in gdf.geometry], out_shape=shape, transform=transform,
                     fill=0, all_touched=True, dtype="uint8")


def build_features(cfg: CityConfig, data_dir: str | Path) -> Path:
    """Build the aligned multi-band feature stack for one city."""
    data_dir = Path(data_dir)
    profile, transform, shape, crs = _ref(data_dir)
    ref_xr = rxr.open_rasterio(data_dir / "flood_mask.tif").squeeze()

    px = abs(transform.a) * 111_320 * np.cos(np.deg2rad(cfg.lat_mid))
    py = abs(transform.e) * 111_320
    px_m = (px + py) / 2

    # --- terrain ---
    dem = _align(data_dir / "dem.tif", ref_xr)
    dem_f = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    gy, gx = np.gradient(dem_f, py, px)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype("float32")
    curvature = laplace(dem_f).astype("float32")
    local_relief = (dem_f - minimum_filter(dem_f, size=51)).astype("float32")
    sink_depth = (grey_closing(dem_f, size=15) - dem_f).astype("float32")
    builtup = _align(data_dir / "builtup.tif", ref_xr, resampling=Resampling.nearest)

    # --- drainage backbone ---
    hand = _align(data_dir / "merit.tif", ref_xr, band=1)                 # MERIT hnd = HAND
    upstream = _align(data_dir / "merit.tif", ref_xr, band=2)             # MERIT upa
    upstream = np.log1p(np.clip(upstream, 0, None)).astype("float32")

    drains = _osm_drains(cfg, crs)
    dr = _rasterize_lines(drains, shape, transform)
    dist_drain = (distance_transform_edt(dr == 0) * px_m).astype("float32")
    drain_density = uniform_filter((dr > 0).astype("float32"), size=51).astype("float32")

    rivers_path = data_dir / "rivers.geojson"
    if rivers_path.exists():
        rivers = gpd.read_file(rivers_path).to_crs(crs)
        rivers = rivers[rivers.geom_type.isin(["LineString", "MultiLineString"])]
        rr = _rasterize_lines(rivers, shape, transform) if len(rivers) else dr
    else:
        rr = dr if dr.any() else np.zeros(shape, "uint8")
    dist_river = (distance_transform_edt(rr == 0) * px_m).astype("float32")

    feats = {
        "elevation": dem.astype("float32"), "slope": slope, "curvature": curvature,
        "local_relief": local_relief, "hand": hand, "dist_river": dist_river,
        "builtup": builtup, "dist_drain": dist_drain, "drain_density": drain_density,
        "upstream_area": upstream, "sink_depth": sink_depth,
    }

    out = data_dir / "feature_stack.tif"
    sp = profile.copy()
    sp.update(count=len(FEATURES), dtype="float32", nodata=None, compress="lzw")
    with rasterio.open(out, "w", **sp) as dst:
        for i, name in enumerate(FEATURES, start=1):
            dst.write(feats[name], i)
            dst.set_band_description(i, name)
    return out
