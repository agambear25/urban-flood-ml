"""Build the data + image overlays for the interactive multi-city dashboard.

For every city it produces:
  docs/assets/<slug>.json              — grid of click-to-explain cells (risk + plain-English SHAP),
                                          rainfall scenarios, people-at-risk, city metadata
  docs/assets/overlays/<slug>_<sc>.png — a risk overlay per rainfall scenario (light/moderate/heavy)

Run from the repo root:  python dashboard/build_data.py
"""
import json
from pathlib import Path

import ee
import geemap
import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
import rioxarray as rxr
import shap
from rasterio.enums import Resampling
import xgboost as xgb
from PIL import Image

from floodml import FEATURES
from floodml.config import load_city

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
OVERLAYS = ASSETS / "overlays"
OVERLAYS.mkdir(parents=True, exist_ok=True)

CITIES = ["delhi", "mumbai", "bengaluru", "chandigarh"]
SCENARIOS = {"light": 0.85, "moderate": 0.70, "heavy": 0.50}  # percentile cutoff for "at risk"
GRID = 22  # cells per side for the click-to-explain layer

# feature -> (phrase when it RAISES risk, phrase when it LOWERS risk)
PHRASE = {
    "elevation":     ("Low-lying ground", "Higher ground"),
    "slope":         ("Flat, slow to drain", "Sloped, drains fast"),
    "curvature":     ("Sits in a hollow", "On a ridge"),
    "local_relief":  ("In a local dip", "On a local rise"),
    "hand":          ("Close to drainage level", "Well above drainage"),
    "dist_river":    ("Near a river", "Far from rivers"),
    "builtup":       ("Densely built-up", "Built-up area"),
    "dist_drain":    ("Near a storm drain", "Far from drains"),
    "drain_density": ("Sparse drainage nearby", "Dense drainage nearby"),
    "upstream_area": ("Lots of upstream runoff", "Little upstream runoff"),
    "sink_depth":    ("A natural low spot", "Not a low spot"),
}


def level(p):
    return ("Very high" if p >= 0.85 else "High" if p >= 0.70
            else "Medium" if p >= 0.50 else "Low")


def percentile_rank(arr, valid):
    p = np.full(arr.shape, np.nan, dtype="float32")
    v = arr[valid]
    order = v.argsort()
    ranks = np.empty(len(order), dtype="float32")
    ranks[order] = np.linspace(0, 1, len(order), dtype="float32")
    p[valid] = ranks
    return p


def render_overlay(p, cutoff, out):
    cmap = matplotlib.colormaps["YlOrRd"]
    mask = np.isfinite(p) & (p >= cutoff)
    norm = np.clip((p - cutoff) / max(1 - cutoff, 1e-6), 0, 1)
    colored = (cmap(np.nan_to_num(norm)) * 255).astype("uint8")
    rgba = np.zeros((*p.shape, 4), dtype="uint8")
    rgba[..., :3] = colored[..., :3]
    rgba[..., 3] = np.where(mask, 205, 0).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(out)


def people_grids(cfg, susc_path, p):
    """Return (population, percentile) on WorldPop's NATIVE 100 m grid.

    We downsample the percentile onto the population grid (area-average) rather than
    inflating population counts up to the 10 m grid — otherwise each 100 m count gets
    replicated ~100x and totals explode.
    """
    region = ee.Geometry.Rectangle([cfg.west, cfg.south, cfg.east, cfg.north])
    img = (ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("year", 2020))
           .filterBounds(region).mosaic().clip(region))
    pth = REPO / "data" / cfg.slug / "worldpop.tif"
    if not pth.exists():
        geemap.download_ee_image(img, str(pth), scale=100, region=region, crs="EPSG:4326")
    pop = rxr.open_rasterio(pth).squeeze()
    ref = rxr.open_rasterio(susc_path).squeeze()
    p_da = ref.copy(data=p.astype("float32"))
    p_pop = p_da.rio.reproject_match(pop, resampling=Resampling.average).values.astype("float64")
    popv = np.nan_to_num(pop.values.astype("float64"))
    popv[popv < 0] = 0
    return popv, p_pop


def build_city(slug):
    cfg = load_city(slug, str(REPO / "configs" / "city"))
    ddir = REPO / "data" / slug
    with rasterio.open(ddir / "susceptibility.tif") as s:
        susc = s.read(1).astype("float32")
        tr = s.transform
    with rasterio.open(ddir / "flood_mask.tif") as s:
        flood = s.read(1)
    with rasterio.open(ddir / "feature_stack.tif") as s:
        bands = list(s.descriptions)
        stack = s.read().astype("float32")
    stack = stack[[bands.index(f) for f in FEATURES]]

    valid = (flood != 255) & np.isfinite(susc) & np.isfinite(stack).all(axis=0)
    p = percentile_rank(susc, valid)
    H, W = susc.shape
    S, Wst = cfg.south, cfg.west
    N, E = cfg.north, cfg.east

    # overlays + people-at-risk per scenario
    popv, p_pop = people_grids(cfg, ddir / "susceptibility.tif", p)
    px_km2 = (abs(tr.a) * 111.32 * np.cos(np.deg2rad(cfg.lat_mid))) * (abs(tr.e) * 111.32)
    scenarios = {}
    for sc, cut in SCENARIOS.items():
        render_overlay(p, cut, OVERLAYS / f"{slug}_{sc}.png")
        people = int(popv[(p_pop >= cut) & np.isfinite(p_pop)].sum())
        area = float((valid & (p >= cut)).sum() * px_km2)
        scenarios[sc] = {"people": people, "area_km2": round(area, 1)}

    # SHAP on a sample, aggregated to the grid
    model = xgb.XGBClassifier()
    model.load_model(str(REPO / "models" / f"{slug}_model.json"))
    rng = np.random.default_rng(0)
    vidx = np.argwhere(valid)
    take = vidx[rng.choice(len(vidx), min(18000, len(vidx)), replace=False)]
    X = np.stack([stack[b][take[:, 0], take[:, 1]] for b in range(len(FEATURES))], axis=1)
    sv = shap.TreeExplainer(model).shap_values(X)

    ch, cw = H / GRID, W / GRID
    gr = (take[:, 0] / ch).astype(int).clip(0, GRID - 1)
    gc = (take[:, 1] / cw).astype(int).clip(0, GRID - 1)
    cell_p = p[take[:, 0], take[:, 1]]

    cells = []
    for r in range(GRID):
        for c in range(GRID):
            m = (gr == r) & (gc == c)
            if m.sum() < 5:
                continue
            pr = float(np.nanmean(cell_p[m]))
            mean_sv = sv[m].mean(axis=0)
            top = np.argsort(-np.abs(mean_sv))[:4]
            why = [{"factor": PHRASE[FEATURES[i]][0 if mean_sv[i] > 0 else 1],
                    "value": round(float(mean_sv[i]), 3),
                    "dir": "up" if mean_sv[i] > 0 else "down"} for i in top]
            cells.append({"r": r, "c": c, "risk": round(pr, 3), "level": level(pr), "why": why})

    # top drivers IN THE AT-RISK AREAS (not the dry city average) — far more intuitive
    himask = cell_p >= 0.85
    drv = sv[himask] if himask.sum() > 30 else sv
    dmean = drv.mean(axis=0)
    order = np.argsort(-np.abs(dmean))[:5]
    top_drivers = [{"factor": PHRASE[FEATURES[i]][0 if dmean[i] > 0 else 1],
                    "value": round(float(abs(dmean[i])), 3),
                    "dir": "up" if dmean[i] > 0 else "down"} for i in order]

    out = {
        "name": cfg.name.split("(")[0].strip(), "slug": slug,
        "bounds": [[S, Wst], [N, E]], "center": [round(cfg.lat_mid, 4), round((Wst + E) / 2, 4)],
        "overlays": {sc: f"assets/overlays/{slug}_{sc}.png" for sc in SCENARIOS},
        "scenarios": scenarios,
        "grid": {"rows": GRID, "cols": GRID}, "cells": cells,
        "top_drivers": top_drivers,
        "note": cfg.notes,
    }
    (ASSETS / f"{slug}.json").write_text(json.dumps(out))
    print(f"  {slug}: {len(cells)} cells | heavy-rain people-at-risk {scenarios['heavy']['people']:,}")
    return out


if __name__ == "__main__":
    ee.Initialize(project="urban-flood-analysis-ncr-in")
    index = []
    for slug in CITIES:
        print(f"building {slug} ...")
        o = build_city(slug)
        index.append({"slug": slug, "name": o["name"]})
    (ASSETS / "cities.json").write_text(json.dumps(index))
    print("wrote docs/assets/cities.json")
