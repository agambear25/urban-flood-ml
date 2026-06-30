"""Export precomputed, web-ready data for the FloodLens map — no live backend.

Writes under web/<city>/:
  risk.png          street-waterlogging susceptibility as RELATIVE quantile bands (nodata transparent)
  meta.json         WGS84 bounds, band colours/thresholds, the REAL flood-event dates, honest caveats
  hotspots.geojson  documented hotspots + the relative band here + the model's plain reasons +
                    an imagery-visibility flag (can satellite radar even check this spot?)

Everything is relative-within-city and presence-only — said plainly in meta.caveats. Deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from . import FEATURES
from . import explain as expl
from .config import load_city

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAND_NAMES = ["low", "moderate", "high", "severe"]
BAND_COLORS = {"low": (46, 125, 50), "moderate": (249, 168, 37), "high": (239, 108, 0), "severe": (198, 40, 40)}
BAND_ALPHA = {"low": 70, "moderate": 120, "high": 150, "severe": 185}


def enrich_rainfall(city, out_dir="docs/web", configs="configs/city", project=None):
    """Pull REAL rainfall for each event and write it into web/<city>/meta.json.

    For each event window we compute, server-side on Earth Engine, the wettest single day's
    rainfall (GPM IMERG, half-hourly -> daily mm, max over the window) averaged over the city.
    That is an honest "how much rain when it flooded" number. Needs Earth Engine; if it isn't
    available, leave the events untouched rather than invent figures.
    """
    import ee
    from datetime import date

    cfg = load_city(city, configs)
    ee.Initialize(project=project or cfg.ee_project)
    aoi = ee.Geometry.Rectangle([cfg.west, cfg.south, cfg.east, cfg.north])
    by_name = {e.name: e for e in cfg.events}

    meta_path = Path(out_dir) / city / "meta.json"
    meta = json.loads(meta_path.read_text())
    for ev in meta["events"]:
        w = by_name.get(ev["name"])
        if not w:
            continue
        ndays = max((date.fromisoformat(w.post_end) - date.fromisoformat(w.post_start)).days, 1)
        ic = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(w.post_start, w.post_end).select("precipitation")
        start = ee.Date(w.post_start)

        def daily(d):
            s = start.advance(ee.Number(d), "day")
            return ic.filterDate(s, s.advance(1, "day")).map(lambda im: im.multiply(0.5)).sum()  # mm/day

        peak = ee.ImageCollection(ee.List.sequence(0, ndays - 1).map(daily)).max()  # wettest day, per pixel
        val = peak.reduceRegion(ee.Reducer.mean(), aoi, 11000, maxPixels=int(1e9)).get("precipitation")
        mm = ee.Number(val).getInfo() if val is not None else None
        ev["peak_day_mm"] = round(mm, 1) if mm is not None else None
        ev["window_days"] = ndays
    meta_path.write_text(json.dumps(meta, indent=2))
    return [(e["date"][:4], e.get("peak_day_mm")) for e in meta["events"]]


def _band_of(value, thr):
    if not np.isfinite(value):
        return None
    if value >= thr[2]:
        return "severe"
    if value >= thr[1]:
        return "high"
    if value >= thr[0]:
        return "moderate"
    return "low"


def export_city(city, data_dir="data", out_dir="docs/web", configs="configs/city",
                hotspots="events/hotspots.geojson", downsample=2):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out = out_dir / city
    out.mkdir(parents=True, exist_ok=True)

    surf = data_dir / city / "waterlog_susceptibility.tif"
    with rasterio.open(surf) as s:
        susc = s.read(1)
        bounds = transform_bounds(s.crs, "EPSG:4326", *s.bounds)  # W, S, E, N
    v = susc[np.isfinite(susc)]
    thr = [float(np.percentile(v, 50)), float(np.percentile(v, 75)), float(np.percentile(v, 90))]

    # --- relative-band PNG overlay (downsampled for the web) ---
    ds = susc[::downsample, ::downsample]
    rgba = np.zeros((*ds.shape, 4), np.uint8)
    fin = np.isfinite(ds)
    masks = {
        "severe": fin & (ds >= thr[2]),
        "high": fin & (ds >= thr[1]) & (ds < thr[2]),
        "moderate": fin & (ds >= thr[0]) & (ds < thr[1]),
        "low": fin & (ds < thr[0]),
    }
    for nm, m in masks.items():
        c = BAND_COLORS[nm]
        rgba[m] = (c[0], c[1], c[2], BAND_ALPHA[nm])
    plt.imsave(out / "risk.png", rgba)

    # --- hotspots enriched with band + the model's reasons + imagery flag ---
    why = expl.explain_city(city, data_dir=str(data_dir))
    why_by = {h["name"]: h for h in why["hotspots"]}
    feats = json.loads(Path(hotspots).read_text())["features"]
    with rasterio.open(data_dir / city / "feature_stack.tif") as fs:
        bnames = list(fs.descriptions)
        builtup = fs.read(bnames.index("builtup") + 1)

    out_feats = []
    with rasterio.open(surf) as s:
        for ft in feats:
            p = ft.get("properties", {})
            if p.get("city") != city or not ft.get("geometry"):
                continue
            lon, lat = ft["geometry"]["coordinates"][:2]
            r, c = expl._lonlat_to_rowcol(s, lon, lat)
            if not (0 <= r < susc.shape[0] and 0 <= c < susc.shape[1]):
                continue
            bu = builtup[r, c]
            checkable = np.isfinite(bu) and bu < 0.5
            w = why_by.get(p.get("name"), {})
            out_feats.append({
                "type": "Feature",
                "geometry": ft["geometry"],
                "properties": {
                    "name": p.get("name"),
                    "tier": p.get("tier"),
                    "band": _band_of(susc[r, c], thr),
                    "relative_score": w.get("relative_risk"),
                    "why": w.get("why", [])[:4],
                    "imagery": "satellite-checkable" if checkable else "street-level — radar can't confirm",
                },
            })
    (out / "hotspots.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": out_feats}, indent=1))

    # --- meta (real event dates + honest caveats) ---
    cfg = load_city(city, configs)
    events = sorted(({"name": e.name, "date": e.post_start} for e in cfg.events), key=lambda e: e["date"])
    meta = {
        "city": cfg.name, "slug": city,
        "bounds": {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3]},
        "surface": "street-waterlogging susceptibility (relative, quantile bands)",
        "bands": [{"name": n, "color": "#%02x%02x%02x" % BAND_COLORS[n]} for n in BAND_NAMES],
        "thresholds_pct": {"moderate": 50, "high": 75, "severe": 90},
        "events": events,
        "n_hotspots": len(out_feats),
        "caveats": [
            "Risk is RELATIVE within this city (quantile bands) from ~30 m data — not an absolute probability, not plot-level.",
            "Hotspots and flood history are presence-only and reporting-biased (news / traffic-police); an unmarked area is 'not on record', not 'safe'.",
            "Satellite radar sees open water (rivers, large junctions), not street water between buildings.",
            "Flood events are the dated satellite events we have (%s) — a small set, not every flood." % ", ".join(e["date"][:4] for e in events),
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"city": city, "n_hotspots": len(out_feats), "thresholds": [round(t, 3) for t in thr],
            "bounds": [round(b, 4) for b in bounds], "events": [e["date"] for e in events]}


def export_inference_grid(city, data_dir="data", out_dir="docs/web", step=38):
    """Precompute a coarse grid so the map can infer at ANY clicked point, not just hotspots.

    For each ~step-pixel cell we store its relative band and the model's top-3 signed feature
    contributions (tree SHAP) — enough to render the same evidence + SHAP chart anywhere.
    """
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(f"models/{city}_waterlog_model.json")
    booster = model.get_booster()
    d = Path(data_dir) / city
    with rasterio.open(d / "waterlog_susceptibility.tif") as s:
        susc = s.read(1)
        tr = s.transform
        h, w = s.height, s.width
    with rasterio.open(d / "feature_stack.tif") as fs:
        bn = list(fs.descriptions)
        stack = fs.read().astype("float32")[[bn.index(f) for f in FEATURES]]
    v = susc[np.isfinite(susc)]
    thr = [float(np.percentile(v, p)) for p in (50, 75, 90)]

    rc, X = [], []
    half = step // 2
    for r in range(half, h, step):
        for c in range(half, w, step):
            vals = stack[:, r, c]
            if np.isfinite(susc[r, c]) and np.isfinite(vals).all():
                rc.append((r, c)); X.append(vals)
    contribs = booster.predict(xgb.DMatrix(np.array(X, "float32")), pred_contribs=True)[:, :len(FEATURES)]

    cells = []
    for i, (r, c) in enumerate(rc):
        lng = tr.c + tr.a * (c + 0.5)
        lat = tr.f + tr.e * (r + 0.5)
        order = np.argsort(-np.abs(contribs[i]))[:3]
        top = [[int(j), round(float(contribs[i][j]), 3)] for j in order]
        cells.append({"lat": round(lat, 4), "lng": round(lng, 4), "band": _band_of(susc[r, c], thr), "top": top})

    grid = {
        "feature_names": list(FEATURES),
        "labels": [list(expl.PHRASES[f]) for f in FEATURES],  # [raises-phrase, lowers-phrase] per feature
        "step_m": round(abs(tr.a) * 111320 * 0.88 * step),
        "cells": cells,
    }
    (Path(out_dir) / city / "grid.json").write_text(json.dumps(grid))
    return {"city": city, "cells": len(cells), "step_m": grid["step_m"]}


def export_drains(city, data_dir="data", out_dir="docs/web", configs="configs/city"):
    """Export the major drain / canal / river network (OSM) as GeoJSON for a map overlay.

    Keeps named waterways (so the Najafgarh drain is labelled) plus any long unnamed segment;
    simplifies geometry to keep the file light. First cut of the 'drains as entities' idea —
    a structural layer the eye can read, not yet a routed catchment model.
    """
    import osmnx as ox
    from shapely.geometry import mapping

    cfg = load_city(city, configs)
    tags = {"waterway": ["drain", "canal", "river"]}
    g = ox.features_from_bbox(bbox=(cfg.west, cfg.south, cfg.east, cfg.north), tags=tags)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"])].to_crs(4326)

    feats = []
    for _, row in g.iterrows():
        geom = row.geometry
        name = row.get("name")
        wtype = row.get("waterway")
        named = isinstance(name, str) and name.strip()
        if not named and geom.length < 0.02:   # ~2 km (degrees); drop tiny unnamed ditches
            continue
        gs = geom.simplify(0.0004, preserve_topology=False)
        nm = name if named else None
        major = bool(named and ("najafgarh" in name.lower() or wtype in ("canal", "river")))
        feats.append({"type": "Feature", "geometry": mapping(gs),
                      "properties": {"name": nm, "waterway": wtype, "major": major}})

    out = Path(out_dir) / city
    out.mkdir(parents=True, exist_ok=True)
    (out / "drains.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    named_ct = sum(1 for f in feats if f["properties"]["name"])
    return {"city": city, "drains": len(feats), "named": named_ct}


def export_wards(city, data_dir="data", out_dir="docs/web", configs="configs/city"):
    """Rank districts by how many PEOPLE sit in high-risk zones (population x risk).

    Population (WorldPop, native grid = correct counts) intersected with the high+severe
    risk band. Honest: relative screening from a presence-only model at ~30 m + ~100 m
    population — a ranking of exposure, not a count of who will flood.
    """
    import rioxarray as rxr
    import osmnx as ox
    from rasterio.features import rasterize

    cfg = load_city(city, configs)
    g = ox.features_from_bbox(bbox=(cfg.west, cfg.south, cfg.east, cfg.north),
                              tags={"boundary": "administrative", "admin_level": ["8"]})
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(4326)

    d = Path(data_dir) / city
    with rasterio.open(d / "worldpop.tif") as p:
        pop = p.read(1).astype("float32")
        ptr = p.transform
        pshape = (p.height, p.width)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    susc = rxr.open_rasterio(d / "waterlog_susceptibility.tif").squeeze().rio.reproject_match(
        rxr.open_rasterio(d / "worldpop.tif").squeeze()).values
    fin = np.isfinite(susc)
    thr = float(np.percentile(susc[fin], 75))   # high + severe
    high = fin & (susc >= thr)

    # OSM admin levels are inconsistent in India — filter by AREA to keep district/zone-sized
    # units (drops country / state / whole-city polygons and tiny slivers); dedup by name.
    areas_km2 = (g.to_crs("ESRI:54009").geometry.area / 1e6).tolist()
    names = g["name"].tolist() if "name" in g.columns else [None] * len(g)
    geoms = g.geometry.tolist()
    rows, seen = [], set()
    for i, geom in enumerate(geoms):
        name = names[i]
        if not isinstance(name, str) or not name.strip():
            continue
        if not (8 <= areas_km2[i] <= 500):
            continue
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        m = rasterize([(geom, 1)], out_shape=pshape, transform=ptr, fill=0, all_touched=True).astype(bool)
        tot = float(pop[m].sum())
        if tot < 5000:
            continue
        ar = float(pop[m & high].sum())
        rows.append({"name": name.strip(), "people": int(round(tot)), "at_risk": int(round(ar)),
                     "pct": round(100 * ar / tot, 1)})
    rows.sort(key=lambda x: -x["at_risk"])
    (Path(out_dir) / city / "wards.json").write_text(
        json.dumps({"unit": "OSM district / zone (admin_level 8)", "wards": rows[:20]}, indent=1))
    return {"city": city, "wards": len(rows)}
