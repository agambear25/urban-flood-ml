"""Add per-cell WATERLOGGING risk + 'why' (from the waterlog model) and a monthly
flood-frequency series to each city's dashboard JSON. Run from repo root."""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import shap
import xgboost as xgb

from floodml import FEATURES

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
CITIES = ["delhi", "mumbai", "bengaluru", "chandigarh"]
GRID = 22

PHRASE = {
    "elevation": ("Low-lying ground", "Higher ground"),
    "slope": ("Flat, slow to drain", "Sloped, drains fast"),
    "curvature": ("Sits in a hollow", "On a ridge"),
    "local_relief": ("In a local dip", "On a local rise"),
    "hand": ("Close to drainage level", "Well above drainage"),
    "dist_river": ("Near a river", "Far from rivers"),
    "builtup": ("Densely built-up", "Built-up area"),
    "dist_drain": ("Near a storm drain", "Far from drains"),
    "drain_density": ("Sparse drainage nearby", "Dense drainage nearby"),
    "upstream_area": ("Lots of upstream runoff", "Little upstream runoff"),
    "sink_depth": ("Sits in a low spot / underpass", "Not a low spot"),
}


def level(p):
    return "Very high" if p >= 0.85 else "High" if p >= 0.70 else "Medium" if p >= 0.50 else "Low"


def pr_rank(a, valid):
    pr = np.full(a.shape, np.nan, "float32")
    v = a[valid]; o = v.argsort(); r = np.empty(len(o)); r[o] = np.linspace(0, 1, len(o)); pr[valid] = r
    return pr


def monthly_freq(slug):
    p = REPO / "events" / f"gdelt_timeline_{slug}.csv"
    if not p.exists():
        return None
    mv = {}
    for row in csv.DictReader(open(p)):
        d = row.get("date") or ""
        m = d[4:6] if len(d) >= 6 and d[:8].isdigit() else ""
        try:
            mv.setdefault(m, []).append(float(row["news_volume"]))
        except Exception:
            pass
    return [round(float(np.mean(mv.get(f"{i:02d}", [0]))), 4) for i in range(1, 13)]


for slug in CITIES:
    jp = ASSETS / f"{slug}.json"
    d = json.load(open(jp))
    ddir = REPO / "data" / slug
    wl = ddir / "waterlog_susceptibility.tif"
    if wl.exists():
        with rasterio.open(wl) as s:
            ws = s.read(1)
        with rasterio.open(ddir / "flood_mask.tif") as s:
            flood = s.read(1)
        with rasterio.open(ddir / "feature_stack.tif") as s:
            bands = list(s.descriptions); stack = s.read().astype("float32")
        stack = stack[[bands.index(f) for f in FEATURES]]
        valid = (flood != 255) & np.isfinite(ws) & np.isfinite(stack).all(0)
        pr = pr_rank(ws, valid)
        h, w = ws.shape; ch, cw = h / GRID, w / GRID

        model = xgb.XGBClassifier(); model.load_model(str(REPO / "models" / f"{slug}_waterlog_model.json"))
        rng = np.random.default_rng(0); vidx = np.argwhere(valid)
        take = vidx[rng.choice(len(vidx), min(15000, len(vidx)), replace=False)]
        X = np.stack([stack[b][take[:, 0], take[:, 1]] for b in range(len(FEATURES))], 1)
        sv = shap.TreeExplainer(model).shap_values(X)
        gr = (take[:, 0] / ch).astype(int).clip(0, GRID - 1)
        gc = (take[:, 1] / cw).astype(int).clip(0, GRID - 1)
        cpr = pr[take[:, 0], take[:, 1]]

        for c in d["cells"]:
            m = (gr == c["r"]) & (gc == c["c"])
            if m.sum() < 5:
                continue
            risk = float(np.nanmean(cpr[m])); mean = sv[m].mean(0)
            top = np.argsort(-np.abs(mean))[:4]
            c["wl_risk"] = round(risk, 3)
            c["wl_level"] = level(risk)
            c["wl_why"] = [{"factor": PHRASE[FEATURES[i]][0 if mean[i] > 0 else 1],
                            "value": round(float(mean[i]), 3),
                            "dir": "up" if mean[i] > 0 else "down"} for i in top]
        hi = cpr >= 0.85
        drv = sv[hi] if hi.sum() > 30 else sv
        dm = drv.mean(0); order = np.argsort(-np.abs(dm))[:5]
        d["wl_top_drivers"] = [{"factor": PHRASE[FEATURES[i]][0 if dm[i] > 0 else 1],
                                "value": round(float(abs(dm[i])), 3),
                                "dir": "up" if dm[i] > 0 else "down"} for i in order]
    fr = monthly_freq(slug)
    if fr:
        d["frequency"] = fr
    json.dump(d, open(jp, "w"))
    print(slug, "updated | freq:", "yes" if fr else "no", "| wl cells:",
          sum(1 for c in d["cells"] if "wl_level" in c))
