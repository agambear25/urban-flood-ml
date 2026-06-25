"""Render the urban-waterlogging susceptibility overlay per city and add it to the
dashboard JSON (fast; no re-run of the heavy pipeline). Run from repo root."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
OV = ASSETS / "overlays"
CITIES = ["delhi", "mumbai", "bengaluru", "chandigarh"]


def render(slug):
    p = REPO / "data" / slug / "waterlog_susceptibility.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as s:
        a = s.read(1)
    valid = np.isfinite(a)
    pr = np.full(a.shape, np.nan, "float32")
    v = a[valid]; o = v.argsort(); r = np.empty(len(o)); r[o] = np.linspace(0, 1, len(o)); pr[valid] = r
    cmap = matplotlib.colormaps["BuPu"]
    cut = 0.55
    mask = np.isfinite(pr) & (pr >= cut)
    norm = np.clip((pr - cut) / (1 - cut), 0, 1)
    col = (cmap(np.nan_to_num(norm)) * 255).astype("uint8")
    rgba = np.zeros((*a.shape, 4), "uint8")
    rgba[..., :3] = col[..., :3]
    rgba[..., 3] = np.where(mask, 205, 0).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(OV / f"{slug}_waterlog.png")
    return f"assets/overlays/{slug}_waterlog.png"


for slug in CITIES:
    path = render(slug)
    jp = ASSETS / f"{slug}.json"
    d = json.load(open(jp))
    if path:
        d["waterlog_overlay"] = path
    json.dump(d, open(jp, "w"))
    print(slug, "->", path)
