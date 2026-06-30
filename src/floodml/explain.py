"""Why is this spot risky? — per-location reasons from the model itself.

For any place, the model can say *why* it flagged it. We use XGBoost's exact per-prediction
feature contributions (`pred_contribs`, i.e. tree SHAP — no extra dependency): for one
location they sum to the model's score, so each feature gets a signed share of "how much it
pushed this spot's risk up or down." We turn the biggest few into plain English.

We run it on the documented, named waterlogging hotspots (Minto Bridge, ITO, …) so the
output reads like "Minto Bridge is risky because: it sits in a pronounced low point, it's
heavily paved, …" — the kind of thing you can sanity-check against what you already know.

Note: the street-waterlogging model is positive-only, so its score is a *relative risk
ranking*, not a probability. The reasons are still meaningful — they explain the ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import xgboost as xgb
from rasterio.warp import transform as warp_transform

from . import FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# For each feature: (phrase when it RAISED the risk, phrase when it LOWERED the risk).
PHRASES: dict[str, tuple[str, str]] = {
    "elevation":     ("low-lying overall",                         "on higher ground overall"),
    "slope":         ("flat ground that water can't run off",      "sloped enough to shed water"),
    "curvature":     ("a concave hollow that collects water",      "convex ground that sheds water"),
    "local_relief":  ("lower than the area around it",             "a local high point"),
    "hand":          ("low above the nearest drainage",            "well above the nearest drainage"),
    "dist_river":    ("near a river or large channel",             "far from any river"),
    "builtup":       ("heavily built-up and paved",                "fairly open and unpaved"),
    "dist_drain":    ("far from any mapped drain",                 "close to a mapped drain"),
    "drain_density": ("few drains nearby",                         "plenty of drains nearby"),
    "upstream_area": ("a lot of land drains toward it",            "little land drains toward it"),
    "sink_depth":    ("a pronounced bowl / low point (underpass-like)", "not a low point"),
}


def _fmt_value(feature: str, v: float) -> str:
    """A plain, units-aware rendering of a feature's raw value (empty if not intuitive)."""
    if feature in ("dist_river", "dist_drain"):
        return f"{v/1000:.1f} km" if v >= 1000 else f"{v:.0f} m"
    if feature == "slope":
        return f"{v:.1f}°"
    if feature in ("hand", "local_relief", "sink_depth"):
        return f"{v:.1f} m"
    if feature == "builtup":
        return f"{min(max(v, 0), 1)*100:.0f}% built-up" if v <= 1 else f"{v:.0f}% built-up"
    return ""


def explain_one(contribs: np.ndarray, values: np.ndarray, k: int = 4) -> list[dict]:
    """Pure helper: turn one location's signed feature contributions into plain reasons.

    `contribs` and `values` are length-len(FEATURES), aligned to FEATURES order.
    Returns up to k factors, largest absolute effect first.
    """
    order = np.argsort(-np.abs(contribs))[:k]
    out = []
    for i in order:
        f = FEATURES[i]
        raises = contribs[i] > 0
        phrase = PHRASES[f][0 if raises else 1]
        val = _fmt_value(f, float(values[i]))
        out.append({
            "factor": f,
            "direction": "raises" if raises else "lowers",
            "plain": phrase + (f" ({val})" if val else ""),
            "contribution": round(float(contribs[i]), 4),
        })
    return out


def _lonlat_to_rowcol(src, lon: float, lat: float):
    """Map a WGS84 lon/lat to a raster row/col, reprojecting if the raster isn't in 4326."""
    if src.crs and src.crs.to_epsg() != 4326:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    else:
        x, y = lon, lat
    row, col = src.index(x, y)
    return int(row), int(col)


def explain_city(city: str, data_dir: str | Path = "data", model_dir: str | Path = "models",
                 hotspots: str | Path = "events/hotspots.geojson",
                 results_dir: str | Path = "results", fig_dir: str | Path = "docs/eval",
                 top: int = 8) -> dict:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / f"{city}_waterlog_model.json"))
    booster = model.get_booster()

    fs = data_dir / city / "feature_stack.tif"
    with rasterio.open(fs) as src:
        bands = list(src.descriptions)
        stack = src.read().astype("float32")[[bands.index(f) for f in FEATURES]]
        h, w = src.height, src.width
        pts = json.loads(Path(hotspots).read_text())["features"]
        spots = []
        for ft in pts:
            if ft.get("properties", {}).get("city") != city or not ft.get("geometry"):
                continue
            lon, lat = ft["geometry"]["coordinates"][:2]
            r, c = _lonlat_to_rowcol(src, lon, lat)
            if not (0 <= r < h and 0 <= c < w):
                continue
            vals = stack[:, r, c]
            if not np.isfinite(vals).all():
                continue
            spots.append((ft["properties"].get("name", "?"),
                          ft["properties"].get("tier", ""), vals))

    if not spots:
        raise ValueError(f"no usable hotspots for {city}")

    X = np.stack([v for _, _, v in spots]).astype("float32")
    risk = model.predict_proba(X)[:, 1]
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :len(FEATURES)]

    rows = []
    for (name, tier, vals), p, ctr in zip(spots, risk, contribs):
        rows.append({"name": name, "tier": tier, "relative_risk": round(float(p), 3),
                     "why": explain_one(ctr, vals)})
    rows.sort(key=lambda d: -d["relative_risk"])

    out = {"city": city, "model": "street-waterlogging (relative ranking, not a probability)",
           "n_hotspots": len(rows), "hotspots": rows}
    (results_dir / f"{city}_why.json").write_text(json.dumps(out, indent=2))
    _plot_why(fig_dir / f"why_{city}.png", city, rows[:6])
    return out


def _plot_why(path: Path, city: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, row in zip(axes, rows):
        fac = row["why"][::-1]  # smallest at bottom
        labels = [f["plain"] for f in fac]
        vals = [f["contribution"] for f in fac]
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{row['name']}  (risk {row['relative_risk']:.2f})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Why these {city.title()} spots are flagged — the model's own reasons\n"
                 "red = raises the risk · blue = lowers it", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
