"""Apply a trained model to every pixel -> a susceptibility surface GeoTIFF."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xgboost as xgb

from . import FEATURES


def predict_city(data_dir: str | Path, model_path: str | Path) -> Path:
    data_dir = Path(data_dir)
    with rasterio.open(data_dir / "feature_stack.tif") as s:
        bands = list(s.descriptions)
        stack = s.read().astype("float32")
        profile = s.profile
    with rasterio.open(data_dir / "flood_mask.tif") as s:
        flood = s.read(1)

    order = [bands.index(f) for f in FEATURES]
    stack = stack[order]
    h, w = flood.shape
    valid = (flood != 255) & np.isfinite(stack).all(axis=0)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    flat = stack.reshape(len(FEATURES), -1).T
    vflat = valid.reshape(-1)
    proba = model.predict_proba(flat[vflat])[:, 1]

    susc = np.full(h * w, np.nan, dtype="float32")
    susc[vflat] = proba
    susc = susc.reshape(h, w)

    out = data_dir / "susceptibility.tif"
    sp = profile.copy()
    sp.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(out, "w", **sp) as dst:
        dst.write(susc, 1)
    return out
