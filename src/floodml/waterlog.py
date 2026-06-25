"""Urban-waterlogging susceptibility model — positive-only (PU) learning from documented hotspots.

Distinct from the SAR riverine model: positives are documented street/underpass waterlogging POINTS;
there are no confirmed-dry negatives, so we draw PU-style pseudo-negatives only from terrain that
genuinely should not pond (far from any hotspot, on local rises, not a sink, not in the SAR flood).

Honest: positive-only labels -> a *relative ranking*, not a calibrated probability; recall is
unmeasurable and absence of a hotspot != safe (the hotspot list is media/advisory-derived).
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import rasterio
import xgboost as xgb
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import FEATURES
from .config import CityConfig


def train_waterlog(cfg: CityConfig, data_dir: str | Path, model_dir: str | Path,
                   tracking_uri: str | None = None, seed: int = 0) -> dict:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    lab_path = data_dir / "waterlog_label.tif"
    if not lab_path.exists():
        raise FileNotFoundError("no waterlog_label.tif — run events/build_events.py first")

    with rasterio.open(lab_path) as s:
        lab = s.read(1); prof = s.profile; tr = s.transform
    with rasterio.open(data_dir / "flood_mask.tif") as s:
        flood = s.read(1)
    with rasterio.open(data_dir / "feature_stack.tif") as s:
        bands = list(s.descriptions); stack = s.read().astype("float32")
    stack = stack[[bands.index(f) for f in FEATURES]]

    valid = (flood != 255) & np.isfinite(stack).all(axis=0)
    pos = (lab == 1) & valid
    npos = int(pos.sum())
    if npos < 50:
        raise ValueError(f"only {npos} hotspot pixels — too few")

    px_m = abs(tr.a) * 111_320 * np.cos(np.deg2rad(cfg.lat_mid))
    dist = distance_transform_edt(lab != 1) * px_m
    relief = stack[FEATURES.index("local_relief")]
    sink = stack[FEATURES.index("sink_depth")]
    r_hi = np.nanpercentile(relief[valid], 60)
    s_lo = np.nanpercentile(sink[valid], 40)
    # pseudo-negatives: far from hotspots, on a rise, not a sink, not in the SAR flood
    neg_ok = valid & (dist > 1000) & (flood != 1) & (relief > r_hi) & (sink < s_lo)

    rng = np.random.default_rng(seed)
    pos_idx = np.argwhere(pos)
    neg_idx = np.argwhere(neg_ok)
    if len(neg_idx) < npos:
        neg_idx = np.argwhere(valid & (dist > 1000) & (flood != 1))
    k = min(len(pos_idx) * 3, len(neg_idx))
    neg_idx = neg_idx[rng.choice(len(neg_idx), k, replace=False)]
    idx = np.vstack([pos_idx, neg_idx])
    y = np.r_[np.ones(len(pos_idx), int), np.zeros(len(neg_idx), int)]
    X = np.stack([stack[b][idx[:, 0], idx[:, 1]] for b in range(len(FEATURES))], axis=1)

    bh, bw = max(lab.shape[0] // 5, 1), max(lab.shape[1] // 5, 1)
    groups = (idx[:, 0] // bh) * 10 + (idx[:, 1] // bw)
    spw = len(neg_idx) / max(len(pos_idx), 1)

    def mk():
        return xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
                                 colsample_bytree=0.9, n_jobs=4, eval_metric="logloss",
                                 random_state=seed, scale_pos_weight=spw)

    nfold = min(5, len(set(groups)))
    spat = [roc_auc_score(y[te], mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            for tr, te in GroupKFold(nfold).split(X, y, groups)]

    model = mk().fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg.slug}_waterlog_model.json"
    model.save_model(model_path)

    # full-grid susceptibility surface
    h, w = lab.shape
    flat = stack.reshape(len(FEATURES), -1).T
    vflat = valid.reshape(-1)
    susc = np.full(h * w, np.nan, dtype="float32")
    susc[vflat] = model.predict_proba(flat[vflat])[:, 1]
    susc = susc.reshape(h, w)
    sp = prof.copy(); sp.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(data_dir / "waterlog_susceptibility.tif", "w", **sp) as dst:
        dst.write(susc, 1)

    importance = {f: float(v) for f, v in zip(FEATURES, model.feature_importances_, strict=False)}
    metrics = {"city": cfg.slug, "spatial_cv_auc_pu": float(np.mean(spat)),
               "n_positives": npos, "n_negatives": int(len(neg_idx)), "importance": importance}

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("urban-flood-ml")
    with mlflow.start_run(run_name=cfg.slug + "-waterlog"):
        mlflow.set_tag("city", cfg.slug); mlflow.set_tag("model", "waterlog")
        mlflow.log_metrics({"spatial_cv_auc_pu": metrics["spatial_cv_auc_pu"], "n_positives": npos})
        mlflow.log_dict(importance, "waterlog_importance.json")
    return metrics
