"""Train a per-city XGBoost susceptibility model with honest spatial cross-validation."""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import rasterio
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import FEATURES
from .config import CityConfig


def _load(data_dir: Path):
    with rasterio.open(data_dir / "flood_mask.tif") as s:
        flood = s.read(1)
    with rasterio.open(data_dir / "feature_stack.tif") as s:
        bands = list(s.descriptions)
        stack = s.read().astype("float32")
    order = [bands.index(f) for f in FEATURES]
    return flood, stack[order]


def _model(seed: int):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, n_jobs=4, eval_metric="logloss", random_state=seed,
    )


def train_city(cfg: CityConfig, data_dir: str | Path, model_dir: str | Path,
               tracking_uri: str | None = None, n: int = 40000, seed: int = 0) -> dict:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    flood, stack = _load(data_dir)
    valid = (flood != 255) & np.isfinite(stack).all(axis=0)

    rng = np.random.default_rng(seed)
    flood_px = np.argwhere(valid & (flood == 1))
    dry_px = np.argwhere(valid & (flood == 0))
    if len(flood_px) < 200:
        raise ValueError(f"only {len(flood_px)} flood pixels — too few to train reliably")
    k = min(n, len(flood_px))
    flood_px = flood_px[rng.choice(len(flood_px), k, replace=False)]
    dry_px = dry_px[rng.choice(len(dry_px), min(k, len(dry_px)), replace=False)]
    idx = np.vstack([flood_px, dry_px])
    y = np.r_[np.ones(len(flood_px), int), np.zeros(len(dry_px), int)]
    X = np.stack([stack[b][idx[:, 0], idx[:, 1]] for b in range(len(FEATURES))], axis=1)

    bh, bw = max(flood.shape[0] // 5, 1), max(flood.shape[1] // 5, 1)
    groups = (idx[:, 0] // bh) * 10 + (idx[:, 1] // bw)

    rand = [roc_auc_score(y[te], _model(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)]
    spat = [roc_auc_score(y[te], _model(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            for tr, te in GroupKFold(5).split(X, y, groups)]

    model = _model(seed).fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg.slug}_model.json"
    model.save_model(model_path)

    metrics = {
        "random_cv_auc": float(np.mean(rand)),
        "spatial_cv_auc": float(np.mean(spat)),
        "spatial_cv_std": float(np.std(spat)),
        "n_samples": int(len(y)),
    }
    importance = {f: float(v) for f, v in zip(FEATURES, model.feature_importances_, strict=False)}

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("urban-flood-ml")
    with mlflow.start_run(run_name=cfg.slug):
        mlflow.set_tag("city", cfg.slug)
        mlflow.log_params({"model": "xgboost", "n_estimators": 300, "max_depth": 4,
                           "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(importance, "feature_importance.json")

    return {"city": cfg.slug, **metrics, "model_path": str(model_path), "importance": importance}
