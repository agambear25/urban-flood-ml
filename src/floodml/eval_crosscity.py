"""Leave-one-city-out evaluation of the riverine susceptibility model.

Within-city cross-validation (what `train.py` reports) answers a soft question:
"does the model work in held-out *areas of the same city* it was trained on?"
This module answers the hard, honest one: **"does it work in a city it has never seen?"**
Train on N-1 cities, test on the held-out one. That is the real test of whether the model
learned where water collects, or merely memorised one city.

It also scores the model the way a *rare-event* problem must be scored — which the headline
ROC-AUC quietly hides:

* **PR-AUC** (average precision), reported against the no-skill baseline = the test city's
  true flood prevalence. ROC-AUC barely moves when positives are rare; PR-AUC tells the truth.
* **Calibration.** The model trains on a balanced 50/50 flood/dry sample, so its raw scores
  are not probabilities at a city's true (low) prevalence. We fit an isotonic calibrator on
  the *training* cities and test whether it transfers to the unseen city (reliability curve
  + Brier score, before vs after).

Everything is deterministic given the seed, and reads only data already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from . import FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CITIES = ["delhi", "mumbai", "bengaluru", "chandigarh"]


def _model(seed: int = 0) -> xgb.XGBClassifier:
    # identical hyper-parameters to train.py, so the comparison is apples-to-apples
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, n_jobs=4, eval_metric="logloss", random_state=seed,
    )


def load_city(data_dir: Path, slug: str, n_train: int = 40000, n_test: int = 300000,
              seed: int = 0) -> dict:
    """Sample one city: a balanced train set (like train.py) + a natural-prevalence test set.

    The balanced set is for training and for within-city spatial CV; the natural-prevalence
    set (a random draw over ALL valid pixels) is what we evaluate on, so PR-AUC, prevalence
    and calibration reflect the real, rare-event distribution rather than a 50/50 fiction.
    """
    d = Path(data_dir) / slug
    with rasterio.open(d / "flood_mask.tif") as s:
        flood = s.read(1)
    with rasterio.open(d / "feature_stack.tif") as s:
        bands = list(s.descriptions)
        stack = s.read().astype("float32")
    stack = stack[[bands.index(f) for f in FEATURES]]

    valid = (flood != 255) & np.isfinite(stack).all(axis=0)
    rng = np.random.default_rng(seed)

    # --- balanced training sample (mirrors train.py) ---
    fpx = np.argwhere(valid & (flood == 1))
    dpx = np.argwhere(valid & (flood == 0))
    if len(fpx) < 200:
        raise ValueError(f"{slug}: only {len(fpx)} flood pixels — too few")
    k = min(n_train, len(fpx))
    fpx = fpx[rng.choice(len(fpx), k, replace=False)]
    dpx = dpx[rng.choice(len(dpx), min(k, len(dpx)), replace=False)]
    coords = np.vstack([fpx, dpx])
    y_tr = np.r_[np.ones(len(fpx), int), np.zeros(len(dpx), int)]
    X_tr = np.stack([stack[b][coords[:, 0], coords[:, 1]] for b in range(len(FEATURES))], axis=1)

    # --- natural-prevalence sample (random over all valid pixels) ---
    flat_valid = np.flatnonzero(valid.reshape(-1))
    m = min(n_test, len(flat_valid))
    sel = flat_valid[rng.choice(len(flat_valid), m, replace=False)]
    flood_flat = flood.reshape(-1)
    stack_flat = stack.reshape(len(FEATURES), -1)
    X_nat = stack_flat[:, sel].T.astype("float32")
    y_nat = (flood_flat[sel] == 1).astype(int)

    return {
        "X_tr": X_tr, "y_tr": y_tr, "coords": coords, "shape": flood.shape,
        "X_nat": X_nat, "y_nat": y_nat,
    }


def _within_city_spatial_auc(city: dict, seed: int = 0) -> float:
    """Reproduce train.py's spatial block CV ROC-AUC for one city (the 'easy' number)."""
    X, y, coords, (h, w) = city["X_tr"], city["y_tr"], city["coords"], city["shape"]
    bh, bw = max(h // 5, 1), max(w // 5, 1)
    groups = (coords[:, 0] // bh) * 10 + (coords[:, 1] // bw)
    scores = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        p = _model(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores))


def crosscity_eval(data_dir: str | Path = "data", results_dir: str | Path = "results",
                   fig_dir: str | Path = "docs/eval", cities: list[str] | None = None,
                   seed: int = 0) -> dict:
    cities = cities or CITIES
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = {c: load_city(data_dir, c, seed=seed) for c in cities}
    within = {c: _within_city_spatial_auc(loaded[c], seed) for c in cities}

    rows, pr_data, rel_data = [], {}, {}
    for test_c in cities:
        train_cs = [c for c in cities if c != test_c]
        X_tr = np.vstack([loaded[c]["X_tr"] for c in train_cs])
        y_tr = np.concatenate([loaded[c]["y_tr"] for c in train_cs])
        X_cal = np.vstack([loaded[c]["X_nat"] for c in train_cs])
        y_cal = np.concatenate([loaded[c]["y_nat"] for c in train_cs])
        X_te, y_te = loaded[test_c]["X_nat"], loaded[test_c]["y_nat"]

        base = _model(seed).fit(X_tr, y_tr)
        p_raw = base.predict_proba(X_te)[:, 1]

        # isotonic calibration learned on the TRAINING cities (natural prevalence),
        # then applied to the unseen test city — does calibration transfer?
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            base.predict_proba(X_cal)[:, 1], y_cal)
        p_cal = iso.predict(p_raw)

        prevalence = float(y_te.mean())
        row = {
            "held_out_city": test_c,
            "prevalence": prevalence,
            "within_city_spatial_roc_auc": within[test_c],
            "crosscity_roc_auc": float(roc_auc_score(y_te, p_raw)),
            "crosscity_pr_auc": float(average_precision_score(y_te, p_raw)),
            "pr_auc_baseline": prevalence,            # a no-skill model scores this
            "brier_raw": float(brier_score_loss(y_te, p_raw)),
            "brier_calibrated": float(brier_score_loss(y_te, p_cal)),
            "n_test": int(len(y_te)),
            "n_train": int(len(y_tr)),
        }
        row["generalization_gap"] = round(row["within_city_spatial_roc_auc"]
                                          - row["crosscity_roc_auc"], 4)
        rows.append(row)

        prec, rec, _ = precision_recall_curve(y_te, p_raw)
        pr_data[test_c] = (rec, prec, row["crosscity_pr_auc"], prevalence)
        rel_data[test_c] = (
            calibration_curve(y_te, np.clip(p_raw, 0, 1), n_bins=10, strategy="quantile"),
            calibration_curve(y_te, np.clip(p_cal, 0, 1), n_bins=10, strategy="quantile"),
            row["brier_raw"], row["brier_calibrated"],
        )

    out = {"task": "leave-one-city-out (riverine SAR model)", "seed": seed,
           "features": FEATURES, "results": rows}
    (results_dir / "crosscity_river.json").write_text(json.dumps(out, indent=2))
    _write_summary_csv(results_dir / "crosscity_summary.csv", rows)
    _plot_pr(fig_dir / "crosscity_pr_curves.png", pr_data)
    _plot_reliability(fig_dir / "crosscity_reliability.png", rel_data)
    _plot_gap(fig_dir / "crosscity_gap.png", rows)
    return out


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    cols = ["held_out_city", "prevalence", "within_city_spatial_roc_auc",
            "crosscity_roc_auc", "generalization_gap", "crosscity_pr_auc",
            "pr_auc_baseline", "brier_raw", "brier_calibrated"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _plot_pr(path: Path, pr_data: dict) -> None:
    plt.figure(figsize=(7, 5.5))
    for city, (rec, prec, ap, prev) in pr_data.items():
        line, = plt.plot(rec, prec, lw=2, label=f"{city.title()}  (PR-AUC {ap:.2f})")
        plt.axhline(prev, ls=":", lw=1, color=line.get_color(), alpha=.6)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Cross-city precision–recall (held-out city)\n"
              "dotted = no-skill baseline (that city's flood prevalence)")
    plt.legend(loc="upper right", fontsize=9); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_reliability(path: Path, rel_data: dict) -> None:
    n = len(rel_data)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes = np.array(axes).reshape(-1)
    for ax, (city, ((yr, xr), (yc, xc), br, bc)) in zip(axes, rel_data.items()):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="perfect")
        ax.plot(xr, yr, "o-", color="#c0392b", label=f"raw (Brier {br:.3f})")
        ax.plot(xc, yc, "s-", color="#2980b9", label=f"calibrated (Brier {bc:.3f})")
        ax.set_title(f"{city.title()} — unseen"); ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    for ax in axes[len(rel_data):]:
        ax.set_visible(False)
    fig.suptitle("Reliability on the unseen city — balanced-trained scores vs isotonic-calibrated",
                 y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_gap(path: Path, rows: list[dict]) -> None:
    cities = [r["held_out_city"].title() for r in rows]
    within = [r["within_city_spatial_roc_auc"] for r in rows]
    cross = [r["crosscity_roc_auc"] for r in rows]
    pr = [r["crosscity_pr_auc"] for r in rows]
    x = np.arange(len(cities)); w = 0.27
    plt.figure(figsize=(8, 5))
    plt.bar(x - w, within, w, label="within-city spatial CV (ROC-AUC)", color="#7f8c8d")
    plt.bar(x, cross, w, label="cross-city, unseen (ROC-AUC)", color="#2c3e50")
    plt.bar(x + w, pr, w, label="cross-city, unseen (PR-AUC)", color="#e67e22")
    plt.xticks(x, cities); plt.ylabel("score"); plt.ylim(0, 1)
    plt.title("The generalization gap: same-city vs unseen-city")
    plt.legend(fontsize=9); plt.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
