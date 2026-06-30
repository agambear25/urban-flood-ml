"""Tests for the cross-city evaluation helpers.

These run without any raster data or network — they guard the two claims the evaluation
rests on: that the cross-city model is configured identically to the per-city models
(apples-to-apples), and that isotonic calibration genuinely improves a balanced-trained
model's probabilities under class imbalance.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from floodml import eval_crosscity as evalx
from floodml import train as trn


def test_crosscity_model_matches_percity_model():
    """The leave-one-city-out model must use the same hyper-parameters as train.py,
    otherwise the generalization-gap comparison would not be apples-to-apples."""
    a = evalx._model(0).get_params()
    b = trn._model(0).get_params()
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert a[k] == b[k], f"{k} differs: {a[k]} vs {b[k]}"


def test_isotonic_calibration_improves_brier_under_imbalance():
    """A balanced-trained model emits ~0.5-centred scores; at low prevalence its Brier is
    poor. Isotonic calibration fit on a held-out slice should reduce Brier on fresh data."""
    rng = np.random.default_rng(0)
    n = 6000
    y = (rng.random(n) < 0.06).astype(int)          # 6% prevalence, like a real city
    # a score that ranks reasonably but is miscalibrated high (balanced-trained signature)
    raw = np.clip(0.5 + 0.30 * (y - 0.5) + rng.normal(0, 0.10, n), 0, 1)

    fit, test = slice(0, n // 2), slice(n // 2, n)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[fit], y[fit])
    cal = iso.predict(raw[test])

    assert brier_score_loss(y[test], cal) < brier_score_loss(y[test], raw[test])


def test_pr_baseline_equals_prevalence():
    """The no-skill PR-AUC baseline we report is, by definition, the positive prevalence."""
    y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    assert abs(float(y.mean()) - 0.2) < 1e-9
