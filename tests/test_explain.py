"""Tests for the per-location explanation helper (no rasters / model needed)."""
import numpy as np

from floodml import FEATURES
from floodml import explain as expl


def test_phrases_cover_every_feature():
    """Every model feature must have a plain-English phrase, or its explanation breaks."""
    for f in FEATURES:
        assert f in expl.PHRASES
        assert len(expl.PHRASES[f]) == 2


def test_explain_one_ranks_and_signs_correctly():
    """Biggest absolute contribution comes first; sign picks the raises/lowers phrasing."""
    contribs = np.zeros(len(FEATURES))
    contribs[FEATURES.index("sink_depth")] = 0.9      # strongest, raises
    contribs[FEATURES.index("builtup")] = 0.4         # raises
    contribs[FEATURES.index("dist_drain")] = -0.6     # lowers, 2nd strongest
    values = np.zeros(len(FEATURES))
    values[FEATURES.index("builtup")] = 0.8           # 80% built-up

    why = expl.explain_one(contribs, values, k=3)
    assert [w["factor"] for w in why] == ["sink_depth", "dist_drain", "builtup"]
    assert why[0]["direction"] == "raises"
    assert why[1]["direction"] == "lowers"
    assert "80% built-up" in why[2]["plain"]


def test_fmt_value_units():
    assert expl._fmt_value("dist_drain", 1500) == "1.5 km"
    assert expl._fmt_value("dist_drain", 300) == "300 m"
    assert expl._fmt_value("slope", 2.0) == "2.0°"
    assert expl._fmt_value("elevation", 210) == ""   # not intuitive -> omitted
