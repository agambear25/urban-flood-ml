"""Build a 4-city susceptibility overview figure + a metrics summary table.

Run from the repo root after `floodml run` for each city:  python scripts/overview.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"; DOCS.mkdir(exist_ok=True)
RESULTS = REPO / "results"

CITIES = ["delhi", "chandigarh", "mumbai", "bengaluru"]

# --- overview figure ---
fig, axes = plt.subplots(2, 2, figsize=(13, 13))
for ax, city in zip(axes.ravel(), CITIES):
    sp = REPO / "data" / city / "susceptibility.tif"
    m = json.loads((RESULTS / f"{city}_metrics.json").read_text())
    with rasterio.open(sp) as s:
        susc = s.read(1)
    ax.imshow(np.ma.masked_invalid(susc), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(f"{city.title()}  —  spatial-CV AUC {m['spatial_cv_auc']:.3f}", fontsize=13)
    ax.axis("off")
fig.suptitle("Urban flood susceptibility — XGBoost (SAR-labelled), 4 cities", fontsize=16, y=0.99)
plt.tight_layout()
plt.savefig(DOCS / "multicity_susceptibility.png", dpi=130, bbox_inches="tight", facecolor="white")
print("wrote docs/multicity_susceptibility.png")

# --- summary csv ---
rows = ["city,spatial_cv_auc,random_cv_auc,spatial_cv_std,n_samples"]
for city in CITIES:
    m = json.loads((RESULTS / f"{city}_metrics.json").read_text())
    rows.append(f"{city},{m['spatial_cv_auc']:.3f},{m['random_cv_auc']:.3f},"
                f"{m['spatial_cv_std']:.3f},{m['n_samples']}")
(RESULTS / "summary.csv").write_text("\n".join(rows) + "\n")
print("wrote results/summary.csv")
print("\n".join(rows))
