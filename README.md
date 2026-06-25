# 🌊 urban-flood-ml — which neighbourhoods flood, and why

**A tool that maps flood risk across Indian cities using free satellite data** — so planners, emergency teams, and residents can see the most at-risk areas at a glance. Today it covers **Delhi, Mumbai, Bengaluru, and Chandigarh**, and it's built to scale to any city.

### ▶️ [Try the live interactive map →](https://agambear25.github.io/urban-flood-ml/)
Switch cities, drag the rainfall scenario, and **click any neighbourhood to see *why* it floods.**

![Flood risk maps for four cities](docs/multicity_susceptibility.png)

---

## What it does

- 🛰️ **Spots real floods from space.** It reads satellite radar to map where a city *actually* flooded — and radar sees through monsoon cloud, when normal satellites are blind.
- 🗺️ **Predicts flood-prone areas.** A model learns the ground pattern of those flooded places (low, flat, near drainage…) and shades every neighbourhood by how flood-prone it is.
- 🏙️ **Runs any city from a single setup file.** The same pipeline maps four cities today; adding a new one is one small config file, not new code.
- 🚧 **Tracks *street* waterlogging the satellite can't see.** A traceable, multi-source [event database](events/) — geocoded chronic hotspots (Minto Bridge, Pul Prahladpur…) + news-mined flood frequency — shown live on the map and usable as model labels.

## Why it stands out

- **Four cities, one reusable engine** — not four copy-pasted projects.
- Every city's map is **checked against a real past flood**, and correctly separates flood-prone from safe ground **85–93% of the time**.
- **Built like real software** — an installable tool you run with one command, with automated tests and experiment tracking — not a throwaway script.
- **Honest about what it can't do** (see below) instead of overpromising.

## How it works, in plain English

1. **Find** — satellite radar spots where water sat during a past flood.
2. **Learn** — a model studies the terrain at those flooded spots (Is it low? Flat? Near a drain?).
3. **Map** — it shades every *other* spot that shares the same risky pattern.

## Honest about its limits

This is an **experimental planning tool, not an official flood warning system.** It detects river and open-water flooding well, but it **can't see shallow street-flooding between buildings**. Risk is *relative* within each city and validated on *one* past flood. Knowing and stating these limits is part of doing it right.

## Built with

`Python` · `Google Earth Engine` (free satellite data) · `machine learning (XGBoost)` · `OpenStreetMap` · `MLflow`

*Companion project: [yamuna-flood-mapper](https://github.com/agambear25/yamuna-flood-mapper) — the original Delhi case study with an interactive dashboard, which this engine grew out of.*

---

<details>
<summary><b>🔧 Technical details</b> (for engineers — click to expand)</summary>

### Pipeline
Config-driven, one command per stage:
```bash
floodml run delhi   # SAR flood mask → fetch layers → features → train → predict
```
- **Flood labels** — server-side Sentinel-1 GRD change detection in Google Earth Engine.
- **Features** — terrain (elevation, slope, HAND, curvature, local relief, sinks) + a uniform drainage backbone (MERIT-Hydro HAND/upstream-area + OSM drains → distance-to-drain, drainage density).
- **Model** — per-city XGBoost, validated with **spatial block cross-validation** (the honest metric; random CV inflates ~0.05 via spatial leakage).

### Results (AUC)

| City | Spatial-CV AUC | Random-CV AUC |
|---|---|---|
| Bengaluru | 0.927 | 0.966 |
| Delhi | 0.918 | 0.972 |
| Mumbai | 0.862 | 0.922 |
| Chandigarh | 0.850 | 0.898 |

### Engineering
- Installable `floodml` package (`src/`) with a **Typer CLI** and **Pydantic** per-city configs.
- **MLflow** experiment tracking (every run's params/metrics/feature-importance, tagged by city).
- **GitHub Actions CI** — ruff lint + offline pytest on every push.
- Deliberately *not* used: Kubernetes, Airflow, feature stores (over-engineering for a 4-city project).

### Honest caveats (detail)
- SAR labels capture riverine/open-water flooding, not in-street urban waterlogging — so the dense-city models are *susceptibility*, not street-flood predictors.
- `built-up` ranks high partly as a SAR blind-spot (radar can't see flooding inside built-up areas).
- Open OSM drainage data is thin, so drainage features contribute marginally.

### Reproduce
```bash
conda env create -f environment.yml && conda activate urban-flood-ml
floodml run delhi      # then chandigarh / mumbai / bengaluru
python scripts/overview.py
```
Needs a free Google Earth Engine account; set `ee_project` in each `configs/city/*.yaml`.

### Roadmap
- A calibrated **SFINCS** 2D hydrodynamic flood simulation for Delhi (validated against the 2023 flood) + a Mumbai tide+rain scenario.
- Bengaluru's official BBMP drainage network as extra features.
- Rainfall-forecast overlay (Open-Meteo + GPM IMERG) for dynamic, rainfall-conditioned risk.

</details>
