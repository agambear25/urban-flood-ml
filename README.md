# urban-flood-ml

**An open tool that maps where Indian cities flood — so the people who live in the most at-risk streets can be warned to prepare.**

Most flood maps stop at the river. But the flooding that actually traps people is the *street waterlogging* — the underpass that fills up, the junction that becomes a lake every monsoon. This project maps **both**, for four cities, from free satellite and public data — built so that, with a rainfall trigger on top, it could one day **alert residents of high-risk neighbourhoods before the water arrives**. It's a public-good project, not a commercial one.

### ▶️ [Open the live map →](https://agambear25.github.io/urban-flood-ml/)
Switch city, switch flood type (river vs. street waterlogging), and **click a named hotspot** (Minto Bridge, ITO, Hebbal…) to locate it.

![Flood risk across four cities](docs/multicity_susceptibility.png)

---

## What it does

- 🛰️ **Finds where floods actually happened** — reads Sentinel-1 satellite radar (which sees through monsoon cloud) across *multiple* historical floods per city.
- 🗺️ **Maps two kinds of flooding** — a **river/floodplain** model *and* a separate **street-waterlogging** model (the ponding in low spots and underpasses that the satellite can't see).
- 📍 **Names the danger spots** — a traceable database of documented waterlogging hotspots, shown on the map by name.
- 🏙️ **Covers 4 cities, scales to any** — Delhi, Mumbai, Bengaluru, Chandigarh today; a new city is one config file, not new code.

**The point:** translate "this area is flood-prone" into "*these specific streets*, and *these people*" — the first step toward warning residents to move vehicles, clear drains, and prepare.

## How it works, in plain terms

1. **Find** — satellite radar shows where water sat during past floods → that's the training signal.
2. **Learn** — a model studies the ground at those flooded spots (low? flat? a dip? near a drain?) and learns the pattern.
3. **Map** — it shades every *other* place that shares the risky pattern, and overlays the named hotspots people already know flood.

A rainfall forecast on top turns this static risk into a *"prepare today"* alert — the natural next step.

## Honest about its limits

This is an **experimental planning aid, not an official warning system.** River flooding is detected well; in-street waterlogging is harder (the satellite barely sees it, so that model leans on documented hotspots + terrain). Risk is *relative* within each city, and the hotspot list is media/advisory-derived, not exhaustive — **absence of a marker doesn't mean safe.** These limits are stated on the map itself. Saying them plainly is part of doing this responsibly.

---

<details>
<summary><b>🔧 For developers</b> (click to expand)</summary>

### Pipeline — config-driven, one command per stage
```bash
floodml run delhi             # multi-event SAR flood label → features → train → predict
floodml train-waterlog delhi  # the urban-waterlogging (hotspot) model
python events/build_events.py # the traceable event database
```
A city is a YAML in `configs/city/` + a hotspot list in `configs/hotspots/`. No copy-pasted code.

### Two models per city (spatial-CV AUC)

| City | River (multi-event SAR) | Street waterlogging (PU) |
|---|---|---|
| Delhi | 0.80 | 0.96 |
| Chandigarh | 0.79 | 0.99 |
| Bengaluru | 0.73 | 0.97 |
| Mumbai | 0.70 | 0.97 |

*River AUCs are honestly lower than single-event numbers — multi-event labels across full metros are a harder, more realistic task. The waterlogging model's top driver is `sink_depth` (local depressions/underpasses) in every city — the real street-ponding mechanism. PU-AUCs are relative ranking, not calibrated probability.*

### How it's built
- **Labels** — multi-event Sentinel-1 change detection (each city stacks several real floods → frequency-weighted label). Street-flood labels come from geocoded documented hotspots (positive-only PU learning).
- **Features** — terrain (elevation, slope, HAND, curvature, local relief, sinks) + drainage backbone (MERIT-Hydro + OSM drains).
- **Models** — per-city XGBoost, validated with **spatial block cross-validation** (the honest metric).
- **Event DB** — `events/` — provenance-rich records from documented hotspots + GDELT news + Global Flood Database, with flood-news frequency over time.
- **Engineering** — installable `floodml` package, **Typer** CLI, **Pydantic** configs, **MLflow** tracking, **GitHub Actions** CI. (Deliberately no Kubernetes/Airflow — overkill for this.)

### Reproduce
```bash
conda env create -f environment.yml && conda activate urban-flood-ml
floodml run delhi   # needs a free Google Earth Engine account (set ee_project in the config)
```

### Roadmap
Rainfall trigger (Open-Meteo / GPM IMERG) → rainfall-conditioned *alerts*; a "why it ponds" panel per hotspot; the India Flood Inventory as an extra label source.

</details>

---

*Built with Sentinel-1, Copernicus DEM, MERIT-Hydro, ESA WorldCover, WorldPop & OpenStreetMap (via Google Earth Engine). Free data, open code, public purpose.*
