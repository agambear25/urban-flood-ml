"""floodml CLI — one command per pipeline stage, driven by per-city YAML configs."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import eval_crosscity as evalx
from . import explain as expl
from . import export_web as expw
from . import features as feat
from . import gee
from . import predict as prd
from . import train as trn
from .config import list_cities, load_city

app = typer.Typer(help="floodml — multi-city urban flood susceptibility pipeline", no_args_is_help=True)

CONFIGS = "configs/city"
DATA = Path("data")
MODELS = Path("models")
RESULTS = Path("results")


def _data_dir(slug: str) -> Path:
    d = DATA / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command("list-cities")
def list_cities_cmd():
    """List the configured cities."""
    for c in list_cities(CONFIGS):
        typer.echo(c)


@app.command("flood-mask")
def flood_mask(city: str):
    """Detect the flood mask from Sentinel-1 SAR change detection (Earth Engine)."""
    cfg = load_city(city, CONFIGS)
    gee.init_ee(cfg.ee_project)
    p, counts = gee.build_flood_mask(cfg, _data_dir(city) / "flood_mask.tif")
    typer.echo(f"[{city}] flood mask ({len(counts)} events) -> {p}")
    for name, c in counts.items():
        typer.echo(f"    {name}: S1 images pre={c['pre']} post={c['post']}")


@app.command()
def fetch(city: str, force: bool = False):
    """Download the static layers (DEM, built-up, MERIT-Hydro)."""
    cfg = load_city(city, CONFIGS)
    gee.init_ee(cfg.ee_project)
    paths = gee.fetch_static_layers(cfg, _data_dir(city), force=force)
    typer.echo(f"[{city}] fetched: {', '.join(paths)}")


@app.command()
def features(city: str):
    """Build the aligned terrain + drainage feature stack."""
    cfg = load_city(city, CONFIGS)
    p = feat.build_features(cfg, _data_dir(city))
    typer.echo(f"[{city}] feature stack -> {p}")


@app.command()
def train(city: str):
    """Train the per-city XGBoost model (random + spatial CV, MLflow-logged)."""
    cfg = load_city(city, CONFIGS)
    res = trn.train_city(cfg, _data_dir(city), MODELS)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{city}_metrics.json").write_text(json.dumps(res, indent=2))
    typer.echo(f"[{city}] spatial-CV AUC {res['spatial_cv_auc']:.3f} "
               f"(random {res['random_cv_auc']:.3f}) -> {res['model_path']}")


@app.command()
def predict(city: str):
    """Predict the susceptibility surface for all pixels."""
    p = prd.predict_city(_data_dir(city), MODELS / f"{city}_model.json")
    typer.echo(f"[{city}] susceptibility -> {p}")


@app.command("train-waterlog")
def train_waterlog_cmd(city: str):
    """Train the urban-waterlogging (documented-hotspot) model — positive-only PU learning."""
    from . import waterlog
    cfg = load_city(city, CONFIGS)
    res = waterlog.train_waterlog(cfg, _data_dir(city), MODELS)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{city}_waterlog_metrics.json").write_text(json.dumps(res, indent=2))
    typer.echo(f"[{city}] waterlog PU spatial-AUC {res['spatial_cv_auc_pu']:.3f} "
               f"({res['n_positives']} hotspot px, {res['n_negatives']} pseudo-neg)")


@app.command("eval-crosscity")
def eval_crosscity_cmd():
    """Leave-one-city-out evaluation: train on N-1 cities, test on the unseen one.

    Reports cross-city ROC-AUC vs within-city spatial CV (the generalization gap),
    PR-AUC against each city's true flood prevalence, and calibration (Brier, raw vs
    isotonic). Reads only on-disk data; writes results/ + docs/eval/ figures.
    """
    res = evalx.crosscity_eval()
    for r in res["results"]:
        typer.echo(
            f"[{r['held_out_city']:>10}] unseen ROC-AUC {r['crosscity_roc_auc']:.3f} "
            f"(within-city {r['within_city_spatial_roc_auc']:.3f}, "
            f"gap {r['generalization_gap']:+.3f}) | PR-AUC {r['crosscity_pr_auc']:.3f} "
            f"vs base {r['pr_auc_baseline']:.3f} | Brier {r['brier_raw']:.3f}->{r['brier_calibrated']:.3f}")


@app.command("export-web")
def export_web_cmd(city: str):
    """Export precomputed web data (risk overlay + hotspots + meta) for the map under web/<city>/."""
    res = expw.export_city(city)
    typer.echo(f"[{city}] web export → web/{city}/  ({res['n_hotspots']} hotspots, events {res['events']})")


@app.command()
def explain(city: str, top: int = 8):
    """Explain WHY the model flags each documented waterlogging hotspot in a city.

    Uses exact per-prediction feature contributions (tree SHAP) and prints them in plain
    English; writes results/<city>_why.json + docs/eval/why_<city>.png.
    """
    res = expl.explain_city(city, top=top)
    typer.echo(f"[{city}] {res['n_hotspots']} hotspots ({res['model']})")
    for h in res["hotspots"][:top]:
        reasons = "; ".join(f"{'↑' if f['direction']=='raises' else '↓'} {f['plain']}"
                            for f in h["why"])
        typer.echo(f"  {h['name']}  (risk {h['relative_risk']:.2f}): {reasons}")


@app.command()
def run(city: str, skip_flood_mask: bool = False):
    """Run the full pipeline for one city: flood-mask -> fetch -> features -> train -> predict."""
    if not skip_flood_mask:
        flood_mask(city)
    fetch(city)
    features(city)
    train(city)
    predict(city)


if __name__ == "__main__":
    app()
