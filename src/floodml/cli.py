"""floodml CLI — one command per pipeline stage, driven by per-city YAML configs."""
from __future__ import annotations

import json
from pathlib import Path

import typer

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
    typer.echo(f"[{city}] flood mask -> {p}  (S1 images: pre={counts['pre']} post={counts['post']})")


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
