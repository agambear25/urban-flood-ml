"""Per-city configuration — one YAML per city is how we run N cities without copy-paste."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SARWindows(BaseModel):
    """Date windows for the pre-flood and post-flood Sentinel-1 composites."""

    pre_start: str
    pre_end: str
    post_start: str
    post_end: str
    orbit: str = "DESCENDING"  # ASCENDING | DESCENDING


class CityConfig(BaseModel):
    name: str
    slug: str
    ee_project: str
    bbox: list[float] = Field(..., description="[west, south, east, north] in EPSG:4326")
    scale: int = 10  # metres
    sar: SARWindows
    threshold_db: float = -3.0   # backscatter drop that counts as flood
    perm_water_db: float = -18.0  # pixels already this dark before = permanent water
    coastal: bool = False  # mask the sea for coastal cities (Mumbai)
    notes: str = ""

    @property
    def west(self) -> float:
        return self.bbox[0]

    @property
    def south(self) -> float:
        return self.bbox[1]

    @property
    def east(self) -> float:
        return self.bbox[2]

    @property
    def north(self) -> float:
        return self.bbox[3]

    @property
    def lat_mid(self) -> float:
        return (self.south + self.north) / 2


def load_city(name: str, configs_dir: str | Path = "configs/city") -> CityConfig:
    """Load a city config by slug (e.g. 'delhi') from configs/city/<name>.yaml."""
    path = Path(configs_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for city '{name}' at {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return CityConfig(**data)


def list_cities(configs_dir: str | Path = "configs/city") -> list[str]:
    return sorted(p.stem for p in Path(configs_dir).glob("*.yaml"))
