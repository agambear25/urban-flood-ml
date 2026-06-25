"""Config tests — real checks that run offline (no Earth Engine / network)."""
import pytest

from floodml.config import CityConfig, list_cities, load_city

CONFIGS = "configs/city"


def test_all_cities_load_and_validate():
    cities = list_cities(CONFIGS)
    assert {"delhi", "chandigarh", "mumbai", "bengaluru"} <= set(cities)
    for c in cities:
        cfg = load_city(c, CONFIGS)
        assert isinstance(cfg, CityConfig)
        assert len(cfg.bbox) == 4
        assert cfg.west < cfg.east, f"{c}: west must be < east"
        assert cfg.south < cfg.north, f"{c}: south must be < north"
        assert cfg.slug == c
        assert len(cfg.events) >= 1, f"{c}: needs at least one flood event"
        for e in cfg.events:
            assert e.pre_start < e.post_start, f"{c}: pre window must precede post"


def test_missing_city_raises():
    with pytest.raises(FileNotFoundError):
        load_city("atlantis", CONFIGS)
