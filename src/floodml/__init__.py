"""floodml — config-driven, multi-city urban flood susceptibility pipeline."""

__version__ = "0.1.0"

FEATURES = [
    "elevation",
    "slope",
    "curvature",
    "local_relief",
    "hand",
    "dist_river",
    "builtup",
    # --- drainage backbone (added in Phase 3) ---
    "dist_drain",
    "drain_density",
    "upstream_area",
    "sink_depth",
]
