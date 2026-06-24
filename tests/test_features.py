"""Feature-contract tests — offline, no heavy geospatial imports."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from floodml import FEATURES


def test_features_unique_and_complete():
    assert len(FEATURES) == len(set(FEATURES)), "duplicate feature names"
    for required in ["elevation", "slope", "hand", "dist_river", "builtup",
                     "dist_drain", "drain_density", "upstream_area", "sink_depth"]:
        assert required in FEATURES


def test_distance_to_drain_monotonic():
    # a single drain cell at the centre -> distance grows away from it
    grid = np.ones((5, 5), dtype=int)
    grid[2, 2] = 0
    dist = distance_transform_edt(grid)
    assert dist[2, 2] == 0
    assert dist[0, 0] > dist[1, 1] > dist[2, 2]
