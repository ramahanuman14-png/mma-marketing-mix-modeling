import pytest
import numpy as np
import sys
sys.path.insert(0, '.')
from src.features.transformations import geometric_adstock, hill_saturation

def test_zero_decay_returns_original():
    spend = np.array([100.0, 200.0, 0.0])
    result = geometric_adstock(spend, decay=0.0)
    np.testing.assert_array_almost_equal(result, spend)

def test_ec50_returns_half():
    result = hill_saturation(np.array([100.0]), ec50=100.0, slope=1.0)
    assert abs(result[0] - 0.5) < 1e-6

def test_hill_output_between_zero_and_one():
    spend = np.array([0.0, 50.0, 100.0, 500.0])
    result = hill_saturation(spend, ec50=100.0, slope=1.0)
    assert all(0.0 <= r <= 1.0 for r in result)

def test_adstock_length_matches_input():
    spend = np.array([10.0, 20.0, 30.0])
    result = geometric_adstock(spend, decay=0.3)
    assert len(result) == len(spend)
