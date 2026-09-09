import pytest
from stats import mean, median

def test_mean_empty():
    assert mean([]) == 0

def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5

def test_median_odd():
    assert median([3, 1, 2]) == 2
