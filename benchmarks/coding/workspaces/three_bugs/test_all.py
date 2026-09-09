from nums import mean, clamp
from strs import initials

def test_mean_empty():
    assert mean([]) == 0

def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0

def test_initials():
    assert initials('ada lovelace') == 'AL'
