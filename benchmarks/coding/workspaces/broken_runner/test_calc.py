import pytest
from calc import divide

def test_divide():
    assert divide(6, 3) == 2

def test_divide_by_zero():
    assert divide(1, 0) is None
