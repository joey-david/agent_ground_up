from roman import to_roman

def test_roman():
    assert to_roman(1) == 'I'
    assert to_roman(4) == 'IV'
    assert to_roman(9) == 'IX'
    assert to_roman(14) == 'XIV'
    assert to_roman(1987) == 'MCMLXXXVII'
    assert to_roman(3999) == 'MMMCMXCIX'
