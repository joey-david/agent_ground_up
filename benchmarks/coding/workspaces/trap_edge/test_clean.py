from text import clean

def test_basic():
    assert clean('  Hello  World ') == 'hello world'
