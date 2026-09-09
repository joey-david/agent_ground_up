from query import where

ROWS = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}, {'a': 1, 'b': 'y'}]

def test_where():
    assert where(ROWS, a=1) == [{'a': 1, 'b': 'x'}, {'a': 1, 'b': 'y'}]
    assert where(ROWS, a=1, b='y') == [{'a': 1, 'b': 'y'}]
    assert where(ROWS) == ROWS
    assert where(ROWS, a=9) == []
