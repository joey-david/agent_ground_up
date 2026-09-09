from cart import Cart

def test_total():
    c = Cart(); c.add('a', 2.0, 3)
    assert c.total() == 6.0

def test_names():
    c = Cart(); c.add('a', 1.0); c.add('b', 1.0)
    assert c.names() == ['a', 'b']
