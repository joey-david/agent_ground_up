from cart import Cart

def test_total_uses_quantity():
    c = Cart()
    c.add('apple', 2.0, qty=3)
    assert c.total() == 6.0

def test_empty_cart():
    assert Cart().total() == 0
