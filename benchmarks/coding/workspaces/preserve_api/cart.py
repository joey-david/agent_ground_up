class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty=1):
        self.items.append((name, price, qty))

    def total(self):
        return sum(price * qty for _, price, qty in self.items)

    def names(self):
        return [name for name, _, _ in self.items]
