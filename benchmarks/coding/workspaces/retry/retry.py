def retry(fn, attempts=3):
    last = None
    for _ in range(attempts - 1):
        try:
            return fn()
        except Exception as e:
            last = e
    raise last
