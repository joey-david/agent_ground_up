def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[:size], values[size:]]
