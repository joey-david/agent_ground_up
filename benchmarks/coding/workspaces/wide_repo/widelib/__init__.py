from importlib import import_module

STEPS = [f'step{i:02d}' for i in range(40)]


def run(record):
    for name in STEPS:
        record = import_module(f'widelib.{name}').apply(record)
    return record
