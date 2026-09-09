"""Stage 27 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step27']
    record['value'] = record['value'] * 2
    return record
