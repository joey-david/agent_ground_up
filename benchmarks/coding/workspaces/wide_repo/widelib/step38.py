"""Stage 38 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step38']
    record['value'] = record['value'] + 1
    return record
