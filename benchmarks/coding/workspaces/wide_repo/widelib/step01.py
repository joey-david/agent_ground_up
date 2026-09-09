"""Stage 1 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step01']
    record['value'] = record['value'] + 1
    return record
