"""Stage 21 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step21']
    record['value'] = record['value'] + 1
    return record
