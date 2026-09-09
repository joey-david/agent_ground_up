"""Stage 28 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step28']
    record['value'] = record['value'] + 1
    return record
