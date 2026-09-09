"""Stage 2 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step02']
    record['value'] = record['value'] + 1
    return record
