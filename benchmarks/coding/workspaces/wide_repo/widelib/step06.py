"""Stage 6 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step06']
    record['value'] = record['value'] + 1
    return record
