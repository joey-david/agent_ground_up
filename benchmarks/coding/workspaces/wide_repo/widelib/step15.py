"""Stage 15 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step15']
    record['value'] = record['value'] + 1
    return record
