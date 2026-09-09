"""Stage 31 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step31']
    record['value'] = record['value'] + 1
    return record
