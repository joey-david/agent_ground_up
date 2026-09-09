"""Stage 26 of the pipeline."""


def apply(record):
    record = dict(record)
    record['trace'] = record.get('trace', []) + ['step26']
    record['value'] = record['value'] + 1
    return record
