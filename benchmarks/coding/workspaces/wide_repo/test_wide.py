from widelib import run


def test_pipeline_increments_once_per_stage():
    out = run({'value': 0})
    assert out['value'] == 40, out['value']
    assert len(out['trace']) == 40
