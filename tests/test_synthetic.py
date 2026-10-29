import numpy as np
from ppg_demo.synthetic import synthetic_record
from ppg_demo.analysis import analyze_loaded_subject


def test_synthetic_fixture_is_consistent_and_motion_perturbs_estimation():
    record = synthetic_record(seconds=1100, movement_from_seconds=600)
    early = analyze_loaded_subject(record, 30, 120)
    late = analyze_loaded_subject(record, 900, 120)
    assert early['complete_windows'] == late['complete_windows'] == 57
    assert early['reference_usage'].startswith('comparison and evaluation only')
    assert late['motion_frequency_overlap_percent'] is not None
    # The fake moving region has larger simulated accelerometer variance.
    et = analyze_loaded_subject(record, 30, 120, return_trace=True)
    lt = analyze_loaded_subject(record, 900, 120, return_trace=True)
    r1 = np.nanmedian([x['motion_rms_relative_units'] for x in et['windows']])
    r2 = np.nanmedian([x['motion_rms_relative_units'] for x in lt['windows']])
    assert r2 > 10*r1
