import numpy as np
from ppg_demo.analysis import Subject, analyze_loaded_subject, estimate_hr_from_ppg, validate_subject
from ppg_demo.agent import template_explanation


def synthetic_record(seconds=64, bpm=90.0, *, ref_offset=0.0):
    t = np.arange(seconds * 64) / 64
    # Synthetic periodic signal, explicitly NOT actual patient data.
    rng = np.random.default_rng(3)
    bvp = np.sin(2 * np.pi * bpm / 60 * t) + 0.1 * rng.standard_normal(len(t))
    a = np.arange(seconds * 32) / 32
    acc = np.stack((0.01 * np.sin(2 * np.pi * .8 * a),
                    0.02 * np.sin(2 * np.pi * .4 * a),
                    0.005 * np.ones_like(a)), axis=1)
    label_count = (seconds - 8) // 2 + 1
    reference = np.full(label_count, bpm + ref_offset)
    return Subject("S1", bvp, acc, reference)


def test_estimate_known_frequency():
    t = np.arange(8 * 64) / 64
    bpm, share, frequency = estimate_hr_from_ppg(np.sin(2 * np.pi * 1.5 * t))
    assert abs(bpm - 90) < 2
    assert 0 < share <= 1
    assert abs(frequency - 1.5) < .03


def test_flat_signal_reports_unknown():
    bpm, _, _ = estimate_hr_from_ppg(np.ones(8 * 64))
    assert np.isnan(bpm)


def test_8s_2s_alignment_and_finite_metrics():
    res = analyze_loaded_subject(synthetic_record(), 0, 60, return_trace=True)
    assert res["complete_windows"] == 27
    assert res["windows"][0]["start_seconds"] == 0
    assert res["windows"][-1]["start_seconds"] == 52
    assert res["filtered_welch"]["mae_bpm"] < 3
    assert res["invalid_reference_windows"] == 0
    assert "Uncertainty" in template_explanation(res)


def test_reference_labels_do_not_affect_estimations():
    res1 = analyze_loaded_subject(synthetic_record(ref_offset=0), 0, 60, return_trace=True)
    res2 = analyze_loaded_subject(synthetic_record(ref_offset=25), 0, 60, return_trace=True)
    assert [w["filtered_bpm"] for w in res1["windows"]] == [w["filtered_bpm"] for w in res2["windows"]]
    assert res1["filtered_welch"]["mae_bpm"] != res2["filtered_welch"]["mae_bpm"]


def test_illegal_subject_rejected():
    for invalid in ["S16", "../S1", "S1/../../foo", "whatever"]:
        try:
            validate_subject(invalid)
        except ValueError:
            continue
        assert False, f"Must reject {invalid}"
