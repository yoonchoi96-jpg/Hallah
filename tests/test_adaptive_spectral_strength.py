import numpy as np

from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking


def _run(source, reference):
    sr = 8000
    meta = {
        "role": "vocal",
        "bpm": 60.0,
        "onset_beats": (0.25,),
        "note_durations_beats": (0.5,),
        "spectral_centroid_hz": 1000.0,
        "_source_id": "reference",
    }
    return apply_spectral_curve_dynamic_masking(
        source.astype(np.float32)[:, None],
        sr,
        {"role": "guitar", "bpm": 60.0},
        meta,
        0.30,
        bands=("mid",),
        ranges={"mid": (500.0, 1800.0)},
        reference_data=reference.astype(np.float32)[:, None],
    )


def test_very_weak_reference_band_has_negligible_strength():
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000.0 * t)
    reference = 0.03 * source
    _, meta = _run(source, reference)
    assert float(meta["tracking_strength_max"]) < 0.12


def test_strong_tonal_overlap_is_stronger_than_broadband_overlap():
    sr = 8000
    rng = np.random.default_rng(7)
    t = np.arange(sr, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000.0 * t)
    tonal_reference = source.copy()
    broadband_reference = rng.normal(0.0, 1.0, sr).astype(np.float32)
    broadband_reference *= 0.35 / max(float(np.sqrt(np.mean(broadband_reference ** 2))), 1e-9)

    _, tonal_meta = _run(source, tonal_reference)
    _, broadband_meta = _run(source, broadband_reference)

    assert float(tonal_meta["tracking_strength_max"]) > float(broadband_meta["tracking_strength_max"])


def test_center_and_side_paths_remain_separate_after_strength_adaptation():
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 1000.0 * t)
    center = np.column_stack([tone, tone]).astype(np.float32)
    side = np.column_stack([tone, -tone]).astype(np.float32)

    reference = center.copy()
    center_out, center_meta = apply_spectral_curve_dynamic_masking(
        center, sr,
        {"role": "guitar", "bpm": 60.0},
        {
            "role": "vocal", "bpm": 60.0,
            "onset_beats": (0.25,), "note_durations_beats": (0.5,),
            "spectral_centroid_hz": 1000.0, "_source_id": "vocal",
        },
        0.30, bands=("mid",), ranges={"mid": (700.0, 1300.0)},
        reference_data=reference,
    )
    side_out, side_meta = apply_spectral_curve_dynamic_masking(
        side, sr,
        {"role": "guitar", "bpm": 60.0},
        {
            "role": "vocal", "bpm": 60.0,
            "onset_beats": (0.25,), "note_durations_beats": (0.5,),
            "spectral_centroid_hz": 1000.0, "_source_id": "vocal",
        },
        0.30, bands=("mid",), ranges={"mid": (700.0, 1300.0)},
        reference_data=reference,
    )

    center_mid = np.mean(center_out[2200:3800], axis=1)
    original_mid = np.mean(center[2200:3800], axis=1)
    side_side = (side_out[2200:3800, 0] - side_out[2200:3800, 1]) * 0.5
    original_side = (side[2200:3800, 0] - side[2200:3800, 1]) * 0.5

    center_ratio = float(np.sqrt(np.mean(center_mid ** 2))) / float(np.sqrt(np.mean(original_mid ** 2)))
    side_ratio = float(np.sqrt(np.mean(side_side ** 2))) / float(np.sqrt(np.mean(original_side ** 2)))

    assert center_meta["stereo_mode"] == "mid_side"
    assert side_meta["stereo_mode"] == "mid_side"
    assert center_ratio < 0.85
    assert side_ratio > 0.95
