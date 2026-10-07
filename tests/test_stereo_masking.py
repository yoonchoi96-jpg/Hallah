import numpy as np

from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking


def _rms(data):
    return float(np.sqrt(np.mean(np.square(data))))


def _render(source, reference):
    sr = 8000
    n = len(source)
    out, meta = apply_spectral_curve_dynamic_masking(
        source.astype(np.float32),
        sr,
        {"role": "guitar", "bpm": 60.0},
        {
            "role": "vocal",
            "bpm": 60.0,
            "onset_beats": (0.25,),
            "note_durations_beats": (0.5,),
            "spectral_centroid_hz": 1000.0,
            "_source_id": "vocal",
        },
        0.30,
        bands=("mid",),
        ranges={"mid": (700.0, 1300.0)},
        reference_data=reference.astype(np.float32),
    )
    assert meta["applied"] is True
    assert meta["stereo_mode"] == "mid_side"
    return out, meta


def test_center_reference_ducks_center_source_more_than_side_source():
    sr = 8000
    t = np.arange(8000, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 1000 * t)
    center = np.column_stack([tone, tone])
    side = np.column_stack([tone, -tone])
    reference = center.copy()

    center_out, center_meta = _render(center, reference)
    side_out, side_meta = _render(side, reference)

    center_mid = np.mean(center_out, axis=1)
    side_mid = np.mean(side_out, axis=1)
    assert _rms(center_mid[2200:3800]) < _rms(side_mid[2200:3800]) * 0.85
    assert center_meta["tracking_strength_max"] > 0.5
    assert side_meta["tracking_strength_max"] < center_meta["tracking_strength_max"]


def test_side_reference_ducks_side_source():
    sr = 8000
    t = np.arange(8000, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 1000 * t)
    side = np.column_stack([tone, -tone])

    out, meta = _render(side, side)
    out_side = (out[:, 0] - out[:, 1]) * 0.5
    in_side = (side[:, 0] - side[:, 1]) * 0.5

    assert _rms(out_side[2200:3800]) < _rms(in_side[2200:3800]) * 0.85
    assert meta["tracked_side_strengths"]


def test_mono_reference_path_remains_mid_only():
    sr = 8000
    t = np.arange(8000, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    source = tone[:, None]

    out, meta = apply_spectral_curve_dynamic_masking(
        source,
        sr,
        {"role": "guitar", "bpm": 60.0},
        {
            "role": "vocal",
            "bpm": 60.0,
            "onset_beats": (0.25,),
            "note_durations_beats": (0.5,),
            "spectral_centroid_hz": 1000.0,
            "_source_id": "vocal",
        },
        0.30,
        bands=("mid",),
        ranges={"mid": (700.0, 1300.0)},
        reference_data=tone[:, None],
    )

    assert meta["stereo_mode"] == "mid_mono"
    assert not meta["tracked_side_centers_hz"]
    assert _rms(out[2200:3800]) < _rms(source[2200:3800])
