import numpy as np

from audio.processing.dynamic_masking import _align_reference_to_source


def test_reference_alignment_estimates_delayed_reference():
    sr = 1000
    source = np.zeros((3000, 1), dtype=np.float32)
    reference = np.zeros((3000, 1), dtype=np.float32)
    source[1000:1010, 0] = 1.0
    reference[1250:1260, 0] = 1.0

    aligned, meta, info = _align_reference_to_source(
        source,
        reference,
        sr,
        {"bpm": 60.0},
        {"bpm": 60.0},
    )

    assert info["lag_samples"] < 0
    assert abs(float(info["lag_seconds"]) + 0.25) <= 0.02
    assert float(np.argmax(np.abs(aligned[:, 0]))) == 1000.0
    assert meta["bpm"] == 60.0


def test_reference_alignment_maps_bpm_timeline_before_lag():
    sr = 1000
    source = np.zeros((3000, 1), dtype=np.float32)
    reference = np.zeros((3000, 1), dtype=np.float32)
    source[1000:1010, 0] = 1.0
    # At 120 BPM this is beat 2.0; the source is 60 BPM, so the same
    # musical event belongs at source second 2.0 after BPM mapping.
    reference[500:510, 0] = 1.0

    aligned, meta, info = _align_reference_to_source(
        source,
        reference,
        sr,
        {"bpm": 60.0},
        {"bpm": 120.0},
    )

    assert info["bpm_ratio"] == 2.0
    assert len(aligned) == 6000
    assert meta["bpm"] == 60.0
    assert abs(int(np.argmax(np.abs(aligned[:, 0]))) - 1000) <= 10


def test_reference_alignment_handles_short_reference_and_mono():
    sr = 1000
    source = np.zeros((2000, 1), dtype=np.float32)
    reference = np.zeros((500, 1), dtype=np.float32)
    reference[100:110, 0] = 1.0

    aligned, _, _ = _align_reference_to_source(
        source,
        reference,
        sr,
        {},
        {},
    )

    assert aligned.shape == reference.shape
    assert aligned.ndim == 2


def test_reference_alignment_preserves_stereo_channels():
    sr = 1000
    source = np.zeros((1000, 2), dtype=np.float32)
    reference = np.zeros((1000, 2), dtype=np.float32)
    reference[400:410, 1] = 1.0

    aligned, _, _ = _align_reference_to_source(
        source,
        reference,
        sr,
        {"bpm": 60.0},
        {"bpm": 60.0},
    )

    assert aligned.shape == reference.shape
    assert float(np.max(np.abs(aligned[:, 1]))) > 0.9
    assert float(np.max(np.abs(aligned[:, 0]))) == 0.0
