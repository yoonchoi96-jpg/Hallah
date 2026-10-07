import numpy as np


def test_spectral_collision_uses_critical_band_overlap_for_nearby_frequencies():
    from audio.processing.dynamic_masking import _spectral_collision_peaks

    sr = 16000
    n = 8192
    time = np.arange(n, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    reference = np.sin(2 * np.pi * 1030 * time).astype(np.float32)
    peaks = _spectral_collision_peaks(
        source[:, None], reference[:, None], sr, 700.0, 1400.0, max_peaks=1
    )
    assert len(peaks) == 1
    assert 950.0 <= float(peaks[0]) <= 1080.0


def test_critical_band_collision_does_not_require_equal_amplitude():
    from audio.processing.dynamic_masking import _spectral_collision_peaks

    sr = 16000
    n = 8192
    time = np.arange(n, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    reference = (0.2 * np.sin(2 * np.pi * 1010 * time)).astype(np.float32)
    peaks = _spectral_collision_peaks(
        source[:, None], reference[:, None], sr, 700.0, 1400.0, max_peaks=1
    )
    assert len(peaks) == 1
    assert 950.0 <= float(peaks[0]) <= 1080.0
