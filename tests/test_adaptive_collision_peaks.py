import numpy as np

from audio.processing.dynamic_masking import _spectral_collision_peaks


def _tone(sr, hz, amplitude=1.0, n=4096):
    t = np.arange(n, dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * hz * t))[:, None]


def test_adaptive_peak_detection_finds_strong_tonal_collision():
    sr = 8000
    source = _tone(sr, 1000.0)
    reference = _tone(sr, 1000.0)
    peaks = _spectral_collision_peaks(source, reference, sr, 700.0, 1300.0)
    assert any(abs(p - 1000.0) < 50.0 for p in peaks)


def test_adaptive_peak_detection_rejects_weak_broadband_tail():
    sr = 8000
    rng = np.random.default_rng(7)
    source = _tone(sr, 1000.0)
    reference = (0.015 * rng.standard_normal((4096, 1))).astype(np.float32)
    peaks = _spectral_collision_peaks(source, reference, sr, 700.0, 1300.0)
    assert peaks == ()


def test_adaptive_peak_detection_preserves_critical_band_nearby_tones():
    sr = 8000
    source = _tone(sr, 1000.0)
    reference = _tone(sr, 1030.0, amplitude=0.2)
    peaks = _spectral_collision_peaks(source, reference, sr, 900.0, 1150.0)
    assert any(abs(p - 1000.0) < 60.0 for p in peaks)
