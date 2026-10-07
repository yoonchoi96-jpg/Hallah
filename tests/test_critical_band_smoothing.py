import numpy as np

from audio.processing.dynamic_masking import _critical_band_smoothing


def test_critical_band_smoothing_preserves_shape_and_dtype():
    freqs = np.linspace(20.0, 8000.0, 1025, dtype=np.float32)
    spectrum = np.zeros_like(freqs)
    spectrum[400] = 1.0
    result = _critical_band_smoothing(spectrum, freqs)
    assert result.shape == spectrum.shape
    assert result.dtype == np.float32
    assert float(result.max()) > 0.0


def test_critical_band_smoothing_spreads_energy_locally_not_globally():
    freqs = np.linspace(20.0, 8000.0, 1025, dtype=np.float32)
    spectrum = np.zeros_like(freqs)
    spectrum[400] = 1.0
    result = _critical_band_smoothing(spectrum, freqs)
    peak = int(np.argmax(result))
    assert abs(peak - 400) <= 2
    assert result[100] < result[400] * 0.01
    assert result[900] < result[400] * 0.01


def test_critical_band_smoothing_reacts_to_nearby_bark_energy():
    freqs = np.linspace(20.0, 8000.0, 1025, dtype=np.float32)
    spectrum = np.zeros_like(freqs)
    spectrum[400] = 1.0
    spectrum[404] = 0.5
    result = _critical_band_smoothing(spectrum, freqs)
    assert float(result[404]) > 0.0
    assert float(result[404]) > float(result[700])
