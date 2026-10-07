from audio.processing.masking import resolve_masking


def test_masking_ranges_follow_source_sample_rate():
    authority = {
        "source": {
            "role": "guitar",
            "dimension": "harmony",
            "sample_rate": 16000,
            "fundamental_hz": 1000.0,
            "spectral_centroid_hz": 3000.0,
            "mid_energy_ratio": 0.8,
            "low_energy_ratio": 0.1,
            "high_energy_ratio": 0.8,
        },
        "reference": {
            "role": "vocal",
            "dimension": "melody",
            "sample_rate": 16000,
            "mid_energy_ratio": 0.8,
            "low_energy_ratio": 0.1,
            "high_energy_ratio": 0.8,
        },
    }
    result = resolve_masking("source", authority)
    decision = result["decisions"][0]
    assert decision["ranges"]["high"][1] == 8000.0


def test_masking_ranges_fall_back_when_sample_rate_is_invalid():
    authority = {
        "source": {
            "role": "guitar",
            "sample_rate": 0,
            "mid_energy_ratio": 0.8,
        },
        "reference": {
            "role": "vocal",
            "mid_energy_ratio": 0.8,
        },
    }
    result = resolve_masking("source", authority)
    assert result["decisions"][0]["ranges"]["high"][1] == 24000.0
