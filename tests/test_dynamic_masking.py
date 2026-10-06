from audio.processing.masking import build_relationship_map


def test_fundamental_and_centroid_change_masking_ranges():
    authority = {
        "bass": {
            "role": "bass",
            "dimension": "low_end",
            "fundamental_hz": 55.0,
            "spectral_centroid_hz": 420.0,
            "low_energy_ratio": 0.60,
            "mid_energy_ratio": 0.25,
            "high_energy_ratio": 0.05,
        },
        "kick": {
            "role": "kick",
            "dimension": "rhythm",
            "fundamental_hz": 48.0,
            "spectral_centroid_hz": 1800.0,
            "low_energy_ratio": 0.65,
            "mid_energy_ratio": 0.25,
            "high_energy_ratio": 0.10,
        },
    }
    resolved = build_relationship_map(authority)
    decision = resolved["bass"]["decisions"][0]
    assert decision["reference_id"] == "kick"
    assert decision["ranges"]["low"][0] > 20.0
    assert decision["ranges"]["low"][1] <= 220.0
