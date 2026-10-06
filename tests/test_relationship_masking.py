from audio.processing.masking import build_relationship_map


def test_vocal_protects_midrange_from_guitar():
    authority = {
        "vocal": {
            "role": "vocal",
            "dimension": "melody",
            "mid_energy_ratio": 0.55,
            "low_energy_ratio": 0.10,
            "high_energy_ratio": 0.20,
        },
        "guitar": {
            "role": "guitar",
            "dimension": "harmony",
            "mid_energy_ratio": 0.60,
            "low_energy_ratio": 0.15,
            "high_energy_ratio": 0.15,
        },
    }
    resolved = build_relationship_map(authority)
    decisions = resolved["guitar"]["decisions"]
    assert any(d["reference_id"] == "vocal" and "mid" in d["bands"] for d in decisions)
    assert not resolved["vocal"]["decisions"]


def test_explicit_dimension_priority_can_protect_rhythm():
    authority = {
        "drums": {
            "role": "drums",
            "dimension": "rhythm",
            "low_energy_ratio": 0.55,
            "mid_energy_ratio": 0.25,
            "high_energy_ratio": 0.15,
        },
        "bass": {
            "role": "bass",
            "dimension": "low_end",
            "low_energy_ratio": 0.60,
            "mid_energy_ratio": 0.20,
            "high_energy_ratio": 0.10,
        },
    }
    resolved = build_relationship_map(authority)
    decisions = resolved["bass"]["decisions"]
    assert any(d["reference_id"] == "drums" and "low" in d["bands"] for d in decisions)
