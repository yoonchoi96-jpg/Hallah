from audio.processing.masking import resolve_masking


def test_explicit_authority_relationship_overrides_role_priority():
    analysis = {
        "guitar": {
            "role": "guitar",
            "dimension": "harmony",
            "mid_energy_ratio": 0.7,
            "low_energy_ratio": 0.1,
            "high_energy_ratio": 0.1,
        },
        "vocal": {
            "role": "vocal",
            "dimension": "melody",
            "mid_energy_ratio": 0.7,
            "low_energy_ratio": 0.1,
            "high_energy_ratio": 0.1,
        },
    }
    relationships = (
        {
            "source_id": "guitar",
            "target_id": "vocal",
            "type": "authority",
            "dimension": "harmony",
            "confidence": 1.0,
        },
    )
    result = resolve_masking("guitar", analysis, relationships)
    assert any(d["reference_id"] == "vocal" for d in result["decisions"])


def test_explicit_conflict_relationship_can_create_masking_decision():
    analysis = {
        "bass": {
            "role": "bass",
            "dimension": "low_end",
            "low_energy_ratio": 0.8,
            "mid_energy_ratio": 0.1,
            "high_energy_ratio": 0.1,
        },
        "kick": {
            "role": "kick",
            "dimension": "rhythm",
            "low_energy_ratio": 0.8,
            "mid_energy_ratio": 0.1,
            "high_energy_ratio": 0.1,
        },
    }
    relationships = (
        {
            "source_id": "bass",
            "target_id": "kick",
            "type": "conflict",
            "dimension": "low_end",
            "confidence": 1.0,
        },
    )
    result = resolve_masking("bass", analysis, relationships)
    assert any(d["reference_id"] == "kick" for d in result["decisions"])
