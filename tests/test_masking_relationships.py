from audio.processing.masking import resolve_masking


def test_authority_direction_means_target_adapts_to_source():
    analysis = {
        "guitar": {"role": "guitar", "dimension": "harmony", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
        "vocal": {"role": "vocal", "dimension": "harmony", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
    }
    relationships = ({"source_id": "guitar", "target_id": "vocal", "type": "authority", "dimension": "harmony", "confidence": 1.0},)
    vocal_result = resolve_masking("vocal", analysis, relationships)
    guitar_result = resolve_masking("guitar", analysis, relationships)
    assert any(d["reference_id"] == "guitar" for d in vocal_result["decisions"])
    assert not any(d["reference_id"] == "vocal" for d in guitar_result["decisions"])


def test_reverse_authority_is_not_treated_as_authority():
    analysis = {
        "guitar": {"role": "guitar", "dimension": "harmony", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
        "vocal": {"role": "vocal", "dimension": "harmony", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
    }
    relationships = ({"source_id": "guitar", "target_id": "vocal", "type": "authority", "dimension": "harmony", "confidence": 1.0},)
    result = resolve_masking("guitar", analysis, relationships)
    assert not any(d["reference_id"] == "vocal" and "explicit authority" in d["reason"] for d in result["decisions"])


def test_explicit_conflict_relationship_can_create_masking_decision():
    analysis = {
        "bass": {"role": "bass", "dimension": "low_end", "low_energy_ratio": 0.8, "mid_energy_ratio": 0.1, "high_energy_ratio": 0.1},
        "kick": {"role": "kick", "dimension": "rhythm", "low_energy_ratio": 0.8, "mid_energy_ratio": 0.1, "high_energy_ratio": 0.1},
    }
    relationships = ({"source_id": "bass", "target_id": "kick", "type": "conflict", "dimension": "low_end", "confidence": 1.0},)
    result = resolve_masking("bass", analysis, relationships)
    assert any(d["reference_id"] == "kick" for d in result["decisions"])


def test_dependency_can_break_a_close_priority_tie_when_dimension_matches():
    analysis = {
        "pad": {"role": "pad", "dimension": "texture", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
        "guitar": {"role": "guitar", "dimension": "texture", "mid_energy_ratio": 0.7, "low_energy_ratio": 0.1, "high_energy_ratio": 0.1},
    }
    relationships = ({"source_id": "pad", "target_id": "guitar", "type": "dependency", "dimension": "texture", "confidence": 1.0},)
    result = resolve_masking("pad", analysis, relationships)
    assert any(d["reference_id"] == "guitar" for d in result["decisions"])
