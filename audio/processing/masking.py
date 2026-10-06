"""Deterministic musical-role-aware masking decisions for V0."""
from __future__ import annotations

ROLE_PRIORITY = {
    "vocal": 100,
    "lead": 95,
    "melody": 95,
    "drums": 90,
    "kick": 92,
    "bass": 88,
    "harmony": 70,
    "guitar": 68,
    "piano": 65,
    "pad": 55,
    "texture": 45,
}

DIMENSION_PRIORITY = {
    "melody": 100,
    "rhythm": 90,
    "low_end": 88,
    "harmony": 70,
    "texture": 50,
    "arrangement": 60,
}

def _priority(meta: dict[str, object]) -> float:
    role = str(meta.get("role", "")).lower()
    dimension = str(meta.get("dimension", "")).lower()
    return max(ROLE_PRIORITY.get(role, 40), DIMENSION_PRIORITY.get(dimension, 40))

def resolve_masking(
    source_id: str,
    authority_analysis: dict[str, dict[str, object]],
    relationships: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Return explainable, overridable spectral protection for one source."""
    source = authority_analysis.get(source_id, {})
    source_priority = _priority(source)
    decisions: list[dict[str, object]] = []

    for ref_id, ref in authority_analysis.items():
        if ref_id == source_id:
            continue
        ref_priority = _priority(ref)
        shared: list[str] = []
        if (source.get("low_energy_ratio", 0) or 0) > 0.10 and (ref.get("low_energy_ratio", 0) or 0) > 0.10:
            shared.append("low")
        if (source.get("mid_energy_ratio", 0) or 0) > 0.10 and (ref.get("mid_energy_ratio", 0) or 0) > 0.10:
            shared.append("mid")
        if (source.get("high_energy_ratio", 0) or 0) > 0.10 and (ref.get("high_energy_ratio", 0) or 0) > 0.10:
            shared.append("high")
        if not shared or source_priority >= ref_priority:
            continue
        decisions.append({
            "reference_id": ref_id,
            "bands": tuple(shared),
            "priority": ref_priority,
            "amount": min(0.24, 0.08 + (ref_priority - source_priority) / 500.0),
            "reason": f"{ref_id} has higher musical-role priority ({ref_priority:.0f} vs {source_priority:.0f}).",
        })

    return {
        "source_priority": source_priority,
        "decisions": tuple(decisions),
    }

def build_relationship_map(
    authority_analysis: dict[str, dict[str, object]],
    relationships: tuple[dict[str, object], ...] = (),
) -> dict[str, dict[str, object]]:
    """Resolve masking independently for every analyzed source."""
    return {
        source_id: resolve_masking(source_id, authority_analysis, relationships)
        for source_id in authority_analysis
    }
