"""Deterministic musical-role-aware masking decisions for V0."""
from __future__ import annotations

ROLE_PRIORITY = {"vocal": 100, "lead": 95, "melody": 95, "drums": 90, "kick": 92, "bass": 88, "harmony": 70, "guitar": 68, "piano": 65, "pad": 55, "texture": 45}
DIMENSION_PRIORITY = {"melody": 100, "rhythm": 90, "low_end": 88, "harmony": 70, "texture": 50, "arrangement": 60}

def _priority(meta: dict[str, object]) -> float:
    return max(ROLE_PRIORITY.get(str(meta.get("role", "")).lower(), 40), DIMENSION_PRIORITY.get(str(meta.get("dimension", "")).lower(), 40))

def _band_ranges(meta: dict[str, object], sr: int) -> dict[str, tuple[float, float]]:
    nyquist = float(sr) * 0.5
    fundamental = meta.get("fundamental_hz")
    centroid = meta.get("spectral_centroid_hz")
    if isinstance(fundamental, (int, float)) and 35.0 <= float(fundamental) <= 1000.0:
        f = float(fundamental)
        low = (max(20.0, f * 0.55), min(220.0, f * 1.8))
    else:
        low = (20.0, 180.0)
    if isinstance(centroid, (int, float)) and float(centroid) > 500.0:
        c = float(centroid)
        mid = (max(180.0, c * 0.45), min(4000.0, c * 1.35))
    else:
        mid = (180.0, 2500.0)
    return {"low": low, "mid": mid, "high": (min(nyquist, max(2500.0, mid[1])), nyquist)}

def resolve_masking(source_id: str, authority_analysis: dict[str, dict[str, object]], relationships: tuple[dict[str, object], ...] = ()) -> dict[str, object]:
    source = authority_analysis.get(source_id, {})
    source_priority = _priority(source)
    decisions = []
    relationship_index: dict[tuple[str, str], list[dict[str, object]]] = {}
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        rel_source = relationship.get("source_id")
        rel_target = relationship.get("target_id")
        rel_type = relationship.get("type")
        if isinstance(rel_source, str) and isinstance(rel_target, str) and isinstance(rel_type, str):
            relationship_index.setdefault((rel_source, rel_target), []).append(relationship)

    for ref_id, ref in authority_analysis.items():
        if ref_id == source_id:
            continue
        explicit = relationship_index.get((source_id, ref_id), []) + relationship_index.get((ref_id, source_id), [])
        conflict_relationship = any(str(rel.get("type")) == "conflict" for rel in explicit)
        explicit_authority = [
            rel for rel in explicit
            if str(rel.get("type")) == "authority"
            and (rel.get("dimension") is None or str(rel.get("dimension")) == str(source.get("dimension", "")))
        ]
        if explicit_authority:
            ref_priority = max(_priority(ref), 100.0 + max(float(rel.get("confidence", 1.0) or 1.0) for rel in explicit_authority) * 10.0)
        else:
            ref_priority = _priority(ref)
        shared = []
        for band, key in (("low", "low_energy_ratio"), ("mid", "mid_energy_ratio"), ("high", "high_energy_ratio")):
            if (source.get(key, 0) or 0) > 0.10 and (ref.get(key, 0) or 0) > 0.10:
                shared.append(band)
        if not shared or source_priority >= ref_priority:
            continue
        reason = f"{ref_id} has higher musical-role priority ({ref_priority:.0f} vs {source_priority:.0f})."
        if explicit_authority:
            reason = f"{ref_id} is an explicit authority relationship for {source.get('dimension', 'the relevant dimension')}."
        elif conflict_relationship:
            reason = f"{ref_id} has an explicit musical conflict relationship."
        decisions.append({"reference_id": ref_id, "bands": tuple(shared), "ranges": {band: _band_ranges(source, 48000)[band] for band in shared}, "priority": ref_priority, "amount": min(0.24, 0.08 + (ref_priority - source_priority) / 500.0), "reason": reason})
    return {"source_priority": source_priority, "decisions": tuple(decisions)}

def build_relationship_map(authority_analysis: dict[str, dict[str, object]], relationships: tuple[dict[str, object], ...] = ()) -> dict[str, dict[str, object]]:
    return {source_id: resolve_masking(source_id, authority_analysis, relationships) for source_id in authority_analysis}

def build_mix_gain_plan(authority_analysis: dict[str, dict[str, object]], relationship_map: dict[str, dict[str, object]], max_total_duck: float = 0.42) -> dict[str, tuple[dict[str, object], ...]]:
    """Bound cumulative ducking so dense mixes do not collapse."""
    plan = {}
    for source_id, relationship in relationship_map.items():
        decisions = relationship.get("decisions", ()) if isinstance(relationship, dict) else ()
        ranked = []
        for decision in decisions if isinstance(decisions, (list, tuple)) else ():
            if not isinstance(decision, dict):
                continue
            ref_id = decision.get("reference_id")
            if not isinstance(ref_id, str) or ref_id not in authority_analysis:
                continue
            ref = authority_analysis[ref_id]
            overlap = sum(float(ref.get(k, 0.0) or 0.0) for k in ("low_energy_ratio", "mid_energy_ratio", "high_energy_ratio"))
            ranked.append((float(decision.get("priority", 0.0)), overlap, decision))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        remaining = max(0.0, float(max_total_duck))
        allocated = []
        for _, _, decision in ranked:
            requested = min(0.24, max(0.0, float(decision.get("amount", 0.0))))
            amount = min(requested, remaining)
            if amount <= 0:
                continue
            item = dict(decision)
            item["allocated_amount"] = amount
            item["budget_remaining"] = max(0.0, remaining - amount)
            allocated.append(item)
            remaining -= amount
            if remaining <= 1e-6:
                break
        plan[source_id] = tuple(allocated)
    return plan
