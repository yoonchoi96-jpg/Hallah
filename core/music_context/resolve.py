"""Resolve musical authority and detect cross-asset conflicts."""
from __future__ import annotations
from dataclasses import dataclass
from core.analysis.contracts import AudioAnalysis
from core.music_context.constraints import MusicalRelationship
from core.music_context.models import MusicalAuthority, SongContext

@dataclass(frozen=True)
class AuthorityResolution:
    dimension: str
    winner: MusicalAuthority | None
    alternatives: tuple[MusicalAuthority, ...] = ()

def resolve_authorities(authorities: list[MusicalAuthority]) -> list[AuthorityResolution]:
    by_dimension: dict[str, list[MusicalAuthority]] = {}
    for authority in authorities:
        by_dimension.setdefault(authority.dimension, []).append(authority)
    return [
        AuthorityResolution(
            dimension=dimension,
            winner=max(items, key=lambda x: x.confidence),
            alternatives=tuple(sorted(items, key=lambda x: x.confidence, reverse=True)[1:]),
        )
        for dimension, items in sorted(by_dimension.items())
    ]

def detect_conflicts(analyses: list[AudioAnalysis], bpm_tolerance: float = 2.0) -> list[str]:
    conflicts: list[str] = []
    bpms = [(a.asset_id, a.bpm) for a in analyses if a.bpm is not None]
    if bpms:
        lo = min(v for _, v in bpms)
        hi = max(v for _, v in bpms)
        if hi - lo > bpm_tolerance:
            conflicts.append(f"BPM disagreement: {lo:.2f}–{hi:.2f} BPM.")
    tonal = {(a.key, a.scale) for a in analyses if a.key and a.scale}
    if len(tonal) > 1:
        conflicts.append("Tonal disagreement: analyzed assets suggest different key/scale.")
    return conflicts

def build_relationships(
    analyses: list[AudioAnalysis],
    authorities: list[MusicalAuthority],
    bpm_tolerance: float = 2.0,
) -> list[MusicalRelationship]:
    relationships: list[MusicalRelationship] = []
    asset_ids = [a.asset_id for a in analyses]
    for authority in authorities:
        for target_id in asset_ids:
            if target_id == authority.source_id:
                continue
            relationships.append(MusicalRelationship(
                source_id=authority.source_id,
                target_id=target_id,
                type="authority",
                dimension=authority.dimension,
                confidence=authority.confidence,
                reason=f"{authority.source_id} is authoritative for {authority.dimension}.",
            ))
    for i, left in enumerate(analyses):
        for right in analyses[i + 1:]:
            if left.bpm is not None and right.bpm is not None and abs(left.bpm - right.bpm) > bpm_tolerance:
                relationships.append(MusicalRelationship(
                    source_id=left.asset_id, target_id=right.asset_id,
                    type="conflict", dimension="rhythm", confidence=1.0,
                    reason=f"BPM differs by {abs(left.bpm - right.bpm):.2f}.",
                ))
            if (left.key and left.scale and right.key and right.scale
                    and (left.key, left.scale) != (right.key, right.scale)):
                relationships.append(MusicalRelationship(
                    source_id=left.asset_id, target_id=right.asset_id,
                    type="conflict", dimension="harmony", confidence=1.0,
                    reason="Key/scale differs between analyzed assets.",
                ))
    return relationships

def enrich_context(context: SongContext, analyses: list[AudioAnalysis]) -> SongContext:
    new = context.next_version()
    new.conflicts = detect_conflicts(analyses)
    new.pending_decisions.clear()
    if new.conflicts:
        new.pending_decisions.extend(new.conflicts)
    return new
