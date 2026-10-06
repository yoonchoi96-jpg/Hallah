"""Resolve musical authority and detect cross-asset conflicts."""
from __future__ import annotations
from dataclasses import dataclass
from core.analysis.contracts import AudioAnalysis
from core.music_context.models import MusicalAuthority, SongContext

@dataclass(frozen=True)
class AuthorityResolution:
    dimension: str
    winner: MusicalAuthority | None
    alternatives: tuple[MusicalAuthority, ...] = ()

def resolve_authorities(authorities: list[MusicalAuthority]) -> list[AuthorityResolution]:
    by_dimension: dict[str,list[MusicalAuthority]] = {}
    for authority in authorities:
        by_dimension.setdefault(authority.dimension, []).append(authority)
    return [
        AuthorityResolution(
            dimension=dimension,
            winner=max(items,key=lambda x:x.confidence),
            alternatives=tuple(sorted(items,key=lambda x:x.confidence,reverse=True)[1:]),
        )
        for dimension,items in sorted(by_dimension.items())
    ]

def detect_conflicts(analyses: list[AudioAnalysis], bpm_tolerance: float = 2.0) -> list[str]:
    conflicts=[]
    bpms=[(a.asset_id,a.bpm) for a in analyses if a.bpm is not None]
    if bpms:
        lo=min(v for _,v in bpms); hi=max(v for _,v in bpms)
        if hi-lo>bpm_tolerance:
            conflicts.append(f"BPM disagreement: {lo:.2f}–{hi:.2f} BPM.")
    tonal={(a.key,a.scale) for a in analyses if a.key and a.scale}
    if len(tonal)>1:
        conflicts.append("Tonal disagreement: analyzed assets suggest different key/scale.")
    return conflicts

def enrich_context(context: SongContext, analyses: list[AudioAnalysis]) -> SongContext:
    new=context.next_version()
    new.conflicts=detect_conflicts(analyses)
    new.pending_decisions.clear()
    if new.conflicts:
        new.pending_decisions.extend(new.conflicts)
    return new
