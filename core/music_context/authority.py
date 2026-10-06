"""Infer musical authority from analysis results."""
from __future__ import annotations
from core.analysis.contracts import AudioAnalysis
from core.music_context.models import MusicalAuthority

ROLE_TO_DIMENSIONS = {
    "drums": ("rhythm",),
    "vocal/lead": ("melody",),
    "bass": ("low_end",),
    "guitar/piano": ("harmony", "texture"),
    "texture": ("texture",),
}

def infer_authority(analyses: list[AudioAnalysis]) -> list[MusicalAuthority]:
    authorities = []
    for a in analyses:
        if not a.role:
            continue
        confidence = a.confidence.get("role", 0.0)
        for dimension in ROLE_TO_DIMENSIONS.get(a.role, ()):
            authorities.append(MusicalAuthority(
                source_id=a.asset_id, dimension=dimension, confidence=confidence,
                rationale=f"Role inferred as {a.role}."))
    return authorities
