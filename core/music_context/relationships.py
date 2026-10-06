"""Deterministic cross-asset musical relationship analysis for V0.

This layer turns independent analysis measurements into explainable relationships.
It deliberately preserves ambiguity instead of pretending that every disagreement
is a hard conflict.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from core.analysis.contracts import AudioAnalysis
from core.music_context.constraints import ContextConstraint, MusicalRelationship
from core.music_context.models import MusicalAuthority

BPM_TOLERANCE = 2.0
BPM_RELATION_TOLERANCE = 0.03
LOW_END_MASK_THRESHOLD = 0.55
SPECTRAL_OVERLAP_THRESHOLD = 0.18

_KEY_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


@dataclass(frozen=True)
class BPMRelationship:
    kind: str
    ratio: float
    difference: float


@dataclass(frozen=True)
class TonalRelationship:
    kind: str
    semitones: int | None = None


def classify_bpm(left: float, right: float, tolerance: float = BPM_TOLERANCE) -> BPMRelationship:
    difference = abs(left - right)
    if difference <= tolerance:
        return BPMRelationship("match", 1.0, difference)

    if left <= 0 or right <= 0:
        return BPMRelationship("unknown", 0.0, difference)

    ratio = max(left, right) / min(left, right)
    for target, kind in ((2.0, "double_time"), (0.5, "half_time")):
        normalized = ratio if target == 2.0 else 1.0 / ratio
        if abs(normalized - 1.0) <= BPM_RELATION_TOLERANCE:
            return BPMRelationship(kind, ratio, difference)

    return BPMRelationship("conflict", ratio, difference)


def _key_pitch_class(key: str) -> int | None:
    match = re.match(r"^([A-G](?:#|b)?)", key.strip())
    return _KEY_TO_PC.get(match.group(1)) if match else None


def classify_tonality(
    left_key: str,
    left_scale: str,
    right_key: str,
    right_scale: str,
) -> TonalRelationship:
    left_pc = _key_pitch_class(left_key)
    right_pc = _key_pitch_class(right_key)
    if left_pc is None or right_pc is None:
        return TonalRelationship("unknown")

    distance = (right_pc - left_pc) % 12
    if distance == 0:
        if left_scale.lower() == right_scale.lower():
            return TonalRelationship("match", 0)
        return TonalRelationship("parallel", 0)

    # Relative major/minor share the same pitch collection.
    scales = {left_scale.lower(), right_scale.lower()}
    if scales == {"major", "minor"}:
        if (left_scale.lower() == "major" and distance == 9) or (
            left_scale.lower() == "minor" and distance == 3
        ):
            return TonalRelationship("relative", distance)

    # Same pitch-class collection is a useful V0 compatibility signal.
    if distance in {5, 7}:
        return TonalRelationship("compatible", distance)

    return TonalRelationship("conflict", distance)


def _spectral_overlap(left: AudioAnalysis, right: AudioAnalysis) -> float:
    bands = (
        (left.low_energy_ratio, right.low_energy_ratio),
        (left.mid_energy_ratio, right.mid_energy_ratio),
        (left.high_energy_ratio, right.high_energy_ratio),
    )
    return sum(min(a, b) for a, b in bands if a is not None and b is not None)


def analyze_relationships(
    analyses: list[AudioAnalysis],
    authorities: list[MusicalAuthority],
) -> tuple[list[MusicalRelationship], list[ContextConstraint], list[str]]:
    relationships: list[MusicalRelationship] = []
    constraints: list[ContextConstraint] = []
    conflicts: list[str] = []

    authority_by_dimension: dict[str, MusicalAuthority] = {}
    for authority in authorities:
        current = authority_by_dimension.get(authority.dimension)
        if current is None or authority.confidence > current.confidence:
            authority_by_dimension[authority.dimension] = authority

    for i, left in enumerate(analyses):
        for right in analyses[i + 1 :]:
            if left.bpm is not None and right.bpm is not None:
                bpm = classify_bpm(left.bpm, right.bpm)
                if bpm.kind == "conflict":
                    relationships.append(MusicalRelationship(
                        left.asset_id, right.asset_id, "conflict", "rhythm", 1.0,
                        f"BPM differs by {bpm.difference:.2f}; no half/double-time relation detected.",
                    ))
                    conflicts.append(
                        f"BPM disagreement: {min(left.bpm, right.bpm):.2f}–{max(left.bpm, right.bpm):.2f} BPM."
                    )

            if all((left.key, left.scale, right.key, right.scale)):
                tonal = classify_tonality(left.key, left.scale, right.key, right.scale)
                if tonal.kind == "conflict":
                    relationships.append(MusicalRelationship(
                        left.asset_id, right.asset_id, "conflict", "harmony", 1.0,
                        f"Tonal centers differ by {tonal.semitones} semitone(s).",
                    ))
                    conflicts.append("Tonal disagreement: analyzed assets suggest different key/scale.")

            overlap = _spectral_overlap(left, right)
            if overlap >= LOW_END_MASK_THRESHOLD:
                if left.low_energy_ratio is not None and right.low_energy_ratio is not None:
                    relationships.append(MusicalRelationship(
                        left.asset_id, right.asset_id, "conflict", "low_end", overlap,
                        f"Likely low-end masking; shared low-band energy={overlap:.2f}.",
                    ))
                    conflicts.append(
                        f"Low-end overlap: {left.asset_id} and {right.asset_id} share substantial low-frequency energy."
                    )
            elif overlap >= SPECTRAL_OVERLAP_THRESHOLD:
                relationships.append(MusicalRelationship(
                    left.asset_id, right.asset_id, "dependency", "texture", overlap,
                    f"Potential spectral competition; shared band energy={overlap:.2f}.",
                ))

    for dimension, authority in authority_by_dimension.items():
        for target in analyses:
            if target.asset_id == authority.source_id:
                continue
            relationships.append(MusicalRelationship(
                authority.source_id, target.asset_id, "authority", dimension,
                authority.confidence,
                f"{authority.source_id} is authoritative for {dimension}.",
            ))
            constraint_type = {
                "rhythm": "adapt",
                "harmony": "adapt",
                "melody": "avoid",
                "low_end": "avoid",
                "texture": "adapt",
            }.get(dimension)
            if constraint_type:
                constraints.append(ContextConstraint(
                    target_id=target.asset_id,
                    type=constraint_type,
                    reference_id=authority.source_id,
                    dimension=dimension,
                    reason=f"Follow the {dimension} authority while preserving the target's identity.",
                ))

    # Deduplicate human-readable conflict entries while preserving order.
    unique_conflicts = list(dict.fromkeys(conflicts))
    return relationships, constraints, unique_conflicts


def merge_conflicts(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group))
