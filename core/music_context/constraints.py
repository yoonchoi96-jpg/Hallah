"""First-class musical constraints and relationships."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ConstraintType = Literal["fixed","follow","adapt","avoid"]

# Adaptation dimensions are intentionally independent: a source may follow
# another source for rhythm while using a different source for harmony.
AdaptationDimension = Literal["bpm","tempo","key","pitch","harmony","rhythm","melody","low_end","texture","arrangement"]
RelationshipType = Literal["authority","dependency","conflict"]

@dataclass(frozen=True)
class ContextConstraint:
    target_id: str
    type: ConstraintType
    reference_id: str | None = None
    dimension: str | None = None
    reason: str = ""
    parameters: tuple[str, ...] = ()

@dataclass(frozen=True)
class MusicalRelationship:
    source_id: str
    target_id: str
    type: RelationshipType
    dimension: str | None = None
    confidence: float = 1.0
    reason: str = ""
