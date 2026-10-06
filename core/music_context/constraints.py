"""First-class musical constraints and relationships."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ConstraintType = Literal["fixed","follow","adapt","avoid"]
RelationshipType = Literal["authority","dependency","conflict"]

@dataclass(frozen=True)
class ContextConstraint:
    target_id: str
    type: ConstraintType
    reference_id: str | None = None
    dimension: str | None = None
    reason: str = ""

@dataclass(frozen=True)
class MusicalRelationship:
    source_id: str
    target_id: str
    type: RelationshipType
    dimension: str | None = None
    confidence: float = 1.0
    reason: str = ""
