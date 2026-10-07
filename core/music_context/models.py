"""Persistent musical state for a Hallah song."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from core.analysis.contracts import AudioAnalysis
from core.candidates.models import Candidate
from core.music_context.constraints import ContextConstraint, MusicalRelationship

AuthorityDimension = Literal["harmony", "rhythm", "melody", "low_end", "texture", "arrangement"]


@dataclass
class MusicalAuthority:
    source_id: str
    dimension: AuthorityDimension
    confidence: float
    rationale: str = ""


@dataclass
class SongContext:
    version: int = 0
    title: str = "Untitled"
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    time_signature: str = "4/4"
    genre: str | None = None
    mood: list[str] = field(default_factory=list)
    chord_progression: list[str] = field(default_factory=list)
    analyses: dict[str, AudioAnalysis] = field(default_factory=dict)
    authorities: list[MusicalAuthority] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    pending_decisions: list[str] = field(default_factory=list)
    constraints: list[ContextConstraint] = field(default_factory=list)
    relationships: list[MusicalRelationship] = field(default_factory=list)
    adaptation_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    candidate_history: list[str] = field(default_factory=list)

    def next_version(self) -> SongContext:
        new = deepcopy(self)
        new.version += 1
        return new

    def add_dependency(self, source_id: str, target_id: str) -> None:
        self.dependencies.setdefault(source_id, [])
        if target_id not in self.dependencies[source_id]:
            self.dependencies[source_id].append(target_id)

    def add_conflict(self, description: str) -> None:
        if description not in self.conflicts:
            self.conflicts.append(description)

    def record_decision(self, description: str, source: str = "user") -> SongContext:
        new = self.next_version()
        new.decisions.append(f"[{source}] {description}")
        return new
