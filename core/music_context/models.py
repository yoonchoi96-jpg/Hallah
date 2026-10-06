"""Canonical domain models for a Hallah song."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

AuthorityDimension = Literal["harmony","rhythm","melody","low_end","texture","arrangement"]

@dataclass(frozen=True)
class MusicalAuthority:
    source_id: str
    dimension: AuthorityDimension
    confidence: float
    rationale: str = ""

@dataclass
class SongContext:
    version: int = 0
    title: str | None = None
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    time_signature: tuple[int, int] = (4, 4)
    genre: str | None = None
    mood: str | None = None
    chord_progression: list[str] = field(default_factory=list)
    authorities: list[MusicalAuthority] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    def next_version(self) -> "SongContext":
        import copy
        result = copy.deepcopy(self)
        result.version += 1
        return result
