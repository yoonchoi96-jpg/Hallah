"""Candidate domain models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

CandidateDirection = Literal["identity","natural","bold","experimental"]

@dataclass(frozen=True)
class Candidate:
    id: str
    direction: CandidateDirection
    intent: str
    rationale: str
    parent_context_version: int
    audio_refs: tuple[str, ...] = ()
    midi_refs: tuple[str, ...] = ()
    parameter_changes: tuple[str, ...] = ()
