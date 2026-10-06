"""Candidate domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

CandidateDirection = Literal["identity", "natural", "bold", "experimental"]
CandidateStatus = Literal[
    "proposed",
    "preview_ready",
    "selected",
    "rejected",
    "applied",
    "superseded",
]


@dataclass(frozen=True)
class Candidate:
    id: str
    direction: CandidateDirection
    intent: str
    rationale: str
    parent_context_version: int
    status: CandidateStatus = "proposed"
    audio_refs: tuple[str, ...] = ()
    midi_refs: tuple[str, ...] = ()
    parameter_changes: Mapping[str, object] = field(default_factory=dict)

    def with_status(self, status: CandidateStatus) -> "Candidate":
        """Return a new candidate with the requested lifecycle status."""
        return Candidate(
            id=self.id,
            direction=self.direction,
            intent=self.intent,
            rationale=self.rationale,
            parent_context_version=self.parent_context_version,
            status=status,
            audio_refs=self.audio_refs,
            midi_refs=self.midi_refs,
            parameter_changes=self.parameter_changes,
        )
