"""Contracts for analysis providers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(frozen=True)
class AudioAnalysis:
    asset_id: str
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    chords: tuple[str, ...] = ()
    role: str | None = None
    confidence: dict[str, float] = field(default_factory=dict)

class AudioAnalyzer(Protocol):
    def analyze(self, asset_id: str) -> AudioAnalysis: ...
