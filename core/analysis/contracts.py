"""Contracts and domain models for Hallah audio analysis V0."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AudioAnalysis:
    asset_id: str
    sample_rate: int | None = None
    duration_seconds: float | None = None
    channels: int | None = None
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    chords: tuple[str, ...] = ()
    role: str | None = None
    rms: float | None = None
    peak: float | None = None
    zero_crossing_rate: float | None = None
    spectral_centroid_hz: float | None = None
    spectral_rolloff_hz: float | None = None
    low_energy_ratio: float | None = None
    mid_energy_ratio: float | None = None
    high_energy_ratio: float | None = None
    stereo_width: float | None = None
    stereo_correlation: float | None = None
    onset_rate: float | None = None
    confidence: dict[str, float] = field(default_factory=dict)


class AudioAnalyzer(Protocol):
    def analyze(self, asset_id: str) -> AudioAnalysis: ...
