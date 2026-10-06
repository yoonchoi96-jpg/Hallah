"""Deterministic V0 candidate renderer.

This renderer produces real WAV preview artifacts from a simple synthetic signal.
It is intentionally provider-neutral: later Austin/AI/DSP generators can implement
the same protocol without changing candidate lifecycle code.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Mapping

import numpy as np

from audio.cache.keys import build_cache_key
from audio.rendering.contracts import RenderRequest, RenderResult


class MockAudioGenerator:
    """Create a short audible preview so the pipeline can be tested end-to-end."""

    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)

    def render(self, request: RenderRequest) -> RenderResult:
        if request.kind != "audio":
            raise ValueError("MockAudioGenerator only renders audio requests.")
        cache_key = build_cache_key(
            request.candidate_id,
            request.context_version,
            request.kind,
            request.parameter_changes,
        )
        output = self.cache_dir / f"{cache_key}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            self._write_preview(output, request)
        return RenderResult(
            candidate_id=request.candidate_id,
            kind="audio",
            artifact_ref=str(output),
            cache_key=cache_key,
            duration_seconds=2.0,
            sample_rate=44_100,
            metadata={"renderer": "mock", "preview": True},
        )

    @staticmethod
    def _write_preview(path: Path, request: RenderRequest) -> None:
        sample_rate = 44_100
        duration = 2.0
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        direction_phase: Mapping[str, float] = {
            "identity": 0.0,
            "natural": 0.35,
            "bold": 0.7,
            "experimental": 1.05,
        }
        phase = direction_phase.get(
            request.parameter_changes.get("direction", request.candidate_id.split("-")[0]),
            0.0,
        )
        frequency = 220.0 * (2.0 ** (phase / (2.0 * math.pi)))
        signal = 0.25 * np.sin((2.0 * math.pi * frequency * t) + phase)
        fade = np.minimum(1.0, np.minimum(t / 0.02, (duration - t) / 0.02))
        signal *= np.clip(fade, 0.0, 1.0)
        pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())


def render_candidate_audio(
    generator: MockAudioGenerator,
    request: RenderRequest,
) -> RenderResult:
    return generator.render(request)
