"""Deterministic audio preview renderer used by the V0 pipeline.

It synthesizes the same context-aware MIDI sequence used by the MIDI renderer, so
audio and MIDI candidates are musically aligned rather than unrelated placeholders.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from audio.cache.keys import build_cache_key
from audio.midi.generator import MidiSequence, generate_sequence
from audio.rendering.contracts import RenderRequest, RenderResult\nfrom audio.rendering.source import SourceAudioGenerator


class MockAudioGenerator:
    """Backward-compatible name for the deterministic V0 audio renderer."""

    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)\n        self.source_generator = SourceAudioGenerator(self.cache_dir)

    def render(self, request: RenderRequest) -> RenderResult:
        if request.kind != "audio":
            raise ValueError("MockAudioGenerator only renders audio requests.")
        if request.source_asset_ids:\n            if len(request.source_asset_ids) > 1:\n                return self.source_generator.render_mix(request)\n            return self.source_generator.render(request)\n        cache_key = build_cache_key(
            request.candidate_id,
            request.context_version,
            request.kind,
            request.parameter_changes,
        )
        output = self.cache_dir / f"{cache_key}.wav"
        sequence = generate_sequence(
            RenderRequest(
                candidate_id=request.candidate_id,
                context_version=request.context_version,
                kind="midi",
                intent=request.intent,
                source_asset_ids=request.source_asset_ids,
                parameter_changes=request.parameter_changes,
                output_format="wav",
            )
        )
        if not output.exists():
            self._write_sequence(output, sequence)
        duration = (
            max((n.start_beat + n.duration_beats for n in sequence.notes), default=4.0)
            * 60.0
            / sequence.tempo_bpm
        )
        return RenderResult(
            candidate_id=request.candidate_id,
            kind="audio",
            artifact_ref=str(output),
            cache_key=cache_key,
            duration_seconds=duration,
            sample_rate=44_100,
            metadata={"renderer": "deterministic-synth-v0.4", "preview": True},
        )

    @staticmethod
    def _oscillator(phase: np.ndarray, channel: int) -> np.ndarray:
        if channel == 1:
            return np.sin(phase)
        if channel == 2:
            return 0.65 * np.sin(phase) + 0.25 * np.sin(phase * 2.0) + 0.10 * np.sin(phase * 3.0)
        if channel == 9:
            return np.sin(phase) * np.exp(-4.5 * np.arange(len(phase)) / max(1, len(phase)))
        return np.sin(phase)

    @classmethod
    def _write_sequence(cls, path: Path, sequence: MidiSequence) -> None:
        sample_rate = 44_100
        beat_seconds = 60.0 / max(1.0, sequence.tempo_bpm)
        end_beat = max((n.start_beat + n.duration_beats for n in sequence.notes), default=4.0)
        duration = max(2.0, end_beat * beat_seconds + 0.1)
        total = int(math.ceil(duration * sample_rate))
        mix = np.zeros(total, dtype=np.float64)

        for note in sequence.notes:
            start = max(0, int(round(note.start_beat * beat_seconds * sample_rate)))
            length = max(1, int(round(note.duration_beats * beat_seconds * sample_rate)))
            stop = min(total, start + length)
            if stop <= start:
                continue
            count = stop - start
            t = np.arange(count, dtype=np.float64) / sample_rate
            phase = 2.0 * math.pi * (440.0 * 2.0 ** ((note.pitch - 69) / 12.0)) * t
            signal = cls._oscillator(phase, note.channel)
            attack = min(int(0.01 * sample_rate), count)
            release = min(int(0.04 * sample_rate), count)
            envelope = np.ones(count)
            if attack:
                envelope[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=False)
            if release:
                envelope[-release:] *= np.linspace(1.0, 0.0, release, endpoint=False)
            gain = min(1.0, max(0.05, note.velocity / 127.0)) * (
                0.18 if note.channel != 9 else 0.24
            )
            mix[start:stop] += signal * envelope * gain

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.92:
            mix *= 0.92 / peak
        pcm = np.clip(mix * 32767.0, -32768, 32767).astype("<i2")
        path.parent.mkdir(parents=True, exist_ok=True)
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
