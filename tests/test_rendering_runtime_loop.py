import math
import wave

import numpy as np
import pytest

from audio.rendering.audition import write_audition_manifest
from audio.rendering.contracts import (
    RenderResult,
    build_audition_manifest,
    build_synchronized_audition,
)
from audio.rendering.runtime import AuditionRuntime


def _write_tone(path, hz, sample_rate=22050, seconds=1.0):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    x = 0.15 * np.sin(2 * math.pi * hz * t)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((x * 32767).astype("<i2").tobytes())


def test_runtime_reads_pcm_across_loop_boundary(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_tone(path, 220)

    result = RenderResult(
        candidate_id="A",
        kind="audio",
        artifact_ref=str(path),
        cache_key="A",
        duration_seconds=1.0,
        sample_rate=22050,
    )
    audition = build_synchronized_audition(
        30,
        [result],
        loop_start_seconds=0.4,
        loop_end_seconds=0.6,
    )
    manifest = build_audition_manifest(audition)
    runtime = AuditionRuntime(
        manifest,
        initial_candidate_id="A",
        initial_position_seconds=0.55,
    )

    frame = runtime.read_frames(round(0.1 * 22050))

    assert frame.start_seconds == 0.55
    assert runtime.state.position_seconds == pytest.approx(0.45)
    assert len(frame.pcm) == frame.frames * frame.channels * frame.sample_width
    assert np.frombuffer(frame.pcm, dtype="<i2").size == frame.frames


def test_runtime_manifest_writer_preserves_disabled_loop(tmp_path):
    path = tmp_path / "candidate.wav"
    _write_tone(path, 220)
    result = RenderResult(
        candidate_id="A",
        kind="audio",
        artifact_ref=str(path),
        cache_key="A",
        duration_seconds=1.0,
        sample_rate=22050,
    )
    audition = build_synchronized_audition(31, [result])
    output = tmp_path / "manifest.json"

    write_audition_manifest(audition, output)

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["loop"] == {"start_seconds": 0.0, "end_seconds": None}
