"""Event-aware time-domain masking for source-backed renders."""
from __future__ import annotations

import numpy as np


def _events_seconds(meta: dict[str, object], duration: float) -> list[tuple[float, float]]:
    bpm = meta.get("bpm")
    onsets = meta.get("onset_beats")
    durations = meta.get("note_durations_beats")
    if not isinstance(bpm, (int, float)) or float(bpm) <= 0:
        return []
    if not isinstance(onsets, (tuple, list)):
        return []
    beat_seconds = 60.0 / float(bpm)
    result = []
    for index, beat in enumerate(onsets):
        if not isinstance(beat, (int, float)):
            continue
        start = max(0.0, float(beat) * beat_seconds)
        if start >= duration:
            continue
        length = 0.0
        if isinstance(durations, (tuple, list)) and index < len(durations):
            value = durations[index]
            if isinstance(value, (int, float)):
                length = max(0.0, float(value) * beat_seconds)
        result.append((start, min(duration, start + length)))
    return result


def _window_for(meta: dict[str, object]) -> tuple[float, float]:
    role = str(meta.get("role", "")).lower()
    dimension = str(meta.get("dimension", "")).lower()
    if role in {"vocal", "lead", "melody"} or dimension == "melody":
        return 0.025, 0.16
    if role in {"kick", "drums"} or dimension == "rhythm":
        return 0.008, 0.12
    return 0.018, 0.10


def build_dynamic_envelope(
    num_samples: int,
    sr: int,
    source_meta: dict[str, object],
    reference_meta: dict[str, object],
    amount: float,
    attack_seconds: float | None = None,
    release_seconds: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a click-safe ducking envelope from reference musical events."""
    if num_samples <= 0 or sr <= 0 or amount <= 0:
        return np.ones(max(0, num_samples), dtype=np.float32), {
            "applied": False, "amount": 0.0, "events": 0,
        }
    duration = num_samples / float(sr)
    events = _events_seconds(reference_meta, duration)
    if not events:
        return np.ones(num_samples, dtype=np.float32), {
            "applied": False, "amount": 0.0, "events": 0,
        }
    attack, release = _window_for(reference_meta)
    if attack_seconds is not None:
        attack = max(0.001, float(attack_seconds))
    if release_seconds is not None:
        release = max(0.001, float(release_seconds))
    minimum = max(0.0, 1.0 - min(0.35, float(amount)))
    envelope = np.ones(num_samples, dtype=np.float32)
    for start, note_end in events:
        active_end = note_end if note_end > start else start + release
        left = max(0.0, start - attack)
        right = min(duration, active_end + release)
        a0 = max(0, int(round(left * sr)))
        a1 = max(a0, int(round(start * sr)))
        r0 = min(num_samples, max(a1, int(round(active_end * sr))))
        r1 = min(num_samples, max(r0, int(round(right * sr))))
        if a1 > a0:
            envelope[a0:a1] = np.minimum(
                envelope[a0:a1],
                np.linspace(1.0, minimum, a1 - a0, endpoint=False, dtype=np.float32),
            )
        if r0 > a1:
            envelope[a1:r0] = np.minimum(envelope[a1:r0], minimum)
        if r1 > r0:
            envelope[r0:r1] = np.minimum(
                envelope[r0:r1],
                np.linspace(minimum, 1.0, r1 - r0, endpoint=True, dtype=np.float32),
            )
    return envelope, {
        "applied": True,
        "amount": min(0.35, max(0.0, float(amount))),
        "events": len(events),
        "reference_id": str(reference_meta.get("_source_id", "")),
    }


def apply_dynamic_masking(
    data: np.ndarray,
    sr: int,
    source_meta: dict[str, object],
    reference_meta: dict[str, object],
    amount: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply event-aware ducking uniformly to all channels."""
    envelope, metadata = build_dynamic_envelope(
        len(data), sr, source_meta, reference_meta, amount
    )
    if not metadata["applied"]:
        return data.astype(np.float32, copy=True), metadata
    return (data.astype(np.float32, copy=False) * envelope[:, None]).astype(np.float32), metadata
