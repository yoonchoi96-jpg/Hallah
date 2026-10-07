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


def apply_frequency_dynamic_masking(
    data: np.ndarray,
    sr: int,
    source_meta: dict[str, object],
    reference_meta: dict[str, object],
    amount: float,
    bands: tuple[str, ...] = ("low", "mid", "high"),
    ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Duck only overlapping spectral regions when reference events are active."""
    if len(data) == 0 or sr <= 0 or amount <= 0:
        return data.astype(np.float32, copy=True), {"applied": False, "events": 0, "bands": ()}
    duration = len(data) / float(sr)
    events = _events_seconds(reference_meta, duration)
    if not events:
        return data.astype(np.float32, copy=True), {"applied": False, "events": 0, "bands": ()}
    if ranges is None:
        ranges = {"low": (20.0, 180.0), "mid": (180.0, 2500.0), "high": (2500.0, sr * 0.5)}
    envelope, _ = build_dynamic_envelope(len(data), sr, source_meta, reference_meta, amount)
    frame = min(2048, max(512, 2 ** int(np.log2(max(512, min(len(data), 2048))))))
    hop = max(128, frame // 4)
    window = np.hanning(frame).astype(np.float32)
    padded = np.pad(data.astype(np.float32, copy=False), ((0, max(0, frame - len(data))), (0, 0)))
    out = np.zeros_like(padded)
    norm = np.zeros(len(padded), dtype=np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band_mask = np.zeros(len(freqs), dtype=bool)
    selected_ranges = {}
    for band in bands:
        lo, hi = ranges.get(band, (0.0, 0.0))
        lo, hi = max(0.0, float(lo)), min(float(sr) * 0.5, float(hi))
        if hi > lo:
            band_mask |= (freqs >= lo) & (freqs < hi)
            selected_ranges[band] = (lo, hi)
    minimum = max(0.0, 1.0 - min(0.35, float(amount)))
    for start in range(0, len(data), hop):
        stop = start + frame
        chunk = padded[start:stop]
        if len(chunk) < frame:
            chunk = np.pad(chunk, ((0, frame - len(chunk)), (0, 0)))
        center = min(len(data) - 1, start + frame // 2)
        event_gain = float(envelope[center])
        gain = np.ones(len(freqs), dtype=np.float32)
        gain[band_mask] *= minimum + (1.0 - minimum) * event_gain
        for channel in range(chunk.shape[1]):
            spectrum = np.fft.rfft(chunk[:, channel] * window)
            rendered = np.fft.irfft(spectrum * gain, n=frame).astype(np.float32)
            out[start:stop, channel] += rendered * window
        norm[start:stop] += window * window
    valid = norm > 1e-8
    out[valid] /= norm[valid, None]
    out[~valid] = 0.0
    return out[:len(data)].astype(np.float32), {
        "applied": True,
        "amount": min(0.35, max(0.0, float(amount))),
        "events": len(events),
        "bands": tuple(selected_ranges),
        "ranges": selected_ranges,
        "reference_id": str(reference_meta.get("_source_id", "")),
    }


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


def _spectral_collision_center(
    data: np.ndarray,
    sr: int,
    reference_center_hz: float,
    low_hz: float,
    high_hz: float,
) -> float:
    """Find the strongest source spectral peak near the reference center."""
    if len(data) == 0 or high_hz <= low_hz:
        return reference_center_hz
    mono = np.mean(data.astype(np.float32, copy=False), axis=1)
    frame = min(4096, max(512, 2 ** int(np.log2(max(512, min(len(mono), 4096))))))
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    spectrum = np.abs(np.fft.rfft(mono[:frame] * np.hanning(frame)))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    mask = (freqs >= max(low_hz, reference_center_hz * 0.45)) & (
        freqs <= min(high_hz, reference_center_hz * 1.8)
    )
    if not np.any(mask):
        return reference_center_hz
    indices = np.flatnonzero(mask)
    return float(freqs[indices[int(np.argmax(spectrum[indices]))]])


def apply_spectral_curve_dynamic_masking(
    data: np.ndarray,
    sr: int,
    source_meta: dict[str, object],
    reference_meta: dict[str, object],
    amount: float,
    bands: tuple[str, ...] = ("low", "mid", "high"),
    ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a smooth spectral ducking curve centered on the reference spectrum."""
    if len(data) == 0 or sr <= 0 or amount <= 0:
        return data.astype(np.float32, copy=True), {"applied": False, "events": 0, "curve": ()}
    duration = len(data) / float(sr)
    events = _events_seconds(reference_meta, duration)
    if not events:
        return data.astype(np.float32, copy=True), {"applied": False, "events": 0, "curve": ()}
    if ranges is None:
        ranges = {"low": (20.0, 180.0), "mid": (180.0, 2500.0), "high": (2500.0, sr * 0.5)}
    frame = min(2048, max(512, 2 ** int(np.log2(max(512, min(len(data), 2048))))))
    hop = max(128, frame // 4)
    window = np.hanning(frame).astype(np.float32)
    padded = np.pad(data.astype(np.float32, copy=False), ((0, max(0, frame - len(data))), (0, 0)))
    out = np.zeros_like(padded)
    norm = np.zeros(len(padded), dtype=np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    mask = np.zeros(len(freqs), dtype=bool)
    centers = []
    for band in bands:
        lo, hi = ranges.get(band, (0.0, 0.0))
        lo, hi = max(20.0, float(lo)), min(sr * 0.5, float(hi))
        if hi > lo:
            mask |= (freqs >= lo) & (freqs <= hi)
            centers.append((band, (lo + hi) * 0.5))
    mono_ref = reference_meta.get("fundamental_hz")
    ref_centroid = reference_meta.get("spectral_centroid_hz")
    if isinstance(mono_ref, (int, float)) and 20.0 <= float(mono_ref) <= sr * 0.5:
        center_hz = float(mono_ref)
    elif isinstance(ref_centroid, (int, float)) and 20.0 <= float(ref_centroid) <= sr * 0.5:
        center_hz = float(ref_centroid)
    elif centers:
        center_hz = sum(c for _, c in centers) / len(centers)
    else:
        center_hz = min(1000.0, sr * 0.25)
    if mask.any():
        selected = np.flatnonzero(mask)
        lo_hz = float(freqs[selected[0]])
        hi_hz = float(freqs[selected[-1]])
        center_hz = _spectral_collision_center(data, sr, center_hz, lo_hz, hi_hz)
    sigma = max(35.0, center_hz * 0.28)
    minimum = max(0.0, 1.0 - min(0.35, float(amount)))
    curve = np.ones(len(freqs), dtype=np.float32)
    if mask.any():
        distance = (freqs - center_hz) / sigma
        curve[mask] = 1.0 - (1.0 - minimum) * np.exp(-0.5 * distance * distance)[mask]
    envelope, _ = build_dynamic_envelope(len(data), sr, source_meta, reference_meta, amount)
    for start in range(0, len(data), hop):
        stop = start + frame
        chunk = padded[start:stop]
        if len(chunk) < frame:
            chunk = np.pad(chunk, ((0, frame - len(chunk)), (0, 0)))
        event_gain = float(envelope[min(len(data) - 1, start + frame // 2)])
        gain = 1.0 + (curve - 1.0) * (1.0 - event_gain)
        for channel in range(chunk.shape[1]):
            spectrum = np.fft.rfft(chunk[:, channel] * window)
            rendered = np.fft.irfft(spectrum * gain, n=frame).astype(np.float32)
            out[start:stop, channel] += rendered * window
        norm[start:stop] += window * window
    valid = norm > 1e-8
    out[valid] /= norm[valid, None]
    out[~valid] = 0.0
    return out[:len(data)].astype(np.float32), {
        "applied": True, "amount": min(0.35, max(0.0, float(amount))),
        "events": len(events), "bands": tuple(b for b, _ in centers),
        "center_hz": center_hz, "sigma_hz": sigma,
    }
