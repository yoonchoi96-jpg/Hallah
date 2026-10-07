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
    reference_data: np.ndarray | None = None,
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



def _hz_to_bark(hz: np.ndarray) -> np.ndarray:
    """Map frequency to the psychoacoustic Bark scale."""
    hz = np.maximum(np.asarray(hz, dtype=np.float32), 0.0)
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def _critical_band_smoothing(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    bark_sigma: float = 0.55,
) -> np.ndarray:
    """Smooth spectral energy over a compact Bark-scale critical-band neighborhood."""
    power = np.square(np.asarray(spectrum, dtype=np.float32))
    bark = _hz_to_bark(freqs)
    result = np.zeros_like(power)
    # A direct Bark-domain kernel is cheap at the small FFT sizes used here.
    for index, center in enumerate(bark):
        distance = (bark - center) / max(0.1, bark_sigma)
        weights = np.exp(-0.5 * distance * distance)
        result[index] = float(np.sum(power * weights) / max(float(np.sum(weights)), 1e-9))
    return result


def _stereo_components(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Mid/Side components while keeping mono input on the Mid path."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        return np.asarray(array, dtype=np.float32).reshape(-1), np.zeros(len(array), dtype=np.float32)
    if array.shape[1] == 1:
        return array[:, 0], np.zeros(len(array), dtype=np.float32)
    left = array[:, 0]
    right = array[:, 1]
    return (left + right) * 0.5, (left - right) * 0.5


def _spectral_component_collision(
    source_data: np.ndarray,
    reference_data: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Find collision peaks independently for Mid and Side energy."""
    if len(source_data) == 0 or len(reference_data) == 0:
        return (), ()
    frame = min(
        4096,
        max(512, 2 ** int(np.log2(max(512, min(len(source_data), len(reference_data), 4096))))),
    )
    source_mid, source_side = _stereo_components(source_data)
    reference_mid, reference_side = _stereo_components(reference_data)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    mask = (freqs >= max(20.0, low_hz)) & (freqs <= min(sr * 0.5, high_hz))
    if not np.any(mask):
        return (), ()

    def spectrum(component: np.ndarray) -> np.ndarray:
        if len(component) < frame:
            component = np.pad(component, (0, frame - len(component)))
        return np.abs(np.fft.rfft(component[:frame] * np.hanning(frame)))

    def peaks_for(source_component: np.ndarray, reference_component: np.ndarray) -> tuple[float, ...]:
        source_spec = spectrum(source_component)
        reference_spec = spectrum(reference_component)
        source_cb = _critical_band_smoothing(source_spec, freqs)
        reference_cb = _critical_band_smoothing(reference_spec, freqs)
        src_peak = max(float(np.max(source_cb)), 1e-9)
        ref_peak = max(float(np.max(reference_cb)), 1e-9)
        src = source_cb / src_peak
        ref = reference_cb / ref_peak
        # Psychoacoustic floor: weak spectral tails should not trigger strong
        # masking merely because they occupy the same Bark band.
        source_threshold = np.maximum(0.0, (src - 0.10) / 0.90)
        reference_threshold = np.maximum(0.0, (ref - 0.10) / 0.90)
        collision = np.sqrt(source_threshold * reference_threshold)
        indices = np.flatnonzero(mask)
        values = collision[indices].copy()
        peaks = []
        spacing = max(2, int(round(45.0 / (freqs[1] - freqs[0]))))
        for _ in range(4):
            i = int(np.argmax(values))
            if values[i] < 0.12:
                break
            idx = int(indices[i])
            center_bark = _hz_to_bark(np.asarray([freqs[idx]], dtype=np.float32))[0]
            bark_distance = np.abs(_hz_to_bark(freqs) - center_bark)
            local = np.where(mask & (bark_distance <= 0.55), source_spec, 0.0)
            peak_idx = int(np.argmax(local))
            peaks.append(float(freqs[peak_idx] if local[peak_idx] > 0 else freqs[idx]))
            left, right = max(0, i - spacing), min(len(values), i + spacing + 1)
            values[left:right] = 0.0
        return tuple(peaks)

    return peaks_for(source_mid, reference_mid), peaks_for(source_side, reference_side)


def _spectral_collision_peaks(
    source_data: np.ndarray,
    reference_data: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    max_peaks: int = 4,
) -> tuple[float, ...]:
    """Backward-compatible collision peaks using the Mid component."""
    mid, _ = _spectral_component_collision(
        source_data, reference_data, sr, low_hz, high_hz
    )
    return mid[:max_peaks]


def _spectral_collision_centers(
    data: np.ndarray,    sr: int,
    reference_center_hz: float,
    low_hz: float,
    high_hz: float,
    max_peaks: int = 4,
) -> tuple[float, ...]:
    """Find several strong source peaks near the reference center."""
    if len(data) == 0 or high_hz <= low_hz:
        return (reference_center_hz,)
    mono = np.mean(data.astype(np.float32, copy=False), axis=1)
    frame = min(4096, max(512, 2 ** int(np.log2(max(512, min(len(mono), 4096))))))
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    spectrum = np.abs(np.fft.rfft(mono[:frame] * np.hanning(frame)))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    mask = (freqs >= max(low_hz, reference_center_hz * 0.45)) & (freqs <= min(high_hz, reference_center_hz * 1.8))
    indices = np.flatnonzero(mask)
    if not len(indices):
        return (reference_center_hz,)
    values = spectrum[indices].copy()
    peaks = []
    spacing = max(2, int(round(40.0 / (freqs[1] - freqs[0]))))
    for _ in range(max(1, max_peaks)):
        i = int(np.argmax(values))
        if values[i] <= 0:
            break
        idx = int(indices[i])
        peaks.append(float(freqs[idx]))
        left, right = max(0, i - spacing), min(len(values), i + spacing + 1)
        values[left:right] = 0.0
    return tuple(peaks) or (reference_center_hz,)


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
    reference_data: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a smooth spectral ducking curve with optional frame-by-frame spectral tracking."""
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
    centers_hz = (center_hz,)
    if mask.any():
        selected = np.flatnonzero(mask)
        lo_hz = float(freqs[selected[0]])
        hi_hz = float(freqs[selected[-1]])
        if reference_data is not None:
            centers_hz = _spectral_collision_peaks(data, reference_data, sr, lo_hz, hi_hz)
        else:
            centers_hz = _spectral_collision_centers(data, sr, center_hz, lo_hz, hi_hz)
        if not centers_hz:
            centers_hz = (center_hz,)
        center_hz = centers_hz[0]
    sigma = max(35.0, center_hz * 0.28)
    minimum = max(0.0, 1.0 - min(0.35, float(amount)))
    curve = np.ones(len(freqs), dtype=np.float32)
    if mask.any():
        distance = (freqs - center_hz) / sigma
        multi_curve = np.zeros_like(freqs, dtype=np.float32)
        for peak_hz in centers_hz:
            peak_sigma = max(35.0, float(peak_hz) * 0.28)
            peak_distance = (freqs - peak_hz) / peak_sigma
            multi_curve = np.maximum(multi_curve, np.exp(-0.5 * peak_distance * peak_distance).astype(np.float32))
        curve[mask] = 1.0 - (1.0 - minimum) * multi_curve[mask]
    envelope, _ = build_dynamic_envelope(len(data), sr, source_meta, reference_meta, amount)
    tracked_centers: list[float] = []
    tracked_side_centers: list[float] = []
    tracked_side_strengths: list[float] = []
    tracked_strengths: list[float] = []
    smoothed_centers: list[float] = []
    smoothed_strengths: list[float] = []
    previous_centers: list[float] = []
    previous_strengths: list[float] = []
    previous_side_centers: list[float] = []
    previous_side_strengths: list[float] = []
    max_center_jump_hz = 180.0
    center_smoothing = 0.35
    strength_attack = 0.45
    strength_release = 0.16
    reference_padded = None
    if reference_data is not None and len(reference_data):
        reference_padded = np.pad(
            reference_data.astype(np.float32, copy=False),
            ((0, max(0, frame - len(reference_data))), (0, 0)),
        )
    for start in range(0, len(data), hop):
        stop = start + frame
        chunk = padded[start:stop]
        if len(chunk) < frame:
            chunk = np.pad(chunk, ((0, frame - len(chunk)), (0, 0)))
        event_gain = float(envelope[min(len(data) - 1, start + frame // 2)])
        local_curve = curve
        if reference_padded is not None and mask.any():
            ref_chunk = reference_padded[start:stop]
            if len(ref_chunk) < frame:
                ref_chunk = np.pad(ref_chunk, ((0, frame - len(ref_chunk)), (0, 0)))
            mid_centers, side_centers = _spectral_component_collision(
                chunk, ref_chunk, sr, lo_hz, hi_hz
            )
            local_centers = mid_centers
            mono_s = np.mean(chunk, axis=1)
            mono_r = np.mean(ref_chunk, axis=1)
            ss = np.abs(np.fft.rfft(mono_s * window))
            rr = np.abs(np.fft.rfft(mono_r * window))
            denom_s = max(float(np.max(ss)), 1e-9)
            denom_r = max(float(np.max(rr)), 1e-9)
            rms_s = max(float(np.sqrt(np.mean(np.square(mono_s)))), 1e-9)
            rms_r = max(float(np.sqrt(np.mean(np.square(mono_r)))), 1e-9)
            level_balance = min(1.0, rms_s / rms_r, rms_r / rms_s)
            freqs_local = np.fft.rfftfreq(frame, 1.0 / sr)

            def component_strengths(
                source_component: np.ndarray,
                reference_component: np.ndarray,
                peaks: tuple[float, ...],
            ) -> list[float]:
                ss_component = np.abs(np.fft.rfft(source_component * window))
                rr_component = np.abs(np.fft.rfft(reference_component * window))
                ds = max(float(np.max(ss_component)), 1e-9)
                dr = max(float(np.max(rr_component)), 1e-9)
                rms_a = max(float(np.sqrt(np.mean(np.square(source_component)))), 1e-9)
                rms_b = max(float(np.sqrt(np.mean(np.square(reference_component)))), 1e-9)
                balance = min(1.0, rms_a / rms_b, rms_b / rms_a)
                return [
                    float(np.sqrt(
                        (ss_component[int(np.argmin(np.abs(freqs_local - peak)))] / ds)
                        * (rr_component[int(np.argmin(np.abs(freqs_local - peak)))] / dr)
                    ) * balance)
                    for peak in peaks
                ]

            source_mid, source_side = _stereo_components(chunk)
            reference_mid, reference_side = _stereo_components(ref_chunk)
            mid_strengths = component_strengths(source_mid, reference_mid, mid_centers)
            side_strengths = component_strengths(source_side, reference_side, side_centers)

            # Track peaks across adjacent frames. Limit movement first, then
            # smooth frequency so a changing collision does not jump abruptly.
            current = list(local_centers)
            matched_previous = set()
            next_centers: list[float] = []
            next_strengths: list[float] = []
            for peak, raw_strength in zip(current, mid_strengths):
                best = None
                best_distance = max_center_jump_hz + 1.0
                for idx, previous in enumerate(previous_centers):
                    if idx in matched_previous:
                        continue
                    distance = abs(float(peak) - float(previous))
                    if distance < best_distance:
                        best = idx
                        best_distance = distance
                if best is None:
                    smoothed_peak = float(peak)
                    smoothed_strength = float(raw_strength)
                else:
                    matched_previous.add(best)
                    previous = float(previous_centers[best])
                    delta = float(peak) - previous
                    delta = max(-max_center_jump_hz, min(max_center_jump_hz, delta))
                    target = previous + delta
                    smoothed_peak = previous + center_smoothing * (target - previous)
                    previous_strength = float(previous_strengths[best])
                    coeff = strength_attack if raw_strength > previous_strength else strength_release
                    smoothed_strength = previous_strength + coeff * (float(raw_strength) - previous_strength)
                next_centers.append(smoothed_peak)
                next_strengths.append(smoothed_strength)

            previous_centers = next_centers
            previous_strengths = next_strengths
            # Smooth the side path independently. Side-only collisions
            # must not force equal ducking in the Mid path.
            current_side = list(side_centers)
            next_side_centers: list[float] = []
            next_side_strengths: list[float] = []
            matched_side = set()
            for peak, raw_strength in zip(current_side, side_strengths):
                best = None
                best_distance = max_center_jump_hz + 1.0
                for idx, previous in enumerate(previous_side_centers):
                    if idx in matched_side:
                        continue
                    distance = abs(float(peak) - float(previous))
                    if distance < best_distance:
                        best = idx
                        best_distance = distance
                if best is None:
                    smooth_peak = float(peak)
                    smooth_strength = float(raw_strength)
                else:
                    matched_side.add(best)
                    previous = float(previous_side_centers[best])
                    delta = max(-max_center_jump_hz, min(max_center_jump_hz, float(peak) - previous))
                    smooth_peak = previous + center_smoothing * delta
                    old_strength = float(previous_side_strengths[best])
                    coeff = strength_attack if raw_strength > old_strength else strength_release
                    smooth_strength = old_strength + coeff * (float(raw_strength) - old_strength)
                next_side_centers.append(smooth_peak)
                next_side_strengths.append(smooth_strength)
            previous_side_centers = next_side_centers
            previous_side_strengths = next_side_strengths
            tracked_side_centers.extend(float(v) for v in side_centers)
            tracked_side_strengths.extend(float(v) for v in side_strengths)

            if next_centers or next_side_centers:
                tracked_centers.extend(float(v) for v in local_centers)
                tracked_strengths.extend(float(v) for v in mid_strengths)
                smoothed_centers.extend(next_centers)
                smoothed_strengths.extend(next_strengths)
                local_curve_mid = np.ones(len(freqs), dtype=np.float32)
                local_curve_side = np.ones(len(freqs), dtype=np.float32)
                for peak, strength in zip(next_centers, next_strengths):
                    peak_sigma = max(35.0, float(peak) * 0.28)
                    peak_curve = np.exp(
                        -0.5 * ((freqs - peak) / peak_sigma) ** 2
                    ).astype(np.float32)
                    strength_floor = 0.12
                    normalized_strength = min(
                        1.0, max(0.0, (float(strength) - strength_floor) / (1.0 - strength_floor))
                    )
                    depth = (1.0 - minimum) * (normalized_strength ** 1.5)
                    local_curve_mid = np.minimum(local_curve_mid, 1.0 - peak_curve * depth)
                for peak, strength in zip(next_side_centers, next_side_strengths):
                    peak_sigma = max(35.0, float(peak) * 0.28)
                    peak_curve = np.exp(-0.5 * ((freqs - peak) / peak_sigma) ** 2).astype(np.float32)
                    normalized_strength = min(1.0, max(0.0, (float(strength) - 0.12) / 0.88))
                    depth = (1.0 - minimum) * (normalized_strength ** 1.5)
                    local_curve_side = np.minimum(local_curve_side, 1.0 - peak_curve * depth)
                local_curve_mid[~mask] = 1.0
                local_curve_side[~mask] = 1.0
                gain_mid = 1.0 + (local_curve_mid - 1.0) * (1.0 - event_gain)
                gain_side = 1.0 + (local_curve_side - 1.0) * (1.0 - event_gain)
        else:
            gain_mid = 1.0 + (local_curve - 1.0) * (1.0 - event_gain)
            gain_side = gain_mid
        if chunk.shape[1] == 1:
            spectrum = np.fft.rfft(chunk[:, 0] * window)
            rendered = np.fft.irfft(spectrum * gain_mid, n=frame).astype(np.float32)
            out[start:stop, 0] += rendered * window
        else:
            mid, side = _stereo_components(chunk)
            mid_rendered = np.fft.irfft(np.fft.rfft(mid * window) * gain_mid, n=frame).astype(np.float32)
            side_rendered = np.fft.irfft(np.fft.rfft(side * window) * gain_side, n=frame).astype(np.float32)
            left = (mid_rendered + side_rendered) * window
            right = (mid_rendered - side_rendered) * window
            out[start:stop, 0] += left
            out[start:stop, 1] += right
        norm[start:stop] += window * window
    valid = norm > 1e-8
    out[valid] /= norm[valid, None]
    out[~valid] = 0.0
    return out[:len(data)].astype(np.float32), {
        "applied": True, "amount": min(0.35, max(0.0, float(amount))),
        "events": len(events), "bands": tuple(b for b, _ in centers),
        "center_hz": center_hz, "center_hz_all": tuple(centers_hz), "sigma_hz": sigma,
        "tracking": reference_data is not None,
        "stereo_mode": "mid_side" if data.shape[1] >= 2 else "mid_mono",
        "tracked_centers_hz": tuple(tracked_centers),
        "tracked_side_centers_hz": tuple(tracked_side_centers),
        "tracked_side_strengths": tuple(tracked_side_strengths),
        "tracked_strengths": tuple(tracked_strengths),
        "smoothed_centers_hz": tuple(smoothed_centers),
        "smoothed_strengths": tuple(smoothed_strengths),
        "tracked_frame_count": len(smoothed_centers),
        "tracking_strength_max": max(tracked_strengths, default=0.0),
        "tracking_strength_mean": float(np.mean(tracked_strengths)) if tracked_strengths else 0.0,
        "smoothed_strength_max": max(smoothed_strengths, default=0.0),
        "smoothed_strength_mean": float(np.mean(smoothed_strengths)) if smoothed_strengths else 0.0,
        "tracking_max_center_jump_hz": max(
            (abs(b - a) for a, b in zip(smoothed_centers, smoothed_centers[1:])),
            default=0.0,
        ),
    }