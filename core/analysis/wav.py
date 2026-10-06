"""Deterministic PCM WAV feature extraction for Hallah V0."""
from __future__ import annotations
import math
import wave
from pathlib import Path
import numpy as np
from core.analysis.contracts import AudioAnalysis

class WavAnalyzer:
    def analyze(self, asset_id: str) -> AudioAnalysis:
        path = Path(asset_id)
        if path.suffix.lower() != ".wav":
            raise ValueError("WavAnalyzer accepts WAV files only")
        with wave.open(str(path), "rb") as wf:
            channels, sr, width, frames = wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getnframes()
            raw = wf.readframes(frames)
        x = self._decode(raw, width, channels)
        mono = x.mean(axis=1) if channels > 1 else x[:, 0]
        if not len(mono):
            raise ValueError("WAV contains no samples")
        n = min(len(mono), max(sr * 8, 4096))
        seg = mono[:n]
        window = np.hanning(n)
        power = np.abs(np.fft.rfft(seg * window)) ** 2
        freqs = np.fft.rfftfreq(n, 1 / sr)
        total = float(power.sum()) + 1e-12
        centroid = float((freqs * power).sum() / total)
        cumulative = np.cumsum(power)
        rolloff = float(freqs[min(int(np.searchsorted(cumulative, total * .85)), len(freqs) - 1)])
        band = lambda a, b: float(power[(freqs >= a) & (freqs < b)].sum() / total)
        rms = float(np.sqrt(np.mean(mono * mono)))
        peak = float(np.max(np.abs(mono)))
        zcr = float(np.mean(mono[:-1] * mono[1:] < 0)) if len(mono) > 1 else 0.0
        bpm, bpm_conf = self._bpm(mono, sr)
        fundamental, pitch_conf = self._pitch(seg, sr)
        key, scale, key_conf = self._key(mono, sr)
        onset_rate, transient_ratio, attack, decay, sustain, release = self._envelope(mono, sr)
        onset_beats = self._onset_beats(mono, sr, bpm)
        note_pitches, note_durations_beats = self._note_events(mono, sr, bpm, role)
        width_value, corr = 0.0, 1.0
        if channels >= 2:
            left, right = x[:, 0], x[:, 1]
            corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else 1.0
            mid_power = float(np.mean(((left + right) * .5) ** 2))
            side_power = float(np.mean(((left - right) * .5) ** 2))
            width_value = min(1.0, side_power / (mid_power + side_power + 1e-12))
        low, mid, high = band(20, 250), band(250, 4000), band(4000, 20000)
        role = (
            "bass" if low > .48 and centroid < 700
            else "drums" if onset_rate > 2.5 and zcr > .08
            else "vocal/lead" if centroid > 2200 and zcr > .06
            else "guitar/piano" if mid > low and mid > high
            else "texture"
        )
        return AudioAnalysis(
            asset_id=str(path), sample_rate=sr, duration_seconds=len(mono) / sr, channels=channels,
            bpm=bpm, key=key, scale=scale, role=role, rms=rms, peak=peak,
            onset_beats=onset_beats, note_pitches=note_pitches,
            note_durations_beats=note_durations_beats,
            zero_crossing_rate=zcr, spectral_centroid_hz=centroid, spectral_rolloff_hz=rolloff,
            low_energy_ratio=low, mid_energy_ratio=mid, high_energy_ratio=high,
            stereo_width=width_value, stereo_correlation=corr, onset_rate=onset_rate,
            transient_ratio=transient_ratio, attack_seconds=attack, decay_seconds=decay,
            sustain_level=sustain, release_seconds=release, fundamental_hz=fundamental,
            pitch_confidence=pitch_conf,
            confidence={"bpm": bpm_conf, "key": key_conf, "pitch": pitch_conf, "role": .45, "spectral": .95},
        )

    @staticmethod
    def _decode(raw, width, channels):
        if width == 1:
            values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        elif width == 3:
            data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            packed = data[:, 0].astype(np.int32) | (data[:, 1].astype(np.int32) << 8) | (data[:, 2].astype(np.int32) << 16)
            packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
            values = packed.astype(np.float32) / 8388608
        elif width == 4:
            values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Unsupported PCM sample width: {width}")
        if values.size % channels:
            raise ValueError("Corrupt WAV frame data")
        return values.reshape(-1, channels)

    @staticmethod
    def _bpm(x, sr):
        if len(x) < sr * 2:
            return None, 0.0
        step = max(128, int(sr * .02))
        count = len(x) // step
        energy = np.mean(x[:count * step].reshape(count, step) ** 2, axis=1)
        flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
        flux -= np.median(flux)
        min_lag, max_lag = int(60 / 180 / .02), min(len(flux) - 1, int(60 / 60 / .02))
        if max_lag <= min_lag:
            return None, 0.0
        autocorr = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
        scores = autocorr[min_lag:max_lag + 1]
        lag = int(np.argmax(scores)) + min_lag
        bpm = 60 / (lag * .02)
        return round(bpm, 2), min(1.0, max(0.0, (float(scores.max()) / (float(np.mean(scores)) + 1e-12) - 1) / 4))

    @staticmethod
    def _pitch(x, sr):
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / sr)
        mask = (freqs >= 50) & (freqs <= 2000)
        if not np.any(mask):
            return None, 0.0
        idx = int(np.argmax(spectrum[mask]))
        hz = float(freqs[mask][idx])
        peak = float(spectrum[mask][idx])
        mean = float(np.mean(spectrum[mask])) + 1e-12
        return round(hz, 2), min(1.0, max(0.0, (peak / mean - 1) / 20))

    @staticmethod
    def _key(x, sr):
        if len(x) < 4096:
            return None, None, 0.0
        n = min(len(x), sr * 12)
        spectrum = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        chroma = np.zeros(12)
        mask = (freqs >= 55) & (freqs <= 1760)
        for hz, magnitude in zip(freqs[mask], spectrum[mask]):
            chroma[int(round(69 + 12 * math.log2(float(hz) / 440))) % 12] += float(magnitude)
        if not chroma.any():
            return None, None, 0.0
        chroma /= chroma.sum()
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        profiles = (
            (np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]), "major"),
            (np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]), "minor"),
        )
        scored = [
            (float(np.dot(chroma, np.roll(profile, shift))), names[shift], mode)
            for profile, mode in profiles
            for shift in range(12)
        ]
        scored.sort(reverse=True)
        return scored[0][1], scored[0][2], min(1.0, max(0.0, (scored[0][0] - scored[1][0]) * 8))

    @staticmethod
    def _onset_beats(x, sr, bpm):
        """Return conservative onset positions expressed in musical beats."""
        if bpm is None or len(x) < max(512, int(sr * .05)):
            return ()
        hop = max(64, int(sr * .01))
        frame = max(hop * 4, int(sr * .04))
        if len(x) < frame:
            return ()
        count = 1 + (len(x) - frame) // hop
        window = np.hanning(frame)
        energies = np.empty(count, dtype=np.float64)
        for i in range(count):
            start = i * hop
            energies[i] = np.sqrt(np.mean((x[start:start + frame] * window) ** 2))
        flux = np.maximum(0.0, np.diff(energies, prepend=energies[0]))
        baseline = np.median(flux)
        spread = np.median(np.abs(flux - baseline)) + 1e-9
        threshold = baseline + max(spread * 2.5, float(np.max(flux)) * .08)
        candidates = np.flatnonzero(flux > threshold)
        if not len(candidates):
            return ()
        min_gap = max(1, int(sr * .08 / hop))
        selected = []
        for idx in candidates:
            if not selected or idx - selected[-1] >= min_gap:
                selected.append(int(idx))
            elif flux[idx] > flux[selected[-1]]:
                selected[-1] = int(idx)
        beat_seconds = 60.0 / bpm
        beats = tuple(round((idx * hop / sr) / beat_seconds, 6) for idx in selected)
        return beats

    @staticmethod
    def _frame_pitch(frame, sr):
        """Estimate a monophonic MIDI pitch from one voiced frame."""
        frame = frame - float(np.mean(frame))
        if np.sqrt(np.mean(frame * frame)) < 1e-4:
            return None, 0.0
        n = len(frame)
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        mask = (freqs >= 50) & (freqs <= 1200)
        if not np.any(mask):
            return None, 0.0
        peak_idx = int(np.argmax(spectrum[mask]))
        peak_hz = float(freqs[mask][peak_idx])
        if peak_hz <= 0:
            return None, 0.0
        # Prefer the spectral fundamental, but reject weak/noisy frames.
        peak = float(spectrum[mask][peak_idx])
        median = float(np.median(spectrum[mask])) + 1e-9
        confidence = min(1.0, max(0.0, (peak / median - 1.0) / 18.0))
        if confidence < .08:
            return None, confidence
        midi = int(round(69 + 12 * math.log2(peak_hz / 440.0)))
        if not 24 <= midi <= 108:
            return None, confidence
        return midi, confidence

    @classmethod
    def _note_events(cls, x, sr, bpm, role):
        """Infer conservative monophonic note events for tonal/lead sources."""
        if bpm is None or role == "drums" or len(x) < int(sr * .08):
            return (), ()
        frame = max(1024, int(sr * .046))
        hop = max(256, int(sr * .0116))
        if len(x) < frame:
            return ()
        notes = []
        beat_seconds = 60.0 / bpm
        for start in range(0, len(x) - frame + 1, hop):
            pitch, confidence = cls._frame_pitch(x[start:start + frame], sr)
            notes.append((start, pitch if confidence >= .12 else None))
        events = []
        active_pitch = None
        active_start = None
        last_pitch = None
        for start, pitch in notes:
            if pitch != active_pitch:
                if active_pitch is not None and active_start is not None:
                    duration = max(hop, start - active_start) / sr / beat_seconds
                    if duration >= .08:
                        events.append((active_pitch, duration))
                active_pitch = pitch
                active_start = start if pitch is not None else None
            elif pitch is not None and active_start is None:
                active_start = start
            last_pitch = pitch
        if active_pitch is not None and active_start is not None:
            duration = max(hop, len(x) - active_start) / sr / beat_seconds
            if duration >= .08:
                events.append((active_pitch, duration))
        # Merge tiny pitch flickers into the preceding event and cap runaway silence.
        if not events:
            return (), ()
        pitches = tuple(int(p) for p, _ in events)
        durations = tuple(round(float(d), 6) for _, d in events)
        return pitches, durations

    @staticmethod
    def _envelope(x, sr):
        step = max(64, int(sr * .01))
        count = len(x) // step
        if count < 4:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        envelope = np.sqrt(np.mean(x[:count * step].reshape(count, step) ** 2, axis=1))
        peak_idx = int(np.argmax(envelope))
        peak_value = float(envelope[peak_idx]) + 1e-12
        threshold = max(peak_value * .1, float(np.median(envelope)) + 1e-6)
        active = np.flatnonzero(envelope > threshold)
        if len(active) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        onset_rate = float(np.count_nonzero((envelope[1:] > threshold) & (envelope[1:] > envelope[:-1])) / (len(x) / sr))
        attack = peak_idx * step / sr
        active_end = int(active[-1])
        decay = max(0.0, (active_end - peak_idx) * step / sr)
        sustain = float(np.median(envelope[active[min(len(active)-1, len(active)//2):]]) / peak_value)
        release = max(0.0, (len(envelope) - active_end - 1) * step / sr)
        transient_ratio = min(1.0, max(0.0, (peak_value - float(np.median(envelope))) / peak_value))
        return onset_rate, transient_ratio, attack, decay, sustain, release
