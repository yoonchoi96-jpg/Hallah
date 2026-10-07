"""Source audio renderer with authority-aware adaptation."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from audio.cache.keys import build_cache_key
from audio.processing.masking import build_mix_gain_plan
from audio.rendering.contracts import RenderRequest, RenderResult


class SourceAudioGenerator:
    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _authority_dimension(meta: dict[str, object]) -> str:
        return str(meta.get("dimension", ""))

    @staticmethod
    def _resolve_asset_path(asset_id: str, changes: dict[str, object]) -> Path:
        """Resolve a logical asset ID to its filesystem path when provided."""
        paths = changes.get("asset_paths")
        if isinstance(paths, dict):
            value = paths.get(asset_id)
            if isinstance(value, str) and value:
                return Path(value)
        return Path(asset_id)

    @classmethod
    def _pitch_allowed(cls, sid: str, meta: dict[str, object], changes: dict[str, object]) -> bool:
        overrides=changes.get("source_adaptations")
        if isinstance(overrides,dict):
            source_override=overrides.get(sid)
            if isinstance(source_override,dict):
                if isinstance(source_override.get("pitch_adaptation"),bool):
                    return bool(source_override["pitch_adaptation"])
                if isinstance(source_override.get("pitch"),str):
                    return True
        override=changes.get("pitch_adaptation")
        if isinstance(override,bool):
            return override
        dimension=cls._authority_dimension(meta)
        return dimension in {"harmony", "melody", "low_end", "texture", "arrangement"}
    @staticmethod
    def _read(path: Path):
        with wave.open(str(path),"rb") as w:
            ch,width,sr,frames=w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes()
            if width != 2: raise ValueError("SourceAudioGenerator requires 16-bit PCM WAV.")
            x=np.frombuffer(w.readframes(frames),dtype="<i2").astype(np.float32)/32768.0
        if x.size % ch: raise ValueError("Corrupt WAV frame data.")
        return x.reshape(-1,ch),sr
    @staticmethod
    def _write(path,data,sr):
        path.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(path),"wb") as w:
            w.setnchannels(data.shape[1]); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(data,-1,1)*32767).astype("<i2").tobytes())
    @staticmethod
    def _stretch(x,rate):
        if rate <= 0: raise ValueError("rate must be positive")
        if abs(rate-1)<1e-6 or len(x)<2048: return x.copy()
        n=len(x); frame=min(4096,max(1024,2**int(math.log2(max(1024,min(n,4096))))))
        hin=max(256,frame//4); hout=max(256,round(hin/rate)); win=np.hanning(frame)
        positions=list(range(0,max(1,n-frame+1),hin)); outlen=max(frame,(len(positions)-1)*hout+frame)
        out=np.zeros((outlen,x.shape[1])); norm=np.zeros(outlen)
        for i,start in enumerate(positions):
            chunk=x[start:start+frame]
            if len(chunk)<frame: chunk=np.pad(chunk,((0,frame-len(chunk)),(0,0)))
            s=i*hout; out[s:s+frame]+=chunk*win[:,None]; norm[s:s+frame]+=win
        valid=norm>1e-8; out[valid]/=norm[valid,None]; out[~valid]=0
        target=max(1,round(n/rate))
        return np.pad(out[:target],((0,max(0,target-len(out)))),mode="constant").astype(np.float32)
    @staticmethod
    def _pitch_shift(data: np.ndarray, semitones: float) -> np.ndarray:
        """Pitch shift while approximately preserving duration.

        V0 uses deterministic resampling followed by OLA duration restoration.
        It is intentionally isolated so a higher-quality phase-vocoder backend
        can replace it without changing the render contract.
        """
        if abs(semitones) < 1e-6 or len(data) < 2048:
            return data.copy()
        factor = 2.0 ** (semitones / 12.0)
        new_len = max(2, round(len(data) / factor))
        positions = np.linspace(0.0, len(data) - 1.0, new_len)
        base = np.arange(len(data), dtype=np.float64)
        resampled = np.empty((new_len, data.shape[1]), dtype=np.float32)
        for channel in range(data.shape[1]):
            resampled[:, channel] = np.interp(positions, base, data[:, channel]).astype(np.float32)
        return SourceAudioGenerator._stretch(resampled, 1.0 / factor)

    @staticmethod
    def _candidate_gain(changes: dict[str, object]) -> float:
        """Deterministic candidate-specific character for V0 auditioning."""
        direction = str(changes.get("direction", "identity"))
        return {"identity": 1.0, "natural": 0.89, "bold": 1.12, "experimental": 0.78}.get(direction, 1.0)

    @classmethod
    def _apply_candidate_character(cls, data: np.ndarray, changes: dict[str, object]) -> tuple[np.ndarray, float]:
        gain = cls._candidate_gain(changes)
        rendered = data.astype(np.float32, copy=True) * gain
        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.98:
            rendered *= 0.98 / peak
        return rendered, gain

    @staticmethod
    def _candidate_tone(data: np.ndarray, sr: int, changes: dict[str, object]) -> tuple[np.ndarray, dict[str, float]]:
        """Small deterministic tone/dynamics signatures for V0.2 auditioning."""
        direction = str(changes.get("direction", "identity"))
        settings = {
            "identity": (0.0, 0.0, 1.0),
            "natural": (0.92, 1.06, 0.97),
            "bold": (1.10, 0.82, 1.08),
            "experimental": (0.72, 1.22, 0.90),
        }.get(direction, (0.0, 0.0, 1.0))
        low_boost, high_boost, dynamics = settings
        rendered = data.astype(np.float32, copy=True)
        if direction == "identity":
            return rendered, {"low_factor": 1.0, "high_factor": 1.0, "dynamics": 1.0}
        mono = np.mean(rendered, axis=1)
        n = len(mono)
        spectrum = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(n, 1.0 / sr) if n else np.array([])
        tilt = np.ones_like(freqs)
        if n:
            low = np.exp(-freqs / 220.0)
            high = 1.0 - np.exp(-freqs / 3200.0)
            tilt *= 1.0 + (low_boost - 1.0) * low
            tilt *= 1.0 + (high_boost - 1.0) * high
            transformed = np.fft.irfft(spectrum * tilt, n=n).astype(np.float32)
            delta = transformed - mono
            rendered += delta[:, None]
        if dynamics != 1.0:
            rendered = np.tanh(rendered * dynamics) / np.tanh(dynamics)
        return rendered, {"low_factor": low_boost, "high_factor": high_boost, "dynamics": dynamics}

    @staticmethod
    def _candidate_space_transient(
        data: np.ndarray, sr: int, changes: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Deterministic spatial/transient signatures for V0 auditioning."""
        direction = str(changes.get("direction", "identity"))
        settings = {
            "identity": (1.00, 1.00, 0.00),
            "natural": (1.06, 0.88, 0.00),
            "bold": (1.24, 1.18, 0.00),
            "experimental": (0.72, 1.32, 0.00),
        }.get(direction, (1.00, 1.00, 0.00))
        width, transient, _ = settings
        rendered = data.astype(np.float32, copy=True)

        # Mid/side width shaping. Mono sources remain mono; stereo sources
        # receive deterministic candidate-specific spatial treatment.
        if rendered.shape[1] >= 2 and rendered.shape[0]:
            mid = (rendered[:, 0] + rendered[:, 1]) * 0.5
            side = (rendered[:, 0] - rendered[:, 1]) * 0.5
            side *= width
            rendered[:, 0] = mid + side
            rendered[:, 1] = mid - side

        # A lightweight high-frequency/transient proxy. This is deliberately
        # not a compressor: it only differentiates audition character while
        # leaving the dedicated dynamics engine free for later versions.
        if len(rendered) > 1 and abs(transient - 1.0) > 1e-6:
            hp = np.empty_like(rendered)
            hp[0] = 0.0
            hp[1:] = rendered[1:] - rendered[:-1]
            rendered += hp * (transient - 1.0) * 0.35

        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.98:
            rendered *= 0.98 / peak
        return rendered, {
            "stereo_width_factor": width,
            "transient_factor": transient,
        }

    @staticmethod
    def _candidate_eq_masking(
        data: np.ndarray,
        sr: int,
        changes: dict[str, object],
        source_id: str,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Deterministic spectral shaping with optional masking awareness."""
        direction = str(changes.get("direction", "identity"))
        settings = {
            "identity": (1.00, 1.00, 1.00),
            "natural": (0.96, 1.00, 1.04),
            "bold": (0.90, 1.08, 1.12),
            "experimental": (1.08, 0.86, 1.20),
        }.get(direction, (1.00, 1.00, 1.00))
        low_factor, mid_factor, high_factor = settings
        rendered = data.astype(np.float32, copy=True)
        if direction == "identity" or not len(rendered):
            return rendered, {
                "low_factor": 1.0,
                "mid_factor": 1.0,
                "high_factor": 1.0,
                "masking_reduction": 0.0,
            }

        mono = np.mean(rendered, axis=1)
        n = len(mono)
        spectrum = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)

        gains = np.ones_like(freqs)
        gains[freqs < 180.0] *= low_factor
        gains[(freqs >= 180.0) & (freqs < 2500.0)] *= mid_factor
        gains[freqs >= 2500.0] *= high_factor

        masking_reduction = 0.0
        authority = changes.get("authority_analysis", {})
        source_meta = authority.get(source_id, {}) if isinstance(authority, dict) else {}
        source_low = source_meta.get("low_energy_ratio")
        source_mid = source_meta.get("mid_energy_ratio")
        source_high = source_meta.get("high_energy_ratio")

        # If another analyzed source occupies the same broad spectral region,
        # gently carve the current source there. Prefer explicit relationship-aware
        # priorities when the render pipeline has resolved musical roles.
        relationship_map = changes.get("mix_relationships", {})
        relationship = relationship_map.get(source_id, {}) if isinstance(relationship_map, dict) else {}
        relationship_decisions = relationship.get("decisions", ()) if isinstance(relationship, dict) else ()
        if isinstance(relationship_decisions, (list, tuple)) and relationship_decisions:
            for decision in relationship_decisions:
                if not isinstance(decision, dict):
                    continue
                amount = float(decision.get("allocated_amount", decision.get("amount", 0.0)))
                ranges = decision.get("ranges", {})
                for band in decision.get("bands", ()):
                    lo, hi = (
                        ranges.get(str(band), (0.0, 0.0))
                        if isinstance(ranges, dict)
                        else (0.0, 0.0)
                    )
                    if hi > lo:
                        band_mask = (freqs >= lo) & (freqs < hi)
                        gains[band_mask] *= 1.0 - min(0.24, max(0.0, amount))
                        masking_reduction = max(masking_reduction, min(0.24, max(0.0, amount)))
        elif isinstance(authority, dict) and authority:
            overlap = {"low": 0.0, "mid": 0.0, "high": 0.0}
            for ref_id, ref_meta in authority.items():
                if ref_id == source_id or not isinstance(ref_meta, dict):
                    continue
                for band, key in (
                    ("low", "low_energy_ratio"),
                    ("mid", "mid_energy_ratio"),
                    ("high", "high_energy_ratio"),
                ):
                    value = ref_meta.get(key)
                    if isinstance(value, (int, float)):
                        overlap[band] = max(overlap[band], float(value))
            source_values = {"low": source_low, "mid": source_mid, "high": source_high}
            for band, (lo, hi) in {
                "low": (0.0, 180.0),
                "mid": (180.0, 2500.0),
                "high": (2500.0, float(sr) * 0.5),
            }.items():
                own = source_values[band]
                other = overlap[band]
                if isinstance(own, (int, float)) and float(own) > 0.0 and other > 0.20:
                    amount = min(0.18, float(other) * 0.24)
                    if hi > lo:
                        band_mask = (freqs >= lo) & (freqs < hi)
                        gains[band_mask] *= 1.0 - amount
                        masking_reduction = max(masking_reduction, amount)

        transformed = np.fft.irfft(spectrum * gains, n=n).astype(np.float32)
        delta = transformed - mono
        rendered += delta[:, None]
        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.98:
            rendered *= 0.98 / peak
        return rendered, {
            "low_factor": low_factor,
            "mid_factor": mid_factor,
            "high_factor": high_factor,
            "masking_reduction": masking_reduction,
        }

    @staticmethod
    def _source_override(sid: str, changes: dict[str, object]) -> dict[str, object]:
        overrides = changes.get("source_adaptations")
        if isinstance(overrides, dict):
            value = overrides.get(sid)
            if isinstance(value, dict):
                return value
        return {}

    @classmethod
    def _reference_meta(cls, sid: str, auth: object, changes: dict[str, object]) -> dict[str, object]:
        meta = auth.get(sid, {}) if isinstance(auth, dict) else {}
        override = cls._source_override(sid, changes)
        reference = override.get("pitch") or override.get("bpm") or override.get("tempo")
        if isinstance(reference, str) and isinstance(auth, dict):
            candidate = auth.get(reference)
            if isinstance(candidate, dict):
                resolved = dict(candidate)
                resolved["_source_id"] = reference
                return resolved
        return meta

    @staticmethod
    def _key_pc(key: object) -> int | None:
        if not isinstance(key, str):
            return None
        names = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
                 "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
        return names.get(key.strip())

    def render_stems(self, request: RenderRequest) -> tuple[RenderResult, ...]:
        """Render every source asset independently as a non-destructive stem set."""
        if request.kind != "audio":
            raise ValueError("SourceAudioGenerator only renders audio.")
        if not request.source_asset_ids:
            raise ValueError("source_asset_ids required")
        results = []
        for sid in request.source_asset_ids:
            stem_request = RenderRequest(
                candidate_id=request.candidate_id,
                context_version=request.context_version,
                kind=request.kind,
                intent=request.intent,
                source_asset_ids=(sid,),
                parameter_changes=request.parameter_changes,
                output_format=request.output_format,
            )
            results.append(self.render(stem_request))
        return tuple(results)

    def render_mix(self, request: RenderRequest) -> RenderResult:
        """Render all source assets to independent stems, then create a cached stereo-compatible mix."""
        stems = self.render_stems(request)
        if len(stems) == 1:
            return stems[0]
        arrays = []
        sample_rate = None
        for stem in stems:
            data, sr = self._read(Path(stem.artifact_ref))
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError("All source stems must use the same sample rate.")
            arrays.append(data)
        channels = max(data.shape[1] for data in arrays)
        total = max(data.shape[0] for data in arrays)
        mix = np.zeros((total, channels), dtype=np.float32)
        for data in arrays:
            if data.shape[1] == 1 and channels == 2:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] != channels:
                raise ValueError("Incompatible source channel layouts.")
            mix[:data.shape[0], :data.shape[1]] += data
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.92:
            mix *= 0.92 / peak
        key = build_cache_key(
            request.candidate_id,
            request.context_version,
            "audio-mix",
            request.parameter_changes,
            request.source_asset_ids,
        )
        out = self.cache_dir / f"{key}.wav"
        if not out.exists():
            self._write(out, mix, sample_rate or 44100)
        return RenderResult(
            candidate_id=request.candidate_id,
            kind="audio",
            artifact_ref=str(out),
            cache_key=key,
            duration_seconds=len(mix) / float(sample_rate or 44100),
            sample_rate=sample_rate or 44100,
            metadata={
                "renderer": "source-audio-mix-v0.1",
                "stem_count": len(stems),
                "stem_refs": tuple(stem.artifact_ref for stem in stems),
            },
        )

    def render(self,request:RenderRequest)->RenderResult:
        if request.kind!="audio": raise ValueError("SourceAudioGenerator only renders audio.")
        if not request.source_asset_ids: raise ValueError("source_asset_ids required")
        sid=request.source_asset_ids[0]
        path=self._resolve_asset_path(sid,request.parameter_changes)
        if not path.exists(): raise FileNotFoundError(path)
        data,sr=self._read(path)
        auth=request.parameter_changes.get("authority_analysis",{})
        meta=auth.get(sid,{}) if isinstance(auth,dict) else {}
        override=self._source_override(sid,request.parameter_changes)
        reference_meta=self._reference_meta(sid,auth,request.parameter_changes)
        sbpm=meta.get("bpm")
        tbpm=(reference_meta.get("bpm")
              if override.get("bpm") or override.get("tempo")
              else request.parameter_changes.get("bpm"))
        rate=1.0
        source_key=meta.get("key")
        target_key=(reference_meta.get("key")
                    if override.get("pitch")
                    else request.parameter_changes.get("key"))
        pitch_allowed=self._pitch_allowed(sid,meta,request.parameter_changes)
        source_pc,target_pc=self._key_pc(source_key),self._key_pc(target_key)
        semitones=0.0
        if pitch_allowed and source_pc is not None and target_pc is not None:
            delta=target_pc-source_pc
            semitones=float(((delta+6)%12)-6)
            if abs(semitones)>1e-6:
                data=self._pitch_shift(data,semitones)
        if isinstance(sbpm,(int,float)) and isinstance(tbpm,(int,float)) and sbpm>0 and tbpm>0:
            rate=float(tbpm)/float(sbpm)
            if .5<=rate<=2:
                data=self._stretch(data,rate)
        data, candidate_gain = self._apply_candidate_character(data, request.parameter_changes)
        data, tone_settings = self._candidate_tone(data, sr, request.parameter_changes)
        data, space_transient = self._candidate_space_transient(data, sr, request.parameter_changes)
        data, eq_masking = self._candidate_eq_masking(data, sr, request.parameter_changes, sid)
        dynamic_masking = {"applied": False, "amount": 0.0, "events": 0}
        relationship_map = request.parameter_changes.get("mix_relationships", {})
        relationship = relationship_map.get(sid, {}) if isinstance(relationship_map, dict) else {}
        decisions = relationship.get("decisions", ()) if isinstance(relationship, dict) else ()
        mix_plan = build_mix_gain_plan(auth if isinstance(auth, dict) else {}, relationship_map) if isinstance(relationship_map, dict) else {}
        decisions = mix_plan.get(sid, decisions)
        for decision in decisions if isinstance(decisions, (list, tuple)) else ():
            if not isinstance(decision, dict):
                continue
            ref_id = decision.get("reference_id")
            if not isinstance(ref_id, str):
                continue
            ref_meta = auth.get(ref_id) if isinstance(auth, dict) else None
            if not isinstance(ref_meta, dict):
                continue
            ref_meta = dict(ref_meta)
            ref_meta["_source_id"] = ref_id
            amount = float(decision.get("allocated_amount", decision.get("amount", 0.0)))
            ranges = decision.get("ranges", {})
            bands = tuple(str(b) for b in decision.get("bands", ()))
            from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
            ref_path = self._resolve_asset_path(ref_id, request.parameter_changes)
            reference_audio = None
            if ref_path.exists():
                ref_audio, ref_sr = self._read(ref_path)
                if ref_sr == sr:
                    reference_audio = ref_audio
            data, dynamic_masking = apply_spectral_curve_dynamic_masking(
                data, sr, meta, ref_meta, amount, bands=bands,
                ranges=ranges if isinstance(ranges, dict) else None,
                reference_data=reference_audio,
            )
            if dynamic_masking.get("applied"):
                dynamic_masking["allocated_amount"] = amount
                dynamic_masking["budget_remaining"] = decision.get("budget_remaining")
                break
        key=build_cache_key(request.candidate_id,request.context_version,request.kind,request.parameter_changes,request.source_asset_ids)
        out=self.cache_dir/f"{key}.wav"
        if not out.exists(): self._write(out,data,sr)
        return RenderResult(
            candidate_id=request.candidate_id,
            kind="audio",
            artifact_ref=str(out),
            cache_key=key,
            duration_seconds=len(data)/sr,
            sample_rate=sr,
            metadata={
                "renderer":"source-audio-v0.1",
                "source_asset_id":sid,
                "source_bpm":sbpm,
                "target_bpm":tbpm,
                "tempo_adapted":abs(rate-1)>1e-6,
                "source_key":source_key,
                "target_key":target_key,
                "pitch_shift_semitones":semitones,
                "pitch_shifted":abs(semitones)>1e-6,
                "adaptation_reference":reference_meta.get("_source_id"),
                "candidate_direction":str(request.parameter_changes.get("direction", "identity")),
                "candidate_gain":candidate_gain,
                "candidate_tone":tone_settings,
                "candidate_space_transient":space_transient,
                "candidate_eq_masking":eq_masking,
                "dynamic_masking":dynamic_masking,
            },
        )
