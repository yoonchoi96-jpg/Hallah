"""Source audio renderer with authority-aware adaptation."""
from __future__ import annotations
import math
import wave
from pathlib import Path
import numpy as np
from audio.cache.keys import build_cache_key
from audio.rendering.contracts import RenderRequest, RenderResult

class SourceAudioGenerator:
    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _authority_dimension(meta: dict[str, object]) -> str:
        return str(meta.get("dimension", ""))
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
        hin=max(256,frame//4); hout=max(256,int(round(hin/rate))); win=np.hanning(frame)
        positions=list(range(0,max(1,n-frame+1),hin)); outlen=max(frame,(len(positions)-1)*hout+frame)
        out=np.zeros((outlen,x.shape[1])); norm=np.zeros(outlen)
        for i,start in enumerate(positions):
            chunk=x[start:start+frame]
            if len(chunk)<frame: chunk=np.pad(chunk,((0,frame-len(chunk)),(0,0)))
            s=i*hout; out[s:s+frame]+=chunk*win[:,None]; norm[s:s+frame]+=win
        valid=norm>1e-8; out[valid]/=norm[valid,None]; out[~valid]=0
        target=max(1,int(round(n/rate)))
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
        new_len = max(2, int(round(len(data) / factor)))
        positions = np.linspace(0.0, len(data) - 1.0, new_len)
        base = np.arange(len(data), dtype=np.float64)
        resampled = np.empty((new_len, data.shape[1]), dtype=np.float32)
        for channel in range(data.shape[1]):
            resampled[:, channel] = np.interp(positions, base, data[:, channel]).astype(np.float32)
        return SourceAudioGenerator._stretch(resampled, 1.0 / factor)

    @staticmethod
    def _key_pc(key: object) -> int | None:
        if not isinstance(key, str):
            return None
        names = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
                 "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
        return names.get(key.strip())

    def render(self,request:RenderRequest)->RenderResult:
        if request.kind!="audio": raise ValueError("SourceAudioGenerator only renders audio.")
        if not request.source_asset_ids: raise ValueError("source_asset_ids required")
        sid=request.source_asset_ids[0]; path=Path(sid)  # primary source
        if not path.exists(): raise FileNotFoundError(path)
        data,sr=self._read(path); auth=request.parameter_changes.get("authority_analysis",{})
        meta=auth.get(sid,{}) if isinstance(auth,dict) else {}
        sbpm,tbpm=meta.get("bpm"),request.parameter_changes.get("bpm"); rate=1.0
        source_key, target_key = meta.get("key"), request.parameter_changes.get("key")
        source_pc, target_pc = self._key_pc(source_key), self._key_pc(target_key)
        semitones = 0.0
        if source_pc is not None and target_pc is not None:
            delta = target_pc - source_pc
            semitones = float(((delta + 6) % 12) - 6)
            if abs(semitones) > 1e-6:
                data = self._pitch_shift(data, semitones)
        if isinstance(sbpm,(int,float)) and isinstance(tbpm,(int,float)) and sbpm>0 and tbpm>0:
            rate=float(tbpm)/float(sbpm)
            if .5<=rate<=2: data=self._stretch(data,rate)
        key=build_cache_key(request.candidate_id,request.context_version,request.kind,request.parameter_changes,request.source_asset_ids)
        out=self.cache_dir/f"{key}.wav"
        if not out.exists(): self._write(out,data,sr)
        return RenderResult(candidate_id=request.candidate_id,kind="audio",artifact_ref=str(out),cache_key=key,
            duration_seconds=len(data)/sr,sample_rate=sr,metadata={"renderer":"source-audio-v0.1","source_asset_id":sid,
            "source_bpm":sbpm,"target_bpm":tbpm,"tempo_adapted":abs(rate-1)>1e-6,\n            "source_key":source_key,"target_key":target_key,"pitch_shift_semitones":semitones,\n            "pitch_shifted":abs(semitones)>1e-6})
