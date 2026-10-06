"""Source audio renderer: preserve source, optionally tempo-adapt with OLA."""
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
    def render(self,request:RenderRequest)->RenderResult:
        if request.kind!="audio": raise ValueError("SourceAudioGenerator only renders audio.")
        if not request.source_asset_ids: raise ValueError("source_asset_ids required")
        sid=request.source_asset_ids[0]; path=Path(sid)
        if not path.exists(): raise FileNotFoundError(path)
        data,sr=self._read(path); auth=request.parameter_changes.get("authority_analysis",{})
        meta=auth.get(sid,{}) if isinstance(auth,dict) else {}
        sbpm,tbpm=meta.get("bpm"),request.parameter_changes.get("bpm"); rate=1.0
        if isinstance(sbpm,(int,float)) and isinstance(tbpm,(int,float)) and sbpm>0 and tbpm>0:
            rate=float(tbpm)/float(sbpm)
            if .5<=rate<=2: data=self._stretch(data,rate)
        key=build_cache_key(request.candidate_id,request.context_version,request.kind,request.parameter_changes,request.source_asset_ids)
        out=self.cache_dir/f"{key}.wav"
        if not out.exists(): self._write(out,data,sr)
        return RenderResult(candidate_id=request.candidate_id,kind="audio",artifact_ref=str(out),cache_key=key,
            duration_seconds=len(data)/sr,sample_rate=sr,metadata={"renderer":"source-audio-v0.1","source_asset_id":sid,
            "source_bpm":sbpm,"target_bpm":tbpm,"tempo_adapted":abs(rate-1)>1e-6,"pitch_shifted":False})
