"""Deterministic PCM WAV feature extraction for Hallah V0."""
from __future__ import annotations
import math, wave
from pathlib import Path
import numpy as np
from core.analysis.contracts import AudioAnalysis

class WavAnalyzer:
    def analyze(self, asset_id: str) -> AudioAnalysis:
        path=Path(asset_id)
        if path.suffix.lower()!=".wav": raise ValueError("WavAnalyzer accepts WAV files only")
        with wave.open(str(path),"rb") as wf:
            channels,sr,width,frames=wf.getnchannels(),wf.getframerate(),wf.getsampwidth(),wf.getnframes()
            raw=wf.readframes(frames)
        x=self._decode(raw,width,channels)
        mono=x.mean(axis=1) if channels>1 else x[:,0]
        if not len(mono): raise ValueError("WAV contains no samples")
        n=min(len(mono),max(sr*8,4096)); seg=mono[:n]; power=np.abs(np.fft.rfft(seg*np.hanning(n)))**2
        freqs=np.fft.rfftfreq(n,1/sr); total=float(power.sum())+1e-12
        centroid=float((freqs*power).sum()/total)
        rolloff=float(freqs[min(int(np.searchsorted(np.cumsum(power),total*.85)),len(freqs)-1)])
        band=lambda a,b: float(power[(freqs>=a)&(freqs<b)].sum()/total)
        rms=float(np.sqrt(np.mean(mono*mono))); peak=float(np.max(np.abs(mono)))
        zcr=float(np.mean(mono[:-1]*mono[1:]<0)) if len(mono)>1 else 0.0
        bpm,bpm_conf=self._bpm(mono,sr); key,scale,key_conf=self._key(mono,sr)
        width_value,corr=0.0,1.0
        if channels>=2:
            l,r=x[:,0],x[:,1]
            corr=float(np.corrcoef(l,r)[0,1]) if np.std(l) and np.std(r) else 1.0
            width_value=min(1.0,float(np.mean(((l-r)*.5)**2)/(np.mean(((l+r)*.5)**2)+1e-12)))
        step=max(256,int(sr*.02)); k=len(mono)//step; onsets=0.0
        if k>2:
            e=np.mean(mono[:k*step].reshape(k,step)**2,axis=1); threshold=float(np.median(e)*2)
            onsets=float(np.count_nonzero((e[1:]>threshold)&(e[1:]>e[:-1]))/(len(mono)/sr))
        metrics={"centroid":centroid,"low":band(20,250),"mid":band(250,4000),"high":band(4000,20000),"onsets":onsets,"zcr":zcr}
        role="bass" if metrics["low"]>.48 and centroid<700 else "drums" if onsets>2.5 and zcr>.08 else "vocal/lead" if centroid>2200 and zcr>.06 else "guitar/piano" if metrics["mid"]>metrics["low"] and metrics["mid"]>metrics["high"] else "texture"
        return AudioAnalysis(asset_id=str(path),sample_rate=sr,duration_seconds=len(mono)/sr,channels=channels,bpm=bpm,key=key,scale=scale,role=role,rms=rms,peak=peak,zero_crossing_rate=zcr,spectral_centroid_hz=centroid,spectral_rolloff_hz=rolloff,low_energy_ratio=metrics["low"],mid_energy_ratio=metrics["mid"],high_energy_ratio=metrics["high"],stereo_width=width_value,stereo_correlation=corr,onset_rate=onsets,confidence={"bpm":bpm_conf,"key":key_conf,"role":.45,"spectral":.95})
    @staticmethod
    def _decode(raw,width,channels):
        if width==1: v=(np.frombuffer(raw,dtype=np.uint8).astype(np.float32)-128)/128
        elif width==2: v=np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768
        elif width==3:
            d=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3); q=d[:,0].astype(np.int32)|(d[:,1].astype(np.int32)<<8)|(d[:,2].astype(np.int32)<<16); q=np.where(q&0x800000,q-0x1000000,q); v=q.astype(np.float32)/8388608
        elif width==4: v=np.frombuffer(raw,dtype="<i4").astype(np.float32)/2147483648
        else: raise ValueError(f"Unsupported PCM sample width: {width}")
        if v.size%channels: raise ValueError("Corrupt WAV frame data")
        return v.reshape(-1,channels)
    @staticmethod
    def _bpm(x,sr):
        if len(x)<sr*2:return None,0.0
        step=max(128,int(sr*.02)); k=len(x)//step; e=np.mean(x[:k*step].reshape(k,step)**2,axis=1); flux=np.maximum(0,np.diff(e,prepend=e[0])); flux-=np.median(flux)
        lo,hi=int(60/180/.02),min(len(flux)-1,int(60/60/.02))
        if hi<=lo:return None,0.0
        ac=np.correlate(flux,flux,mode="full")[len(flux)-1:]; s=ac[lo:hi+1]; lag=int(np.argmax(s))+lo; bpm=60/(lag*.02)
        return round(bpm,2),min(1.0,max(0.0,(float(s.max())/(float(np.mean(s))+1e-12)-1)/4))
    @staticmethod
    def _key(x,sr):
        if len(x)<4096:return None,None,0.0
        n=min(len(x),sr*12); spec=np.abs(np.fft.rfft(x[:n]*np.hanning(n))); f=np.fft.rfftfreq(n,1/sr); chroma=np.zeros(12); mask=(f>=55)&(f<=1760)
        for hz,mag in zip(f[mask],spec[mask]): chroma[int(round(69+12*math.log2(float(hz)/440)))%12]+=float(mag)
        if not chroma.any():return None,None,0.0
        chroma/=chroma.sum(); names=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
        profiles=((np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]),"major"),(np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]),"minor"))
        scores=sorted((float(np.dot(chroma,np.roll(p,r))),names[r],s) for p,s in profiles for r in range(12),reverse=True)
        return scores[0][1],scores[0][2],min(1.0,max(0.0,(scores[0][0]-scores[1][0])*8))
