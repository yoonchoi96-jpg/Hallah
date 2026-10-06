import wave
import numpy as np
from core.analysis.wav import WavAnalyzer

def test_wav_analyzer_extracts_features(tmp_path):
    sr=8000; t=np.arange(sr*4)/sr
    x=np.zeros_like(t)
    for start in range(0,len(x),sr//2):
        x[start:start+sr//20]=.8*np.sin(2*np.pi*100*t[start:start+sr//20])
    x+=.15*np.sin(2*np.pi*440*t)
    path=tmp_path/"fixture.wav"
    with wave.open(str(path),"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(x,-1,1)*32767).astype("<i2").tobytes())
    result=WavAnalyzer().analyze(str(path))
    assert result.sample_rate==sr and result.channels==1
    assert 3.9<result.duration_seconds<4.1
    assert result.rms and result.peak and result.spectral_centroid_hz
    assert result.bpm is not None
    assert result.role in {"bass","drums","vocal/lead","guitar/piano","texture"}
