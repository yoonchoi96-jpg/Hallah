import math
import wave
import numpy as np
from audio.rendering.contracts import build_render_request
from audio.rendering.source import SourceAudioGenerator

def _tone(path, seconds=4.0, sr=22050):
    t=np.arange(int(sr*seconds))/sr
    x=.25*np.sin(2*math.pi*220*t)
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x*32767).astype("<i2").tobytes())

def test_source_audio_passthrough(tmp_path):
    src=tmp_path/"source.wav"; _tone(src)
    req=build_render_request("source-v1",1,"keep",{},source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.sample_rate==22050
    assert result.metadata["tempo_adapted"] is False
    assert result.metadata["pitch_shifted"] is False
    assert result.duration_seconds==4.0

def test_source_audio_tempo_adaptation(tmp_path):
    src=tmp_path/"source.wav"; _tone(src)
    req=build_render_request("source-v2",1,"fit",{"bpm":90,"authority_analysis":{str(src):{"bpm":120.0}}},
        source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["tempo_adapted"] is True
    assert 4.5 < result.duration_seconds < 5.5
