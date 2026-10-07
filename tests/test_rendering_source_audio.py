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

def test_source_audio_pitch_adaptation_preserves_duration(tmp_path):
    src=tmp_path/"source.wav"; _tone(src,seconds=3.0)
    req=build_render_request("source-v3",1,"fit key",{
        "key":"D","authority_analysis":{str(src):{"key":"C"}}
    },source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["pitch_shifted"] is True
    assert result.metadata["pitch_shift_semitones"] == 2.0
    assert abs(result.duration_seconds-3.0) < .08

def test_source_audio_same_key_does_not_shift(tmp_path):
    src=tmp_path/"source.wav"; _tone(src,seconds=3.0)
    req=build_render_request("source-v4",1,"keep key",{
        "key":"C","authority_analysis":{str(src):{"key":"C"}}
    },source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["pitch_shifted"] is False
    assert result.metadata["pitch_shift_semitones"] == 0.0


def test_source_audio_source_specific_tempo_reference(tmp_path):
    src=tmp_path/"drums.wav"
    ref=tmp_path/"guitar.wav"
    _tone(src,seconds=4.0)
    _tone(ref,seconds=2.0)
    req=build_render_request("source-v5",1,"follow guitar bpm",{
        "source_adaptations":{str(src):{"bpm":str(ref)}},
        "authority_analysis":{
            str(src):{"bpm":120.0},
            str(ref):{"bpm":90.0},
        },
    },source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["target_bpm"] == 90.0
    assert result.metadata["tempo_adapted"] is True


def test_source_audio_rhythm_source_can_explicitly_pitch_shift(tmp_path):
    src=tmp_path/"drums.wav"
    ref=tmp_path/"guitar.wav"
    _tone(src,seconds=3.0)
    _tone(ref,seconds=3.0)
    req=build_render_request("source-v6",1,"pitch drums to guitar",{
        "source_adaptations":{str(src):{"pitch":str(ref)}},
        "authority_analysis":{
            str(src):{"bpm":120.0,"key":"C","dimension":"rhythm"},
            str(ref):{"bpm":120.0,"key":"D","dimension":"harmony"},
        },
    },source_asset_ids=(str(src),))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["pitch_shifted"] is True
    assert result.metadata["pitch_shift_semitones"] == 2.0
    assert result.metadata["adaptation_reference"] == str(ref)

def test_source_audio_mix_builds_from_independent_stems(tmp_path):
    first=tmp_path/"first.wav"
    second=tmp_path/"second.wav"
    _tone(first,seconds=2.0)
    _tone(second,seconds=3.0)
    req=build_render_request("source-mix-v1",1,"mix sources",{
        "authority_analysis":{
            str(first):{"bpm":120.0,"dimension":"rhythm"},
            str(second):{"bpm":120.0,"dimension":"harmony"},
        },
    },source_asset_ids=(str(first),str(second)))
    result=SourceAudioGenerator(tmp_path/"cache").render_mix(req)
    assert result.metadata["stem_count"] == 2
    assert len(result.metadata["stem_refs"]) == 2
    assert result.duration_seconds == 3.0


def test_source_audio_candidate_directions_have_distinct_gain_character(tmp_path):
    src=tmp_path/"source.wav"
    _tone(src,seconds=2.0)
    generator=SourceAudioGenerator(tmp_path/"cache")
    peaks={}
    for direction in ("identity","natural","bold","experimental"):
        req=build_render_request(
            f"direction-{direction}",1,"candidate audition",{"direction":direction},
            source_asset_ids=(str(src),),
        )
        result=generator.render(req)
        data,_=generator._read(Path(result.artifact_ref))
        peaks[direction]=float(np.max(np.abs(data)))
        assert result.metadata["candidate_direction"] == direction
    assert len(set(round(value, 4) for value in peaks.values())) == 4


def test_source_audio_resolves_logical_asset_ids(tmp_path):
    src=tmp_path/"source.wav"
    _tone(src,seconds=2.0)
    req=build_render_request("logical-v1",1,"logical asset",{
        "asset_paths":{"drums-main":str(src)},
        "authority_analysis":{"drums-main":{"bpm":120.0,"key":"C","dimension":"rhythm"}},
    },source_asset_ids=("drums-main",))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.sample_rate==22050
    assert result.metadata["source_asset_id"]=="drums-main"
    assert result.duration_seconds==2.0


def test_source_audio_resolves_logical_reference_ids_for_masking(tmp_path):
    src=tmp_path/"source.wav"
    ref=tmp_path/"reference.wav"
    _tone(src,seconds=2.0)
    _tone(ref,seconds=2.0)
    req=build_render_request("logical-v2",1,"logical reference",{
        "asset_paths":{"bass-main":str(src),"guitar-main":str(ref)},
        "authority_analysis":{
            "bass-main":{
                "bpm":120.0,
                "key":"C",
                "dimension":"low_end",
                "low_energy_ratio":0.8,
                "mid_energy_ratio":0.3,
                "high_energy_ratio":0.1,
            },
            "guitar-main":{
                "bpm":120.0,
                "key":"C",
                "dimension":"harmony",
                "low_energy_ratio":0.5,
                "mid_energy_ratio":0.6,
                "high_energy_ratio":0.2,
            },
        },
        "mix_relationships":{
            "bass-main":{
                "decisions":({
                    "reference_id":"guitar-main",
                    "bands":("low",),
                    "ranges":{"low":(100.0,400.0)},
                    "amount":0.12,
                },)
            }
        },
    },source_asset_ids=("bass-main",))
    result=SourceAudioGenerator(tmp_path/"cache").render(req)
    assert result.metadata["dynamic_masking"]["applied"] is True
    assert result.metadata["dynamic_masking"]["reference_id"]=="guitar-main"
