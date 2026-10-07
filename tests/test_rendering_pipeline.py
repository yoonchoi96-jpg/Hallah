from core.candidates.engine import build_candidates
from core.music_context.models import SongContext
from audio.cache.keys import build_cache_key
from audio.rendering.contracts import build_render_request
from audio.rendering.mock import MockAudioGenerator
from audio.rendering.pipeline import candidate_to_render_request


def test_candidate_becomes_render_request():
    context = SongContext(version=3)
    candidate = build_candidates(context, "make it darker")[0]
    request = candidate_to_render_request(candidate, context)
    assert request.candidate_id == candidate.id
    assert request.context_version == 3
    assert request.parameter_changes["direction"] == candidate.direction


def test_mock_generator_creates_real_wav(tmp_path):
    context = SongContext(version=1)
    candidate = build_candidates(context, "test")[0]
    request = candidate_to_render_request(candidate, context)
    result = MockAudioGenerator(tmp_path).render(request)
    assert result.artifact_ref.endswith(".wav")
    assert result.sample_rate == 44_100
    assert (tmp_path / f"{result.cache_key}.wav").exists()


def test_rendering_is_cached(tmp_path):
    request = build_render_request("identity-v1", 1, "test", {"direction": "identity"})
    generator = MockAudioGenerator(tmp_path)
    first = generator.render(request)
    second = generator.render(request)
    assert first.cache_key == second.cache_key
    assert first.artifact_ref == second.artifact_ref
    assert build_cache_key("identity-v1", 1, "audio", {"direction": "identity"}) == first.cache_key


def test_audio_preview_uses_generated_sequence_not_placeholder(tmp_path):
    context = SongContext(version=1, bpm=120, key="C", scale="major", chord_progression=["C", "Am", "F", "G"])
    candidate = build_candidates(context, "make chord harmony")[0]
    request = candidate_to_render_request(candidate, context)
    result = MockAudioGenerator(tmp_path).render(request)
    import wave
    import numpy as np
    with wave.open(result.artifact_ref, "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert samples.size > 0
    assert np.max(np.abs(samples)) > 100
    assert result.duration_seconds >= 2.0

def test_multi_source_audio_preview_mixes_authority_sources(tmp_path):
    import math
    import wave
    import numpy as np
    def tone(path, seconds):
        sr=22050
        t=np.arange(int(sr*seconds))/sr
        x=.15*np.sin(2*math.pi*220*t)
        with wave.open(str(path),"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes((x*32767).astype("<i2").tobytes())

    first=tmp_path/"drums.wav"
    second=tmp_path/"guitar.wav"
    tone(first,2.0)
    tone(second,2.0)
    context=SongContext(version=2)
    candidate=build_candidates(context,"test")[0]
    request=build_render_request(
        candidate.id, context.version, candidate.intent,
        {"authority_analysis":{
            str(first):{"bpm":120.0,"dimension":"rhythm"},
            str(second):{"bpm":120.0,"dimension":"harmony"},
        }},
        source_asset_ids=(str(first),str(second)),
    )
    result=MockAudioGenerator(tmp_path/"cache").render(request)
    assert result.metadata["stem_count"] == 2
    assert len(result.metadata["stem_refs"]) == 2

def test_four_audio_candidates_render_as_distinct_cached_artifacts(tmp_path):
    from audio.rendering.candidates import preview_candidates
    from core.music_context.models import SongContext
    context=SongContext(version=0)
    context.candidates=build_candidates(context,"make a bass")
    rendered=preview_candidates(context,context.candidates,MockAudioGenerator(tmp_path),kind="audio")
    assert rendered.version == context.version
    refs=[candidate.audio_refs[0] for candidate in rendered.candidates]
    assert len(refs) == 4
    assert len(set(refs)) == 4
    assert all(candidate.status == "preview_ready" for candidate in rendered.candidates)


def test_project_assets_are_injected_into_render_request(tmp_path):
    from core.project.models import AudioAsset, MusicProject
    context = SongContext(version=4)
    candidate = build_candidates(context, "render project")[0]
    source = tmp_path / "guitar.wav"
    project = MusicProject(
        id="project-v1",
        context=context,
        assets=[AudioAsset(id="guitar-main", path=str(source), role_hint="guitar")],
    )
    request = candidate_to_render_request(candidate, context, project=project)
    assert request.parameter_changes["asset_paths"]["guitar-main"] == str(source)


def test_project_context_version_must_match_render_context():
    from core.project.models import MusicProject
    context = SongContext(version=4)
    stale = MusicProject(id="stale", context=SongContext(version=3), assets=[])
    candidate = build_candidates(context, "render project")[0]
    import pytest
    with pytest.raises(ValueError, match="belongs to context"):
        candidate_to_render_request(candidate, context, project=stale)


def test_preview_candidates_injects_project_asset_paths(tmp_path):
    from audio.rendering.candidates import preview_candidates
    from core.project.models import AudioAsset, MusicProject
    context = SongContext(version=5)
    candidate = build_candidates(context, "render project")[0]
    source = tmp_path / "guitar.wav"
    project = MusicProject(
        id="project-preview-v1",
        context=context,
        assets=[AudioAsset(id="guitar-main", path=str(source), role_hint="guitar")],
    )

    class CaptureRenderer:
        def __init__(self):
            self.request = None
        def render(self, request):
            self.request = request
            from audio.rendering.contracts import RenderResult
            return RenderResult(candidate_id=request.candidate_id, kind="audio", artifact_ref="preview.wav", cache_key="preview", duration_seconds=1.0, sample_rate=44100, metadata={})

    renderer = CaptureRenderer()
    rendered = preview_candidates(context, [candidate], renderer, kind="audio", project=project)
    assert rendered.candidates[0].audio_refs == ("preview.wav",)
    assert renderer.request.parameter_changes["asset_paths"]["guitar-main"] == str(source)
