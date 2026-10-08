import pytest

from audio.cache.keys import build_cache_key
from audio.rendering.contracts import build_render_request
from audio.rendering.mock import MockAudioGenerator
from audio.rendering.pipeline import candidate_to_render_request
from core.candidates.engine import build_candidates
from core.music_context.models import SongContext


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


def test_project_assets_become_render_sources_in_project_order(tmp_path):
    from core.project.models import AudioAsset, MusicProject
    context = SongContext(version=6)
    candidate = build_candidates(context, "render all project assets")[0]
    first = tmp_path / "drums.wav"
    second = tmp_path / "guitar.wav"
    project = MusicProject(
        id="project-registry-v1",
        context=context,
        assets=[
            AudioAsset(id="drums-main", path=str(first), role_hint="drums"),
            AudioAsset(id="guitar-main", path=str(second), role_hint="guitar"),
        ],
    )
    request = candidate_to_render_request(candidate, context, project=project)
    assert request.source_asset_ids == ("drums-main", "guitar-main")
    assert request.parameter_changes["asset_paths"] == {
        "drums-main": str(first),
        "guitar-main": str(second),
    }


def test_project_registry_reaches_real_source_audio_generator(tmp_path):
    import math
    import wave

    import numpy as np

    from audio.rendering.source import SourceAudioGenerator
    from core.project.models import AudioAsset, MusicProject

    def tone(path, hz):
        sr = 22050
        t = np.arange(int(sr * 1.0)) / sr
        x = 0.15 * np.sin(2 * math.pi * hz * t)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes((x * 32767).astype("<i2").tobytes())

    drums = tmp_path / "drums.wav"
    guitar = tmp_path / "guitar.wav"
    tone(drums, 120)
    tone(guitar, 220)

    context = SongContext(version=7)
    candidate = build_candidates(context, "render the project")[0]
    project = MusicProject(
        id="e2e-project",
        context=context,
        assets=[
            AudioAsset(id="drums-main", path=str(drums), role_hint="drums"),
            AudioAsset(id="guitar-main", path=str(guitar), role_hint="guitar"),
        ],
    )
    request = candidate_to_render_request(candidate, context, project=project)
    result = SourceAudioGenerator(tmp_path / "cache").render_mix(request)
    assert result.metadata["stem_count"] == 2
    assert len(result.metadata["stem_refs"]) == 2
    assert result.duration_seconds == 1.0

def test_project_asset_registry_rejects_duplicate_ids(tmp_path):
    import pytest

    from core.project.models import AudioAsset, MusicProject
    context = SongContext(version=1)
    with pytest.raises(ValueError, match="Duplicate AudioAsset id"):
        MusicProject(
            id="invalid-project",
            context=context,
            assets=[
                AudioAsset(id="same", path=str(tmp_path / "a.wav")),
                AudioAsset(id="same", path=str(tmp_path / "b.wav")),
            ],
        )


def test_project_asset_registry_rejects_empty_paths():
    import pytest

    from core.project.models import AudioAsset, MusicProject
    with pytest.raises(ValueError, match="path must not be empty"):
        MusicProject(
            id="invalid-project",
            context=SongContext(version=1),
            assets=[AudioAsset(id="drums-main", path="")],
        )

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


def test_synchronized_audition_shares_one_transport_clock():
    from audio.rendering.contracts import RenderResult, build_synchronized_audition
    results = tuple(
        RenderResult(candidate_id=cid, kind="audio", artifact_ref=f"{cid}.wav", cache_key=cid, duration_seconds=4.0, sample_rate=44100)
        for cid in ("A", "B", "C", "D")
    )
    audition = build_synchronized_audition(7, results, loop_start_seconds=1.0, loop_end_seconds=3.0)
    assert audition.context_version == 7
    assert audition.sample_rate == 44100
    assert audition.duration_seconds == 4.0
    assert audition.candidate_ids == ("A", "B", "C", "D")
    assert all(track.start_seconds == 0.0 for track in audition.tracks)
    assert audition.loop_start_seconds == 1.0
    assert audition.loop_end_seconds == 3.0


def test_synchronized_audition_rejects_mismatched_sample_rates():
    from audio.rendering.contracts import RenderResult, build_synchronized_audition
    results = [
        RenderResult(candidate_id="A", kind="audio", artifact_ref="a.wav", cache_key="a", duration_seconds=2.0, sample_rate=44100),
        RenderResult(candidate_id="B", kind="audio", artifact_ref="b.wav", cache_key="b", duration_seconds=2.0, sample_rate=48000),
    ]
    with pytest.raises(ValueError, match="share one sample rate"):
        build_synchronized_audition(1, results)


def test_synchronized_audition_uses_longest_candidate_as_transport_duration():
    from audio.rendering.contracts import RenderResult, build_synchronized_audition
    results = [
        RenderResult(candidate_id="A", kind="audio", artifact_ref="a.wav", cache_key="a", duration_seconds=2.0, sample_rate=44100),
        RenderResult(candidate_id="B", kind="audio", artifact_ref="b.wav", cache_key="b", duration_seconds=3.5, sample_rate=44100),
    ]
    audition = build_synchronized_audition(2, results)
    assert audition.duration_seconds == 3.5
    assert tuple(track.start_seconds for track in audition.tracks) == (0.0, 0.0)


def test_audition_manifest_is_playback_ready_and_deterministic(tmp_path):
    from audio.rendering.audition import write_audition_manifest
    from audio.rendering.contracts import RenderResult, build_synchronized_audition

    results = [
        RenderResult(
            candidate_id=cid,
            kind="audio",
            artifact_ref=f"/cache/{cid}.wav",
            cache_key=cid,
            duration_seconds=2.5,
            sample_rate=44100,
        )
        for cid in ("A", "B", "C", "D")
    ]
    audition = build_synchronized_audition(
        9, results, loop_start_seconds=0.5, loop_end_seconds=2.0
    )
    path = write_audition_manifest(audition, tmp_path / "audition.json")
    payload = __import__("json").loads((tmp_path / "audition.json").read_text())
    assert path.endswith("audition.json")
    assert payload["sample_rate"] == 44100
    assert payload["duration_seconds"] == 2.5
    assert payload["loop"] == {"start_seconds": 0.5, "end_seconds": 2.0}
    assert [track["candidate_id"] for track in payload["tracks"]] == ["A", "B", "C", "D"]
    assert all(track["start_seconds"] == 0.0 for track in payload["tracks"])


def test_playback_selection_switches_candidates_without_moving_playhead():
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.playback import build_playback_state, select_audition_track

    results = [
        RenderResult(
            candidate_id=cid,
            kind="audio",
            artifact_ref=f"/cache/{cid}.wav",
            cache_key=cid,
            duration_seconds=4.0,
            sample_rate=44100,
        )
        for cid in ("A", "B", "C", "D")
    ]
    manifest = build_audition_manifest(
        build_synchronized_audition(12, results, loop_start_seconds=1.0, loop_end_seconds=3.0)
    )
    track = select_audition_track(manifest, "C", 2.25)
    state = build_playback_state(manifest, "C", 2.25)
    assert track.candidate_id == "C"
    assert track.artifact_ref == "/cache/C.wav"
    assert state == state.__class__("C", 2.25, 1.0, 3.0)


def test_playback_selection_rejects_unknown_candidate_and_invalid_position():
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.playback import select_audition_track

    result = RenderResult(
        candidate_id="A",
        kind="audio",
        artifact_ref="a.wav",
        cache_key="a",
        duration_seconds=2.0,
        sample_rate=44100,
    )
    manifest = build_audition_manifest(build_synchronized_audition(1, [result]))
    with pytest.raises(ValueError, match="Unknown audition candidate"):
        select_audition_track(manifest, "B", 0.5)
    with pytest.raises(ValueError, match="position_seconds"):
        select_audition_track(manifest, "A", 2.1)



def test_playback_candidate_switch_preserves_shared_playhead():
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.playback import build_playback_state, switch_audition_candidate

    results = [
        RenderResult(
            candidate_id=cid,
            kind="audio",
            artifact_ref=f"/cache/{cid}.wav",
            cache_key=cid,
            duration_seconds=4.0,
            sample_rate=44100,
        )
        for cid in ("A", "B", "C", "D")
    ]
    manifest = build_audition_manifest(build_synchronized_audition(13, results))
    state = build_playback_state(manifest, "A", 2.75)
    switched = switch_audition_candidate(manifest, state, "D")
    assert switched.candidate_id == "D"
    assert switched.position_seconds == 2.75
    assert switched.loop_start_seconds == state.loop_start_seconds
    assert switched.loop_end_seconds == state.loop_end_seconds


def test_playback_advances_and_wraps_inside_loop():
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.playback import advance_playback, build_playback_state

    result = RenderResult(
        candidate_id="A",
        kind="audio",
        artifact_ref="a.wav",
        cache_key="a",
        duration_seconds=5.0,
        sample_rate=44100,
    )
    manifest = build_audition_manifest(
        build_synchronized_audition(14, [result], loop_start_seconds=1.0, loop_end_seconds=3.0)
    )
    state = build_playback_state(manifest, "A", 2.5)
    wrapped = advance_playback(manifest, state, 1.0)
    assert wrapped.position_seconds == 1.5
    backwards = advance_playback(manifest, wrapped, -1.0)
    assert backwards.position_seconds == 2.5


def test_playback_without_loop_clamps_to_transport_bounds():
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.playback import advance_playback, build_playback_state

    result = RenderResult(
        candidate_id="A",
        kind="audio",
        artifact_ref="a.wav",
        cache_key="a",
        duration_seconds=5.0,
        sample_rate=44100,
    )
    manifest = build_audition_manifest(build_synchronized_audition(15, [result]))
    state = build_playback_state(manifest, "A", 4.0)
    assert advance_playback(manifest, state, 3.0).position_seconds == 5.0
    assert advance_playback(manifest, state, -6.0).position_seconds == 0.0


def test_audition_runtime_switches_wav_artifacts_at_shared_playhead(tmp_path):
    import wave

    import numpy as np

    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.runtime import AuditionRuntime

    def tone(path, hz):
        sr = 22050
        t = np.arange(int(sr * 2.0)) / sr
        x = 0.15 * np.sin(2 * np.pi * hz * t)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes((x * 32767).astype("<i2").tobytes())

    first, second = tmp_path / "A.wav", tmp_path / "B.wav"
    tone(first, 220)
    tone(second, 330)
    results = [
        RenderResult(candidate_id="A", kind="audio", artifact_ref=str(first),
                     cache_key="A", duration_seconds=2.0, sample_rate=22050),
        RenderResult(candidate_id="B", kind="audio", artifact_ref=str(second),
                     cache_key="B", duration_seconds=2.0, sample_rate=22050),
    ]
    manifest = build_audition_manifest(build_synchronized_audition(20, results))
    runtime = AuditionRuntime(manifest, initial_candidate_id="A", initial_position_seconds=0.5)
    first_block = runtime.read_frames(2205)
    assert first_block.candidate_id == "A"
    assert first_block.start_seconds == 0.5
    assert runtime.state.position_seconds == 0.6
    runtime.select_candidate("B")
    second_block = runtime.read_frames(2205)
    assert second_block.candidate_id == "B"
    assert second_block.start_seconds == 0.6
    assert runtime.state.position_seconds == 0.7
    assert second_block.pcm != first_block.pcm


def test_audition_runtime_pads_short_candidate_with_silence(tmp_path):
    import wave

    import numpy as np

    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.runtime import AuditionRuntime

    sr = 22050
    t = np.arange(int(sr * 0.5)) / sr
    x = 0.15 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes((x * 32767).astype("<i2").tobytes())
    result = RenderResult(candidate_id="A", kind="audio", artifact_ref=str(path),
                          cache_key="A", duration_seconds=0.5, sample_rate=sr)
    manifest = build_audition_manifest(build_synchronized_audition(21, [result]))
    runtime = AuditionRuntime(manifest, initial_position_seconds=0.4)
    block = runtime.read_frames(int(sr * 0.2))
    assert block.frames == int(sr * 0.2)
    assert any(block.pcm)
    assert runtime.state.position_seconds == manifest.duration_seconds


def test_render_manifest_runtime_end_to_end_for_four_candidates(tmp_path):
    from audio.rendering.candidates import preview_candidates
    from audio.rendering.contracts import (
        RenderResult,
        build_audition_manifest,
        build_synchronized_audition,
    )
    from audio.rendering.runtime import AuditionRuntime

    context = SongContext(version=30)
    candidates = build_candidates(context, "make the bass more present")
    renderer = MockAudioGenerator(tmp_path)
    rendered_context = preview_candidates(
        context, candidates, renderer, kind="audio"
    )

    results = []
    for candidate in rendered_context.candidates:
        request = candidate_to_render_request(candidate, rendered_context, kind="audio")
        result = renderer.render(request)
        results.append(result)

    audition = build_synchronized_audition(rendered_context.version, results)
    manifest = build_audition_manifest(audition)
    runtime = AuditionRuntime(
        manifest,
        initial_candidate_id=manifest.candidate_ids[0],
        initial_position_seconds=0.25,
    )

    first = runtime.read_frames(4410)
    assert first.candidate_id == manifest.candidate_ids[0]
    assert first.start_seconds == 0.25
    assert runtime.state.position_seconds == 0.35

    runtime.select_candidate(manifest.candidate_ids[-1])
    second = runtime.read_frames(4410)
    assert second.candidate_id == manifest.candidate_ids[-1]
    assert second.start_seconds == 0.35
    assert runtime.state.position_seconds == 0.45
    assert second.pcm != first.pcm
