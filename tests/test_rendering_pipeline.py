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
