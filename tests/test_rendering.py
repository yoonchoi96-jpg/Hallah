from audio.rendering.contracts import RenderResult, build_render_request
from audio.cache import build_cache_key


def test_render_request_is_bound_to_context_and_candidate():
    request = build_render_request(
        "natural-v3", 3, "fit guitar to drums", {"bpm": 120}
    )
    assert request.candidate_id == "natural-v3"
    assert request.context_version == 3
    assert request.kind == "audio"


def test_render_result_exposes_cacheable_artifact_reference():
    result = RenderResult(
        candidate_id="bold-v3",
        kind="midi",
        artifact_ref="cache://bold-v3.mid",
        cache_key="abc",
    )
    assert result.artifact_ref.startswith("cache://")
    assert result.kind == "midi"


def test_cache_key_is_deterministic():
    a = build_cache_key("x", 1, "audio", {"b": 2, "a": 1})
    b = build_cache_key("x", 1, "audio", {"a": 1, "b": 2})
    assert a == b
