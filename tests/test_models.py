from core.candidates.models import Candidate
from core.music_context.models import SongContext


def test_song_context_versioning():
    context = SongContext(title="Untitled")
    next_context = context.next_version()
    assert context.version == 0
    assert next_context.version == 1
    assert context.title == next_context.title


def test_candidate_is_bound_to_context_version():
    candidate = Candidate(
        id="candidate-a",
        direction="identity",
        intent="Preserve the drum groove.",
        rationale="Protect rhythmic authority.",
        parent_context_version=3,
    )
    assert candidate.parent_context_version == 3


def test_project_with_context_preserves_asset_registry():
    from core.project.models import AudioAsset, MusicProject

    context = SongContext(version=1)
    project = MusicProject(
        id="demo",
        context=context,
        assets=[AudioAsset(id="guitar", path="guitar.wav", role_hint="guitar")],
    )
    next_context = context.next_version()
    updated = project.with_context(next_context)

    assert updated.context.version == 2
    assert updated.asset_ids() == ("guitar",)
    assert updated.asset_map()["guitar"] == "guitar.wav"
    assert project.context.version == 1
