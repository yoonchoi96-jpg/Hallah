import pytest

from core.candidates.engine import build_candidates
from core.candidates.lifecycle import (
    CandidateLifecycleError,
    apply_candidate,
    reject_candidate,
    select_candidate,
    transition_candidate,
    validate_candidate,
)
from core.candidates.models import Candidate
from core.music_context.constraints import ContextConstraint
from core.music_context.models import MusicalAuthority, SongContext


def test_candidate_lifecycle_transition():
    candidate = Candidate("a", "identity", "x", "r", 0)
    candidate = transition_candidate(candidate, "preview_ready")
    candidate = transition_candidate(candidate, "selected")
    candidate = transition_candidate(candidate, "applied")
    assert candidate.status == "applied"


def test_invalid_candidate_transition():
    candidate = Candidate("a", "identity", "x", "r", 0, status="rejected")
    with pytest.raises(CandidateLifecycleError):
        transition_candidate(candidate, "selected")


def test_selection_is_non_destructive():
    context = SongContext(version=0, candidates=build_candidates(SongContext(), "make bass"))
    new = select_candidate(context, "identity-v0")
    assert context.version == 0
    assert context.candidates[0].status == "proposed"
    assert new.version == 1
    assert next(c.status for c in new.candidates if c.id == "identity-v0") == "selected"


def test_rejection_preserves_history():
    context = SongContext(version=0, candidates=build_candidates(SongContext(), "make bass"))
    new = reject_candidate(context, "bold-v0")
    assert context.version == 0
    assert any("rejected:bold-v0" in item for item in new.candidate_history)
    assert next(c.status for c in new.candidates if c.id == "bold-v0") == "rejected"


def test_stale_candidate_is_rejected():
    context = SongContext(version=2)
    candidate = Candidate("old", "identity", "x", "r", 1)
    result = validate_candidate(context, candidate)
    assert not result.valid
    assert result.errors


def test_fixed_constraint_is_enforced():
    context = SongContext(constraints=[ContextConstraint("drums", "fixed")])
    candidate = Candidate(
        "x",
        "natural",
        "change drums",
        "r",
        0,
        parameter_changes={"changes": {"drums": "new"}},
    )
    result = validate_candidate(context, candidate)
    assert not result.valid
    assert result.conflicts


def test_authority_override_is_allowed():
    context = SongContext(
        authorities=[MusicalAuthority("guitar", "harmony", 1.0)],
    )
    candidate = Candidate(
        "x",
        "bold",
        "change harmony",
        "r",
        0,
        parameter_changes={"authority_changes": {"harmony": "piano"}},
    )
    result = validate_candidate(context, candidate)
    assert result.valid


def test_apply_creates_new_context_and_keeps_parent_unchanged():
    context = SongContext(
        version=0,
        bpm=120,
        candidates=[
            Candidate(
                "b",
                "natural",
                "faster",
                "r",
                0,
                parameter_changes={"changes": {"bpm": 128}},
            )
        ],
    )
    selected = select_candidate(context, "b")
    applied = apply_candidate(selected, next(c for c in selected.candidates if c.id == "b"))
    assert context.version == 0
    assert selected.version == 1
    assert applied.version == 2
    assert context.bpm == 120
    assert applied.bpm == 128
    assert next(c.status for c in applied.candidates if c.id == "b") == "applied"


def test_apply_commits_authority_and_adaptation_changes():
    context = SongContext(
        version=0,
        authorities=[MusicalAuthority("guitar", "harmony", 0.8)],
        adaptation_overrides={"drums": {"rhythm": "guitar"}},
        candidates=[
            Candidate(
                "a",
                "bold",
                "make drums follow guitar pitch",
                "r",
                0,
                parameter_changes={
                    "authority_changes": {"harmony": "piano"},
                    "adaptation_overrides": {
                        "drums": {"pitch": "guitar"},
                    },
                },
            )
        ],
    )
    selected = select_candidate(context, "a")
    applied = apply_candidate(selected, next(c for c in selected.candidates if c.id == "a"))

    harmony = [a for a in applied.authorities if a.dimension == "harmony"]
    assert len(harmony) == 1
    assert harmony[0].source_id == "piano"
    assert harmony[0].confidence == 1.0
    assert applied.adaptation_overrides["drums"] == {
        "rhythm": "guitar",
        "pitch": "guitar",
    }
