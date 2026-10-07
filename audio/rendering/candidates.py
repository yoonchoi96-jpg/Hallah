"""Render candidate previews and attach artifacts immutably."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Literal

from audio.rendering.contracts import CandidateRenderer
from audio.rendering.pipeline import candidate_to_render_request
from core.candidates.lifecycle import CandidateLifecycleError, transition_candidate
from core.candidates.models import Candidate
from core.music_context.models import SongContext
from core.project.models import MusicProject


def preview_candidates(
    context: SongContext,
    candidates: Iterable[Candidate],
    renderer: CandidateRenderer,
    *,
    kind: Literal["audio", "midi"] = "midi",
    project: MusicProject | None = None,
) -> SongContext:
    """Render candidates without advancing musical context version."""
    source = tuple(candidates)
    updated: list[Candidate] = []
    for candidate in source:
        if candidate.parent_context_version != context.version:
            raise CandidateLifecycleError(
                f"Cannot preview stale candidate {candidate.id}: "
                f"v{candidate.parent_context_version} != v{context.version}."
            )
        result = renderer.render(candidate_to_render_request(candidate, context, kind=kind, project=project))
        ready = transition_candidate(candidate, "preview_ready")
        updated.append(
            replace(
                ready,
                midi_refs=ready.midi_refs + ((result.artifact_ref,) if result.kind == "midi" else ()),
                audio_refs=ready.audio_refs + ((result.artifact_ref,) if result.kind == "audio" else ()),
            )
        )
    by_id = {candidate.id: candidate for candidate in updated}
    base_candidates = list(context.candidates)
    base_ids = {candidate.id for candidate in base_candidates}
    base_candidates.extend(candidate for candidate in updated if candidate.id not in base_ids)
    new = context.next_version()
    new.version = context.version
    new.candidates = [by_id.get(candidate.id, candidate) for candidate in base_candidates]
    new.candidate_history.extend(f"preview:{candidate.id}:v{context.version}" for candidate in updated)
    return new
