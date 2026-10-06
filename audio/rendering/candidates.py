"""Render candidate previews and attach artifacts immutably."""
from __future__ import annotations
from dataclasses import replace
from typing import Iterable
from audio.rendering.contracts import CandidateRenderer
from audio.rendering.pipeline import candidate_to_render_request
from core.candidates.lifecycle import CandidateLifecycleError, transition_candidate
from core.candidates.models import Candidate
from core.music_context.models import SongContext

def preview_candidates(context: SongContext, candidates: Iterable[Candidate], renderer: CandidateRenderer, *, kind: str = "midi") -> SongContext:
    """Render candidates and return a new context containing preview-ready artifacts."""
    source = tuple(candidates)
    new = context.next_version()
    updated: list[Candidate] = []
    for candidate in source:
        if candidate.parent_context_version != context.version:
            raise CandidateLifecycleError(
                f"Cannot preview stale candidate {candidate.id}: v{candidate.parent_context_version} != v{context.version}."
            )
        result = renderer.render(candidate_to_render_request(candidate, context, kind=kind))
        ready = transition_candidate(candidate, "preview_ready")
        updated.append(replace(
            ready,
            midi_refs=ready.midi_refs + ((result.artifact_ref,) if result.kind == "midi" else ()),
            audio_refs=ready.audio_refs + ((result.artifact_ref,) if result.kind == "audio" else ()),
        ))
    by_id = {candidate.id: candidate for candidate in updated}
    new.candidates = [by_id.get(candidate.id, candidate) for candidate in context.candidates]
    new.candidate_history.extend(f"preview:{candidate.id}:v{new.version}" for candidate in updated)
    return new
