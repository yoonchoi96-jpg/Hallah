"""Bridge candidate definitions to asynchronous render requests."""
from __future__ import annotations

from core.candidates.models import Candidate
from core.music_context.models import SongContext
from audio.rendering.contracts import RenderRequest, build_render_request


def candidate_to_render_request(
    candidate: Candidate,
    context: SongContext,
    *,
    kind: str = "audio",
) -> RenderRequest:
    if candidate.parent_context_version != context.version:
        raise ValueError(
            f"Candidate {candidate.id} belongs to context v{candidate.parent_context_version}, "
            f"not v{context.version}."
        )
    changes = dict(candidate.parameter_changes)
    changes["direction"] = candidate.direction
    return build_render_request(
        candidate.id,
        context.version,
        candidate.intent,
        changes,
        kind=kind,  # type: ignore[arg-type]
    )
