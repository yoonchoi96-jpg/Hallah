"""Bridge candidate definitions to render requests."""
from __future__ import annotations
from dataclasses import asdict
from typing import Literal
from core.candidates.models import Candidate
from core.music_context.models import SongContext
from audio.rendering.contracts import RenderRequest, build_render_request

def _authority_analysis(context: SongContext, changes: dict[str, object]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for item in changes.get("constraints", ()):
        token = str(item)
        if not token.startswith("authority:"):
            continue
        parts = token.split(":", 3)
        if len(parts) != 4:
            continue
        _, dimension, source_id, _confidence = parts
        analysis = context.analyses.get(source_id)
        if analysis is None:
            continue
        selected[source_id] = {"dimension": dimension, **asdict(analysis)}
    return selected

def candidate_to_render_request(candidate: Candidate, context: SongContext, *, kind: Literal["audio","midi"] = "audio") -> RenderRequest:
    if candidate.parent_context_version != context.version:
        raise ValueError(
            f"Candidate {candidate.id} belongs to context v{candidate.parent_context_version}, not v{context.version}."
        )
    changes = dict(candidate.parameter_changes)
    changes["direction"] = candidate.direction
    if context.bpm is not None: changes.setdefault("bpm", context.bpm)
    if context.key is not None: changes.setdefault("key", context.key)
    if context.scale is not None: changes.setdefault("scale", context.scale)
    if context.chord_progression: changes.setdefault("chord_progression", tuple(context.chord_progression))
    authority_data = _authority_analysis(context, changes)
    if authority_data: changes["authority_analysis"] = authority_data
    return build_render_request(
        candidate.id, context.version, candidate.intent, changes,
        kind=kind, output_format=("mid" if kind == "midi" else "wav"),
    )
