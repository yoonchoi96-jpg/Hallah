"""Bridge candidate definitions to render requests."""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from audio.rendering.contracts import RenderRequest, build_render_request
from core.candidates.models import Candidate
from core.music_context.models import SongContext
from core.project.models import MusicProject


def _authority_analysis(
    context: SongContext, changes: dict[str, object]
) -> dict[str, dict[str, object]]:
    selected = {}
    for item in changes.get("constraints", ()):
        token = str(item)
        if not token.startswith("authority:"):
            continue
        parts = token.split(":", 3)
        if len(parts) != 4:
            continue
        _, dimension, source_id, _ = parts
        analysis = context.analyses.get(source_id)
        if analysis is not None:
            selected[source_id] = {"dimension": dimension, **asdict(analysis)}
    return selected


def candidate_to_render_request(
    candidate: Candidate,
    context: SongContext,
    *,
    kind: Literal["audio", "midi"] = "audio",
    project: MusicProject | None = None,
) -> RenderRequest:
    if candidate.parent_context_version != context.version:
        raise ValueError(
            f"Candidate {candidate.id} belongs to context v{candidate.parent_context_version}, "
            f"not v{context.version}."
        )

    changes = dict(candidate.parameter_changes)
    changes["direction"] = candidate.direction

    project_asset_ids: tuple[str, ...] = ()
    if project is not None:
        if project.context.version != context.version:
            raise ValueError(
                f"Project {project.id} belongs to context v{project.context.version}, "
                f"not v{context.version}."
            )
        # MusicProject is the canonical asset registry. Keep logical IDs in the
        # render contract and paths in parameter_changes so generators remain
        # provider-neutral.
        changes["asset_paths"] = project.asset_map()
        project_asset_ids = project.asset_ids()

    if context.bpm is not None:
        changes.setdefault("bpm", context.bpm)
    if context.key is not None:
        changes.setdefault("key", context.key)
    if context.scale is not None:
        changes.setdefault("scale", context.scale)
    if context.chord_progression:
        changes.setdefault("chord_progression", tuple(context.chord_progression))
    if context.adaptation_overrides:
        changes.setdefault("source_adaptations", context.adaptation_overrides)

    authority_data = _authority_analysis(context, changes)
    if authority_data:
        changes["authority_analysis"] = authority_data
        from audio.processing.masking import build_relationship_map

        relationship_rows = tuple(
            {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.type,
                "dimension": rel.dimension,
                "confidence": rel.confidence,
            }
            for rel in context.relationships
        )
        changes["mix_relationships"] = build_relationship_map(
            authority_data, relationship_rows
        )

    # A project defines the complete renderable source set. Authority analysis
    # can narrow metadata, but must never accidentally narrow the actual stems.
    # This is what lets Asset Registry -> Analysis -> Authority -> Adaptation ->
    # Candidate -> real stem rendering remain one continuous project flow.
    source_asset_ids = project_asset_ids
    if not source_asset_ids:
        explicit_ids = changes.get("source_asset_ids")
        if isinstance(explicit_ids, (list, tuple)):
            source_asset_ids = tuple(str(value) for value in explicit_ids)
        elif authority_data:
            source_asset_ids = tuple(authority_data.keys())

    return build_render_request(
        candidate.id,
        context.version,
        candidate.intent,
        changes,
        kind=kind,
        source_asset_ids=source_asset_ids,
        output_format=("mid" if kind == "midi" else "wav"),
    )
