"""Non-destructive candidate generation bound to SongContext."""
from __future__ import annotations

from core.candidates.models import Candidate, CandidateDirection
from core.music_context.models import SongContext

DIRECTIONS: tuple[CandidateDirection, ...] = ("identity", "natural", "bold", "experimental")


def _authority_constraints(context: SongContext, intent: str) -> tuple[str, ...]:
    text = intent.lower()
    if any(x in text for x in ("bass", "베이스", "low end", "저음")):
        dimension = "low_end"
    elif any(x in text for x in ("drum", "rhythm", "groove", "드럼", "리듬")):
        dimension = "rhythm"
    elif any(x in text for x in ("chord", "harmony", "pad", "화음", "코드", "패드")):
        dimension = "harmony"
    else:
        dimension = "melody"
    ranked = sorted(
        (a for a in context.authorities if a.dimension == dimension),
        key=lambda a: a.confidence,
        reverse=True,
    )
    return tuple(f"authority:{dimension}:{a.source_id}:{a.confidence:.3f}" for a in ranked)


def _constraints_for(context: SongContext, direction: CandidateDirection, intent: str) -> tuple[str, ...]:
    fixed = [c.target_id for c in context.constraints if c.type == "fixed"]
    blocked = ", ".join(fixed) if fixed else "none"
    return (
        f"fixed:{blocked}",
        f"context_version:{context.version}",
        f"direction:{direction}",
        *_authority_constraints(context, intent),
    )


def build_candidates(context: SongContext, intent: str) -> list[Candidate]:
    candidates = [
        Candidate(
            id=f"{direction}-v{context.version}",
            direction=direction,
            intent=intent,
            rationale={
                "identity": "Preserve the established musical identity.",
                "natural": "Integrate naturally with the current context.",
                "bold": "Make a stronger but coherent creative change.",
                "experimental": "Explore a deliberately unusual direction.",
            }[direction],
            parent_context_version=context.version,
            parameter_changes={
                "constraints": _constraints_for(context, direction, intent),
            },
        )
        for direction in DIRECTIONS
    ]
    return candidates
