"""Non-destructive candidate generation bound to SongContext."""
from __future__ import annotations

from core.candidates.models import Candidate, CandidateDirection
from core.music_context.models import SongContext

DIRECTIONS: tuple[CandidateDirection, ...] = ("identity", "natural", "bold", "experimental")


def _constraints_for(context: SongContext, direction: CandidateDirection) -> tuple[str, ...]:
    fixed = [c.target_id for c in context.constraints if c.type == "fixed"]
    blocked = ", ".join(fixed) if fixed else "none"
    return (
        f"fixed:{blocked}",
        f"context_version:{context.version}",
        f"direction:{direction}",
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
                "constraints": _constraints_for(context, direction),
            },
        )
        for direction in DIRECTIONS
    ]
    return candidates
