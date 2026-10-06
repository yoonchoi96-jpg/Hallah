"""Candidate generation primitives."""

from core.candidates.models import Candidate, CandidateDirection
from core.music_context.models import SongContext


DIRECTIONS: tuple[CandidateDirection, ...] = (
    "identity", "natural", "bold", "experimental"
)


def build_candidates(context: SongContext, intent: str) -> tuple[Candidate, ...]:
    """Create four non-destructive candidate records for one request."""
    return tuple(
        Candidate(
            id=f"v{context.version}-{direction}",
            direction=direction,
            intent=intent,
            rationale=_rationale(direction),
            parent_context_version=context.version,
        )
        for direction in DIRECTIONS
    )


def _rationale(direction: CandidateDirection) -> str:
    return {
        "identity": "Preserve the strongest existing musical identity.",
        "natural": "Integrate the requested change with minimal friction.",
        "bold": "Push the requested idea further for a stronger production result.",
        "experimental": "Explore a deliberately unconventional interpretation.",
    }[direction]
