"""Candidate validation and lifecycle transitions."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.candidates.models import Candidate, CandidateStatus
from core.music_context.models import SongContext


class CandidateLifecycleError(ValueError):
    """Raised when a candidate lifecycle operation is invalid."""


@dataclass(frozen=True)
class CandidateValidation:
    valid: bool
    warnings: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)


_ALLOWED_TRANSITIONS: dict[CandidateStatus, frozenset[CandidateStatus]] = {
    "proposed": frozenset({"preview_ready", "selected", "rejected", "superseded"}),
    "preview_ready": frozenset({"selected", "rejected", "superseded"}),
    "selected": frozenset({"applied", "rejected", "superseded"}),
    "rejected": frozenset(),
    "applied": frozenset(),
    "superseded": frozenset(),
}


def transition_candidate(candidate: Candidate, status: CandidateStatus) -> Candidate:
    if status not in _ALLOWED_TRANSITIONS[candidate.status]:
        raise CandidateLifecycleError(
            f"Invalid candidate transition: {candidate.status} -> {status}."
        )
    return candidate.with_status(status)


def _candidate_in_context(context: SongContext, candidate_id: str) -> Candidate:
    for candidate in context.candidates:
        if candidate.id == candidate_id:
            return candidate
    raise CandidateLifecycleError(f"Candidate not found: {candidate_id}")


def validate_candidate(context: SongContext, candidate: Candidate) -> CandidateValidation:
    warnings: list[str] = []
    conflicts: list[str] = []
    errors: list[str] = []

    expected_parent = context.version
    selected_from_previous_version = (
        candidate.status == "selected"
        and candidate.parent_context_version == context.version - 1
    )
    if candidate.parent_context_version != expected_parent and not selected_from_previous_version:
        errors.append(
            f"Stale candidate: candidate belongs to context v{candidate.parent_context_version}, "
            f"current context is v{context.version}."
        )

    if candidate.status in {"rejected", "applied", "superseded"}:
        errors.append(f"Candidate is not selectable/applicable in state: {candidate.status}.")

    changes = candidate.parameter_changes.get("changes", {})
    if isinstance(changes, dict):
        fixed_targets = {
            constraint.target_id
            for constraint in context.constraints
            if constraint.type == "fixed"
        }
        for target in changes:
            if target in fixed_targets:
                conflicts.append(f"Candidate attempts to modify fixed target: {target}.")

    authority_changes = candidate.parameter_changes.get("authority_changes", {})
    if isinstance(authority_changes, dict):
        for dimension, source_id in authority_changes.items():
            authorities = [a for a in context.authorities if a.dimension == dimension]
            if authorities and all(a.source_id != source_id for a in authorities):
                conflicts.append(
                    f"Candidate authority conflict on {dimension}: {source_id} is not the "
                    "current authoritative source."
                )

    if conflicts:
        warnings.append("Candidate contains musical conflicts and cannot be safely applied.")

    return CandidateValidation(
        valid=not errors and not conflicts,
        warnings=tuple(warnings),
        conflicts=tuple(conflicts),
        errors=tuple(errors),
    )


def select_candidate(context: SongContext, candidate_id: str) -> SongContext:
    candidate = _candidate_in_context(context, candidate_id)
    validation = validate_candidate(context, candidate)
    if not validation.valid:
        raise CandidateLifecycleError(
            f"Candidate cannot be selected: {candidate_id}. "
            f"{'; '.join(validation.errors + validation.conflicts)}"
        )

    new = context.next_version()
    selected = transition_candidate(candidate, "selected")
    new.candidates = [
        selected if item.id == candidate_id else item
        for item in new.candidates
    ]
    new.candidate_history.append(f"selected:{candidate_id}:v{new.version}")
    new.decisions.append(f"[system] Selected candidate {candidate_id}.")
    return new


def reject_candidate(context: SongContext, candidate_id: str) -> SongContext:
    candidate = _candidate_in_context(context, candidate_id)
    if candidate.status in {"applied", "superseded"}:
        raise CandidateLifecycleError(f"Candidate cannot be rejected from state: {candidate.status}.")

    new = context.next_version()
    rejected = transition_candidate(candidate, "rejected")
    new.candidates = [
        rejected if item.id == candidate_id else item
        for item in new.candidates
    ]
    new.candidate_history.append(f"rejected:{candidate_id}:v{new.version}")
    new.decisions.append(f"[system] Rejected candidate {candidate_id}.")
    return new


def apply_candidate(context: SongContext, candidate: Candidate) -> SongContext:
    validation = validate_candidate(context, candidate)
    if not validation.valid:
        raise CandidateLifecycleError(
            f"Candidate cannot be applied: {candidate.id}. "
            f"{'; '.join(validation.errors + validation.conflicts)}"
        )

    current = _candidate_in_context(context, candidate.id)
    if current.status not in {"selected", "preview_ready", "proposed"}:
        raise CandidateLifecycleError(f"Candidate cannot be applied from state: {current.status}.")

    new = context.next_version()
    applied = transition_candidate(current, "applied")
    new.candidates = [
        applied if item.id == current.id else (
            transition_candidate(item, "superseded")
            if item.status == "selected" and item.id != current.id
            else item
        )
        for item in new.candidates
    ]

    changes = current.parameter_changes.get("changes", {})
    if isinstance(changes, dict):
        for key, value in changes.items():
            if key == "bpm":
                new.bpm = float(value)
            elif key == "key":
                new.key = str(value)
            elif key == "scale":
                new.scale = str(value)
            elif key == "title":
                new.title = str(value)

    new.candidate_history.append(f"applied:{current.id}:v{new.version}")
    new.decisions.append(
        f"[system] Applied candidate {current.id} from context v{context.version} to v{new.version}."
    )
    return new
