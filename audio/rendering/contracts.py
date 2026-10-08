"""Provider-neutral contracts for asynchronous candidate rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

RenderKind = Literal["audio", "midi"]


@dataclass(frozen=True)
class RenderRequest:
    candidate_id: str
    context_version: int
    kind: RenderKind
    intent: str
    source_asset_ids: tuple[str, ...] = ()
    parameter_changes: dict[str, object] = field(default_factory=dict)
    output_format: str = "wav"


@dataclass(frozen=True)
class RenderResult:
    candidate_id: str
    kind: RenderKind
    artifact_ref: str
    cache_key: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class AudioGenerator(Protocol):
    def render(self, request: RenderRequest) -> RenderResult: ...


class MidiGenerator(Protocol):
    def render(self, request: RenderRequest) -> RenderResult: ...


class CandidateRenderer(Protocol):
    def render(self, request: RenderRequest) -> RenderResult: ...


def build_render_request(
    candidate_id: str,
    context_version: int,
    intent: str,
    parameter_changes: dict[str, object],
    *,
    kind: RenderKind = "audio",
    source_asset_ids: tuple[str, ...] = (),
    output_format: str = "wav",
) -> RenderRequest:
    return RenderRequest(
        candidate_id=candidate_id,
        context_version=context_version,
        kind=kind,
        intent=intent,
        source_asset_ids=source_asset_ids,
        parameter_changes=parameter_changes,
        output_format=output_format,
    )


@dataclass(frozen=True)
class AuditionTrack:
    """One candidate positioned on the shared audition clock."""
    candidate_id: str
    artifact_ref: str
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class SynchronizedAudition:
    """Immutable A/B/C/D audition set sharing one transport timeline."""
    context_version: int
    sample_rate: int
    duration_seconds: float
    tracks: tuple[AuditionTrack, ...]
    loop_start_seconds: float = 0.0
    loop_end_seconds: float | None = None

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(track.candidate_id for track in self.tracks)


def build_synchronized_audition(
    context_version: int,
    results: tuple[RenderResult, ...] | list[RenderResult],
    *,
    loop_start_seconds: float = 0.0,
    loop_end_seconds: float | None = None,
) -> SynchronizedAudition:
    """Create a shared-clock audition session without mutating Song Context.

    Every candidate starts at t=0 and must use the same sample rate. Duration
    may differ; the transport duration is the longest candidate so switching
    candidates never changes the playhead origin or musical position.
    """
    if not results:
        raise ValueError("At least one render result is required")
    if any(result.kind != "audio" for result in results):
        raise ValueError("Synchronized audition requires audio render results")
    sample_rates = {result.sample_rate for result in results}
    if None in sample_rates or len(sample_rates) != 1:
        raise ValueError("All audition candidates must share one sample rate")
    if any(result.duration_seconds is None or result.duration_seconds < 0 for result in results):
        raise ValueError("All audition candidates must provide a non-negative duration")
    duration = max(float(result.duration_seconds) for result in results)
    if loop_start_seconds < 0 or loop_start_seconds > duration:
        raise ValueError("loop_start_seconds must fall inside the audition duration")
    if loop_end_seconds is not None and not loop_start_seconds <= loop_end_seconds <= duration:
        raise ValueError("loop_end_seconds must be after loop_start_seconds")
    tracks = tuple(
        AuditionTrack(
            candidate_id=result.candidate_id,
            artifact_ref=result.artifact_ref,
            duration_seconds=result.duration_seconds,
            sample_rate=result.sample_rate,
        )
        for result in results
    )
    return SynchronizedAudition(
        context_version=context_version,
        sample_rate=next(iter(sample_rates)),
        duration_seconds=duration,
        tracks=tracks,
        loop_start_seconds=loop_start_seconds,
        loop_end_seconds=loop_end_seconds,
    )


@dataclass(frozen=True)
class AuditionManifest:
    """Playback-ready manifest for a synchronized candidate audition."""
    context_version: int
    sample_rate: int
    duration_seconds: float
    tracks: tuple[AuditionTrack, ...]
    loop_start_seconds: float
    loop_end_seconds: float

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(track.candidate_id for track in self.tracks)


def build_audition_manifest(audition: SynchronizedAudition) -> AuditionManifest:
    """Freeze a synchronized audition into a playback-oriented manifest."""
    loop_end = (
        audition.duration_seconds
        if audition.loop_end_seconds is None
        else audition.loop_end_seconds
    )
    return AuditionManifest(
        context_version=audition.context_version,
        sample_rate=audition.sample_rate,
        duration_seconds=audition.duration_seconds,
        tracks=audition.tracks,
        loop_start_seconds=audition.loop_start_seconds,
        loop_end_seconds=loop_end,
    )
