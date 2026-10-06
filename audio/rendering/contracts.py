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
