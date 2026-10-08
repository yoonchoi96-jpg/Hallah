"""Pure playback state helpers for synchronized candidate auditions."""
from __future__ import annotations

from dataclasses import dataclass

from audio.rendering.contracts import AuditionManifest, AuditionTrack


@dataclass(frozen=True)
class AuditionPlaybackState:
    """Current transport position and selected candidate."""

    candidate_id: str
    position_seconds: float
    loop_start_seconds: float
    loop_end_seconds: float


def select_audition_track(
    manifest: AuditionManifest,
    candidate_id: str,
    position_seconds: float,
) -> AuditionTrack:
    """Select a candidate without changing the shared transport position."""
    if not 0.0 <= position_seconds <= manifest.duration_seconds:
        raise ValueError("position_seconds must fall inside the audition duration")
    for track in manifest.tracks:
        if track.candidate_id == candidate_id:
            return track
    raise ValueError(f"Unknown audition candidate: {candidate_id}")


def build_playback_state(
    manifest: AuditionManifest,
    candidate_id: str,
    position_seconds: float = 0.0,
) -> AuditionPlaybackState:
    """Create a playback state that preserves the shared audition clock."""
    select_audition_track(manifest, candidate_id, position_seconds)
    return AuditionPlaybackState(
        candidate_id=candidate_id,
        position_seconds=position_seconds,
        loop_start_seconds=manifest.loop_start_seconds,
        loop_end_seconds=manifest.loop_end_seconds,
    )
