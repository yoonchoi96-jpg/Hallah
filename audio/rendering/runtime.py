"""Runtime adapter for synchronized candidate audio audition."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from audio.rendering.contracts import AuditionManifest, AuditionTrack
from audio.rendering.playback import (
    AuditionPlaybackState,
    advance_playback,
    build_playback_state,
    switch_audition_candidate,
)


@dataclass(frozen=True)
class AuditionAudioFrame:
    """PCM frame block returned from the currently selected artifact."""

    candidate_id: str
    start_seconds: float
    frames: int
    sample_rate: int
    channels: int
    sample_width: int
    pcm: bytes


class AuditionRuntime:
    """Read separate candidate artifacts through one shared transport clock."""

    def __init__(
        self,
        manifest: AuditionManifest,
        *,
        initial_candidate_id: str | None = None,
        initial_position_seconds: float = 0.0,
    ) -> None:
        candidate_id = initial_candidate_id or manifest.candidate_ids[0]
        self.manifest = manifest
        self._state = build_playback_state(
            manifest, candidate_id, initial_position_seconds
        )

    @property
    def state(self) -> AuditionPlaybackState:
        """Return the immutable current playback state."""
        return self._state

    @property
    def track(self) -> AuditionTrack:
        """Return the currently selected artifact descriptor."""
        for track in self.manifest.tracks:
            if track.candidate_id == self._state.candidate_id:
                return track
        raise RuntimeError("Playback state references an unknown candidate")

    def seek(self, position_seconds: float) -> AuditionPlaybackState:
        """Move the shared transport without changing the selected candidate."""
        self._state = build_playback_state(
            self.manifest, self._state.candidate_id, position_seconds
        )
        return self._state

    def select_candidate(self, candidate_id: str) -> AuditionPlaybackState:
        """Switch artifacts without moving the shared playhead."""
        self._state = switch_audition_candidate(
            self.manifest, self._state, candidate_id
        )
        return self._state

    def advance(self, delta_seconds: float) -> AuditionPlaybackState:
        """Advance the shared transport using the manifest loop semantics."""
        self._state = advance_playback(self.manifest, self._state, delta_seconds)
        return self._state

    def _read_from_artifact(self, position_seconds: float, frame_count: int) -> bytes:
        """Read one contiguous block, padding past artifact EOF with silence."""
        track = self.track
        with wave.open(str(Path(track.artifact_ref)), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            if sample_rate != self.manifest.sample_rate:
                raise ValueError(
                    "Audition artifact sample rate does not match manifest"
                )
            start_frame = min(
                wav.getnframes(), max(0, round(position_seconds * sample_rate))
            )
            wav.setpos(start_frame)
            pcm = wav.readframes(frame_count)
            actual_frames = len(pcm) // (channels * sample_width)
            missing = frame_count - actual_frames
            if missing:
                pcm += bytes(missing * channels * sample_width)
            return pcm

    def read_frames(self, frame_count: int) -> AuditionAudioFrame:
        """Read PCM frames at the shared playhead and advance transport time.

        Loop boundaries are applied to both transport and audio reads, so a
        block crossing the loop end contains the beginning of the loop rather
        than audio from beyond the loop boundary.
        """
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

        track = self.track
        start_seconds = self._state.position_seconds
        sample_rate = self.manifest.sample_rate
        channels = 1
        sample_width = 2
        remaining = frame_count
        position = start_seconds
        chunks: list[bytes] = []

        loop_start = self.manifest.loop_start_seconds
        loop_end = self.manifest.loop_end_seconds
        if loop_end is not None and loop_end > loop_start:
            loop_length = loop_end - loop_start
            position = loop_start + ((position - loop_start) % loop_length)

        while remaining:
            if loop_end is not None and loop_end > loop_start:
                frames_until_loop = max(
                    1, round((loop_end - position) * sample_rate)
                )
                chunk_frames = min(remaining, frames_until_loop)
            else:
                chunk_frames = remaining

            with wave.open(str(Path(track.artifact_ref)), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
            if sample_rate != self.manifest.sample_rate:
                raise ValueError(
                    "Audition artifact sample rate does not match manifest"
                )

            chunks.append(self._read_from_artifact(position, chunk_frames))
            remaining -= chunk_frames

            if (
                remaining
                and loop_end is not None
                and loop_end > loop_start
            ):
                position = loop_start

        self.advance(frame_count / float(sample_rate))
        return AuditionAudioFrame(
            candidate_id=track.candidate_id,
            start_seconds=start_seconds,
            frames=frame_count,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            pcm=b"".join(chunks),
        )
