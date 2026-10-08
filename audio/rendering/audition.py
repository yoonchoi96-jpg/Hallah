"""Materialize synchronized audition metadata for playback clients."""
from __future__ import annotations

import json
from pathlib import Path

from audio.rendering.contracts import SynchronizedAudition, build_audition_manifest


def write_audition_manifest(
    audition: SynchronizedAudition,
    output_path: str | Path,
) -> str:
    """Write a deterministic JSON manifest referencing rendered audio artifacts."""
    manifest = build_audition_manifest(audition)
    payload = {
        "version": 1,
        "context_version": manifest.context_version,
        "sample_rate": manifest.sample_rate,
        "duration_seconds": manifest.duration_seconds,
        "loop": {
            "start_seconds": manifest.loop_start_seconds,
            "end_seconds": manifest.loop_end_seconds,
        },
        "tracks": [
            {
                "candidate_id": track.candidate_id,
                "artifact_ref": track.artifact_ref,
                "start_seconds": track.start_seconds,
                "duration_seconds": track.duration_seconds,
                "sample_rate": track.sample_rate,
            }
            for track in manifest.tracks
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path)
