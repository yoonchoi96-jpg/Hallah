"""Project and asset models."""
from __future__ import annotations
from dataclasses import dataclass
from core.music_context.models import SongContext

@dataclass(frozen=True)
class AudioAsset:
    id: str
    path: str
    role_hint: str | None = None

@dataclass
class MusicProject:
    id: str
    context: SongContext
    assets: list[AudioAsset]
