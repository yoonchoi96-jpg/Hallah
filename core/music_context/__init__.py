"""Hallah persistent musical context primitives."""
from core.music_context.authority import infer_authority
from core.music_context.builder import build_song_context
from core.music_context.resolve import (
    AuthorityResolution, build_relationships, detect_conflicts, enrich_context, resolve_authorities,
)
__all__ = [
    "AuthorityResolution", "build_relationships", "build_song_context",
    "detect_conflicts", "enrich_context", "infer_authority", "resolve_authorities",
]
