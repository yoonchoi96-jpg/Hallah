"""Build SongContext from analyzed musical assets."""
from __future__ import annotations
from core.analysis.contracts import AudioAnalysis
from core.music_context.authority import infer_authority
from core.music_context.models import SongContext

def build_song_context(analyses: list[AudioAnalysis], title: str = "Untitled") -> SongContext:
    ctx = SongContext(title=title)
    if not analyses:
        return ctx
    bpms = [a.bpm for a in analyses if a.bpm is not None]
    tonal = [a for a in analyses if a.key and a.scale]
    if bpms:
        ctx.bpm = round(sum(bpms) / len(bpms), 2)
    if tonal:
        best = max(tonal, key=lambda a: a.confidence.get("key", 0.0))
        ctx.key, ctx.scale = best.key, best.scale
    ctx.authorities = infer_authority(analyses)
    ctx.decisions.append(f"Initialized from {len(analyses)} analyzed asset(s).")
    return ctx
