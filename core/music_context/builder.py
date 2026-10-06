"""Build SongContext from analyzed musical assets."""
from __future__ import annotations

from core.analysis.contracts import AudioAnalysis
from core.music_context.authority import infer_authority
from core.music_context.models import SongContext
from core.music_context.relationships import analyze_relationships, merge_conflicts
from core.music_context.resolve import resolve_authorities


def _best_for_dimension(
    analyses: list[AudioAnalysis], authorities, dimension: str
) -> AudioAnalysis | None:
    winner = next((r.winner for r in resolve_authorities(authorities) if r.dimension == dimension), None)
    return next((a for a in analyses if winner and a.asset_id == winner.source_id), None)


def build_song_context(analyses: list[AudioAnalysis], title: str = "Untitled") -> SongContext:
    ctx = SongContext(title=title)
    if not analyses:
        return ctx

    ctx.analyses = {analysis.asset_id: analysis for analysis in analyses}
    ctx.authorities = infer_authority(analyses)
    relationships, constraints, relationship_conflicts = analyze_relationships(
        analyses, ctx.authorities
    )
    ctx.relationships = relationships
    ctx.constraints = constraints
    ctx.conflicts = merge_conflicts(relationship_conflicts)

    rhythm = _best_for_dimension(analyses, ctx.authorities, "rhythm")
    tonal = _best_for_dimension(analyses, ctx.authorities, "harmony")
    bpm_source = rhythm or max(
        (a for a in analyses if a.bpm is not None),
        key=lambda a: a.confidence.get("bpm", 0.0),
        default=None,
    )
    tonal_source = tonal or max(
        (a for a in analyses if a.key and a.scale),
        key=lambda a: a.confidence.get("key", 0.0),
        default=None,
    )
    if bpm_source is not None:
        ctx.bpm = bpm_source.bpm
    if tonal_source is not None:
        ctx.key, ctx.scale = tonal_source.key, tonal_source.scale
        ctx.chord_progression = list(tonal_source.chords)

    if ctx.conflicts:
        ctx.pending_decisions.extend(ctx.conflicts)
    ctx.decisions.append(
        f"Initialized from {len(analyses)} analyzed asset(s); "
        "dimension-specific authorities resolve shared musical state."
    )
    return ctx
