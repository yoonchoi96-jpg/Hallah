"""Deterministic analysis orchestration for Music Agent V0."""

from dataclasses import dataclass
from typing import Protocol

from core.analysis.contracts import AudioAnalysis
from core.project.models import MusicProject


class Analyzer(Protocol):
    def analyze(self, asset_id: str) -> AudioAnalysis: ...


@dataclass(frozen=True)
class AnalysisBatch:
    results: tuple[AudioAnalysis, ...]

    def by_asset(self) -> dict[str, AudioAnalysis]:
        return {result.asset_id: result for result in self.results}


def analyze_assets(analyzer: Analyzer, asset_ids: list[str]) -> AnalysisBatch:
    """Analyze all assets through a provider-neutral contract."""
    return AnalysisBatch(tuple(analyzer.analyze(asset_id) for asset_id in asset_ids))



def analyze_project_assets(analyzer: Analyzer, project: MusicProject) -> AnalysisBatch:
    """Analyze a MusicProject while preserving its logical asset IDs.

    An Analyzer operates on a concrete source reference (for example a WAV path),
    while the rest of Hallah should reason about stable project asset IDs. This
    bridge keeps those concerns separate: resolve the project registry here,
    analyze the real files, then relabel each result with the canonical ID.
    """
    from dataclasses import replace

    results = []
    for asset in project.assets:
        analysis = analyzer.analyze(asset.path)
        if analysis.asset_id != asset.id:
            analysis = replace(analysis, asset_id=asset.id)
        if asset.role_hint:
            confidence = dict(analysis.confidence)
            confidence["role_hint"] = 1.0
            analysis = replace(analysis, role=asset.role_hint, confidence=confidence)
        results.append(analysis)
    return AnalysisBatch(tuple(results))
