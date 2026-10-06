"""Deterministic analysis orchestration for Music Agent V0."""

from dataclasses import dataclass
from typing import Protocol

from core.analysis.contracts import AudioAnalysis


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
