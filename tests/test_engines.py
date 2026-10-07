from agent.executive_producer.contracts import ProductionRequest
from agent.executive_producer.default import DefaultExecutiveProducer
from core.analysis.contracts import AudioAnalysis
from core.analysis.engine import analyze_assets, analyze_project_assets
from core.candidates.engine import build_candidates
from core.music_context.authority import infer_authority
from core.music_context.models import SongContext


class FakeAnalyzer:
    def analyze(self, asset_id: str) -> AudioAnalysis:
        return AudioAnalysis(asset_id=asset_id, role="drums", confidence={"rhythm": 0.9})


def test_analysis_batch():
    batch = analyze_assets(FakeAnalyzer(), ["drums", "guitar"])
    assert list(batch.by_asset()) == ["drums", "guitar"]


def test_authority_inference():
    result = infer_authority([FakeAnalyzer().analyze("drums")])
    assert result[0].dimension == "rhythm"
    assert result[0].confidence == 0.9


def test_four_candidates_share_parent_context():
    candidates = build_candidates(SongContext(version=7), "make it darker")
    assert len(candidates) == 4
    assert {c.parent_context_version for c in candidates} == {7}


def test_default_ep_does_not_mutate_context():
    context = SongContext(version=2)
    plan = DefaultExecutiveProducer().plan(ProductionRequest("make it darker", context))
    assert context.version == 2
    assert len(plan.candidates) == 4


def test_candidates_preserve_authority_for_requested_dimension():
    context = SongContext(
        version=3,
        authorities=[
            __import__("core.music_context.models", fromlist=["MusicalAuthority"]).MusicalAuthority(
                source_id="guitar-loop", dimension="harmony", confidence=0.97
            ),
            __import__("core.music_context.models", fromlist=["MusicalAuthority"]).MusicalAuthority(
                source_id="drum-loop", dimension="rhythm", confidence=0.91
            ),
        ],
    )
    candidate = build_candidates(context, "make a chord progression")
    constraints = candidate[0].parameter_changes["constraints"]
    assert "authority:harmony:guitar-loop:0.970" in constraints
    assert "authority:rhythm:drum-loop:0.910" not in constraints


def test_project_analysis_resolves_paths_and_preserves_logical_ids(tmp_path):
    from core.project.models import AudioAsset, MusicProject

    class ProjectAnalyzer:
        def __init__(self):
            self.paths = []

        def analyze(self, asset_id: str) -> AudioAnalysis:
            self.paths.append(asset_id)
            return AudioAnalysis(
                asset_id=asset_id,
                role="detected",
                confidence={"role": 0.4},
            )

    first = tmp_path / "drums.wav"
    second = tmp_path / "guitar.wav"
    project = MusicProject(
        id="analysis-project",
        context=SongContext(version=1),
        assets=[
            AudioAsset(id="drums-main", path=str(first), role_hint="drums"),
            AudioAsset(id="guitar-main", path=str(second), role_hint="guitar"),
        ],
    )
    analyzer = ProjectAnalyzer()
    batch = analyze_project_assets(analyzer, project)

    assert analyzer.paths == [str(first), str(second)]
    assert list(batch.by_asset()) == ["drums-main", "guitar-main"]
    assert batch.by_asset()["drums-main"].role == "drums"
    assert batch.by_asset()["drums-main"].confidence["role_hint"] == 1.0
