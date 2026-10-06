from agent.executive_producer.default import DefaultExecutiveProducer
from agent.executive_producer.contracts import ProductionRequest
from core.analysis.contracts import AudioAnalysis
from core.analysis.engine import analyze_assets
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
