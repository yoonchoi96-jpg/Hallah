from core.analysis.contracts import AudioAnalysis
from core.music_context.authority import infer_authority
from core.music_context.resolve import build_relationships, detect_conflicts


def test_authority_relationships_are_dimension_specific():
    analyses = [
        AudioAnalysis("drums.wav", bpm=120, role="drums", confidence={"role": .9}),
        AudioAnalysis("guitar.wav", bpm=124, role="guitar/piano", confidence={"role": .8}),
        AudioAnalysis("bass.wav", bpm=120, role="bass", confidence={"role": .85}),
        AudioAnalysis("vocal.wav", bpm=120, role="vocal/lead", confidence={"role": .8}),
    ]
    relationships = build_relationships(analyses, infer_authority(analyses))
    assert any(r.source_id == "drums.wav" and r.dimension == "rhythm" for r in relationships)
    assert any(r.source_id == "guitar.wav" and r.dimension == "harmony" for r in relationships)
    assert any(r.source_id == "bass.wav" and r.dimension == "low_end" for r in relationships)
    assert any(r.source_id == "vocal.wav" and r.dimension == "melody" for r in relationships)

def test_conflicting_bpm_is_explicit():
    analyses = [AudioAnalysis("a.wav", bpm=100), AudioAnalysis("b.wav", bpm=106)]
    assert detect_conflicts(analyses) == ["BPM disagreement: 100.00–106.00 BPM."]
