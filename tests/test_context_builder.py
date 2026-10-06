from core.analysis.contracts import AudioAnalysis
from core.music_context.builder import build_song_context

def test_build_song_context_uses_dimension_authority():
    analyses = [
        AudioAnalysis("drums.wav", bpm=120, key="C", scale="major", role="drums",
                      confidence={"bpm": .95, "key": .2, "role": .8}),
        AudioAnalysis("guitar.wav", bpm=124, key="A", scale="minor", role="guitar/piano",
                      confidence={"bpm": .5, "key": .9, "role": .8}),
    ]
    ctx = build_song_context(analyses, "Demo")
    assert ctx.title == "Demo"
    assert ctx.bpm == 120
    assert (ctx.key, ctx.scale) == ("A", "minor")
    assert ctx.conflicts == [
        "BPM disagreement: 120.00–124.00 BPM.",
        "Tonal disagreement: analyzed assets suggest different key/scale.",
    ]
    assert ctx.pending_decisions == ctx.conflicts
    dims = {(a.source_id, a.dimension) for a in ctx.authorities}
    assert ("drums.wav", "rhythm") in dims
    assert ("guitar.wav", "harmony") in dims

def test_build_song_context_falls_back_to_strongest_measurement():
    analyses = [AudioAnalysis("a.wav", bpm=100, confidence={"bpm": .4}),
                AudioAnalysis("b.wav", bpm=102, confidence={"bpm": .9})]
    ctx = build_song_context(analyses)
    assert ctx.bpm == 102
    assert ctx.conflicts == []
