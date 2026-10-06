from core.analysis.contracts import AudioAnalysis
from core.music_context.builder import build_song_context

def test_build_song_context():
    analyses = [
        AudioAnalysis("drums.wav", bpm=120, key="C", scale="major", role="drums",
                      confidence={"key": .2, "role": .8}),
        AudioAnalysis("guitar.wav", bpm=124, key="A", scale="minor", role="guitar/piano",
                      confidence={"key": .9, "role": .8}),
    ]
    ctx = build_song_context(analyses, "Demo")
    assert ctx.title == "Demo"
    assert ctx.bpm == 122
    assert (ctx.key, ctx.scale) == ("A", "minor")
    dims = {(a.source_id, a.dimension) for a in ctx.authorities}
    assert ("drums.wav", "rhythm") in dims
    assert ("guitar.wav", "harmony") in dims
