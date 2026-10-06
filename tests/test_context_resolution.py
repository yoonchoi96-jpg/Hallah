from core.analysis.contracts import AudioAnalysis
from core.music_context.models import MusicalAuthority, SongContext
from core.music_context.resolve import detect_conflicts, resolve_authorities

def test_authority_resolution_prefers_confidence():
    items=[MusicalAuthority("a","harmony",.4),MusicalAuthority("b","harmony",.9)]
    result=resolve_authorities(items)[0]
    assert result.winner and result.winner.source_id=="b"
    assert result.alternatives[0].source_id=="a"

def test_conflicts_detected():
    analyses=[
        AudioAnalysis("a",bpm=120,key="C",scale="major"),
        AudioAnalysis("b",bpm=128,key="F",scale="major"),
    ]
    conflicts=detect_conflicts(analyses)
    assert len(conflicts)==2

def test_context_version_is_non_destructive():
    ctx=SongContext(bpm=120)
    new=ctx.record_decision("Keep drums unchanged.")
    assert ctx.version==0 and new.version==1
    assert new.bpm==120
