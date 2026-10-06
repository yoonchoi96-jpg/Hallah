from core.analysis.contracts import AudioAnalysis
from core.music_context.models import MusicalAuthority
from core.music_context.relationships import (
    analyze_relationships,
    classify_bpm,
    classify_tonality,
)


def test_bpm_relationships_distinguish_match_double_time_and_conflict():
    assert classify_bpm(120, 121).kind == "match"
    assert classify_bpm(120, 60).kind == "double_time"
    assert classify_bpm(120, 240).kind == "double_time"
    assert classify_bpm(120, 128).kind == "conflict"


def test_tonal_relationships_cover_common_compatibility():
    assert classify_tonality("C", "major", "C", "major").kind == "match"
    assert classify_tonality("C", "major", "A", "minor").kind == "relative"
    assert classify_tonality("C", "major", "F", "major").kind == "compatible"
    assert classify_tonality("C", "major", "F#", "major").kind == "conflict"


def test_authority_drives_adaptation_constraints():
    analyses = [
        AudioAnalysis("drums", bpm=120, role="drums", confidence={"role": .9}),
        AudioAnalysis("guitar", bpm=124, role="guitar/piano", confidence={"role": .8}),
        AudioAnalysis("bass", bpm=120, role="bass", confidence={"role": .8}),
    ]
    authorities = [
        MusicalAuthority("drums", "rhythm", .9),
        MusicalAuthority("guitar", "harmony", .8),
        MusicalAuthority("bass", "low_end", .8),
    ]
    relationships, constraints, conflicts = analyze_relationships(analyses, authorities)
    assert any(r.type == "authority" and r.dimension == "rhythm" for r in relationships)
    assert any(c.type == "adapt" and c.dimension == "rhythm" and c.reference_id == "drums" for c in constraints)
    assert any(c.type == "avoid" and c.dimension == "low_end" and c.reference_id == "bass" for c in constraints)
    assert any("BPM disagreement" in c for c in conflicts)


def test_vocal_melody_is_protected():
    analyses = [
        AudioAnalysis("vocal", role="vocal/lead", confidence={"role": .9}),
        AudioAnalysis("guitar", role="guitar/piano", confidence={"role": .8}),
    ]
    authorities = [
        MusicalAuthority("vocal", "melody", .9),
    ]
    _, constraints, _ = analyze_relationships(analyses, authorities)
    assert any(c.type == "avoid" and c.dimension == "melody" and c.reference_id == "vocal" for c in constraints)
