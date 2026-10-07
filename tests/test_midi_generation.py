from pathlib import Path

from audio.midi.generator import DeterministicMidiGenerator, generate_sequence
from audio.midi.harmony import parse_chord, voice_chord
from audio.rendering.candidates import preview_candidates
from audio.rendering.pipeline import candidate_to_render_request
from core.candidates.engine import build_candidates
from core.music_context.models import SongContext


def test_midi_generator_writes_standard_midi(tmp_path: Path) -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major")
    candidate = build_candidates(context, "create a melody")[0]
    request = candidate_to_render_request(candidate, context, kind="midi")
    result = DeterministicMidiGenerator(tmp_path).render(request)
    data = Path(result.artifact_ref).read_bytes()
    assert data[:4] == b"MThd"
    assert data[14:18] == b"MTrk"


def test_chord_parser_supports_common_symbols_and_slash_bass() -> None:
    assert parse_chord("C").pitch_classes == (0, 4, 7)
    assert parse_chord("Cm7").pitch_classes == (0, 3, 7, 10)
    assert parse_chord("G7").pitch_classes == (7, 11, 2, 5)
    chord = parse_chord("Cmaj7/E")
    assert chord.root_pc == 0
    assert chord.bass_pc == 4
    assert chord.pitch_classes == (0, 4, 7, 11)


def test_voice_chord_is_ascending_and_near_center() -> None:
    voicing = voice_chord(parse_chord("F#m7"), center=60)
    assert list(voicing) == sorted(voicing)
    assert 55 <= voicing[0] <= 70


def test_role_aware_generation_uses_chords_and_registers() -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major", chord_progression=["C", "Am", "F", "G7"])
    bass = build_candidates(context, "make a bass line")[0]
    melody = build_candidates(context, "make a melody")[0]
    bass_seq = generate_sequence(candidate_to_render_request(bass, context, kind="midi"))
    melody_seq = generate_sequence(candidate_to_render_request(melody, context, kind="midi"))
    assert max(n.pitch for n in bass_seq.notes) < min(n.pitch for n in melody_seq.notes)
    for note, chord_name in zip(melody_seq.notes, ("C", "Am", "F", "G7")):
        assert note.pitch % 12 in parse_chord(chord_name).pitch_classes


def test_four_directions_produce_distinct_midi_cache_keys(tmp_path: Path) -> None:
    context = SongContext(version=0, bpm=128, key="A", scale="minor", chord_progression=["Am", "F", "C", "G"])
    candidates = build_candidates(context, "make a bass idea")
    generator = DeterministicMidiGenerator(tmp_path)
    results = [generator.render(candidate_to_render_request(c, context, kind="midi")) for c in candidates]
    assert len({result.cache_key for result in results}) == 4
    assert all(Path(result.artifact_ref).exists() for result in results)


def test_preview_candidates_attaches_midi_without_advancing_context(tmp_path: Path) -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major")
    candidates = build_candidates(context, "make a melody")
    previewed = preview_candidates(context, candidates, DeterministicMidiGenerator(tmp_path))
    assert previewed.version == 0
    assert all(candidate.status == "preview_ready" for candidate in previewed.candidates)
    assert all(candidate.midi_refs for candidate in previewed.candidates)
    assert len(previewed.candidate_history) == 4


def test_natural_direction_adds_scale_passing_tones_and_groove() -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major", chord_progression=["C", "F", "G", "C"])
    candidate = build_candidates(context, "make a melody")[1]
    sequence = generate_sequence(candidate_to_render_request(candidate, context, kind="midi"))
    assert any(note.start_beat != round(note.start_beat) for note in sequence.notes)
    scale_pcs = {0, 2, 4, 5, 7, 9, 11}
    assert all(note.pitch % 12 in scale_pcs for note in sequence.notes)


def test_experimental_direction_has_distinct_timing() -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major", chord_progression=["C", "Am", "F", "G"])
    candidates = build_candidates(context, "make a melody")
    identity = generate_sequence(candidate_to_render_request(candidates[0], context, kind="midi"))
    experimental = generate_sequence(candidate_to_render_request(candidates[3], context, kind="midi"))
    assert tuple(n.start_beat for n in identity.notes) != tuple(n.start_beat for n in experimental.notes)


def test_generic_candidate_inherits_strongest_musical_authority() -> None:
    from core.music_context.models import MusicalAuthority

    context = SongContext(
        version=0,
        bpm=120,
        key="C",
        scale="major",
        chord_progression=["C", "Am", "F", "G"],
        authorities=[
            MusicalAuthority("guitar-loop", "harmony", 0.97),
            MusicalAuthority("drum-loop", "rhythm", 0.91),
        ],
    )
    candidate = build_candidates(context, "make something")[0]
    request = candidate_to_render_request(candidate, context, kind="midi")
    sequence = generate_sequence(request)
    assert "role:harmony" in request.parameter_changes["constraints"]
    assert all(note.channel == 2 for note in sequence.notes)


def test_harmony_voice_leading_avoids_large_octave_jumps() -> None:
    context = SongContext(
        version=0,
        bpm=120,
        key="C",
        scale="major",
        chord_progression=["C", "G", "Am", "F", "C", "G", "Am", "F"],
    )
    candidate = build_candidates(context, "make chord harmony")[0]
    sequence = generate_sequence(candidate_to_render_request(candidate, context, kind="midi"))
    first = [n.pitch for n in sequence.notes[:3]]
    second = [n.pitch for n in sequence.notes[3:6]]
    assert max(abs(a - b) for a, b in zip(first, second)) <= 12



def test_harmony_authority_uses_source_chords() -> None:
    from core.analysis.contracts import AudioAnalysis
    from core.music_context.models import MusicalAuthority
    context = SongContext(
        version=0, bpm=120, key="C", scale="major",
        chord_progression=["C", "F", "G", "C"],
        authorities=[MusicalAuthority("guitar", "harmony", .99)],
        analyses={"guitar": AudioAnalysis("guitar", chords=("Dm7", "G7", "Cmaj7"))},
    )
    candidate = build_candidates(context, "make chord harmony")[0]
    request = candidate_to_render_request(candidate, context, kind="midi")
    sequence = generate_sequence(request)
    assert tuple(n.pitch % 12 for n in sequence.notes[:4]) == parse_chord("Dm7").pitch_classes


def test_rhythm_authority_uses_source_onsets() -> None:
    from core.analysis.contracts import AudioAnalysis
    from core.music_context.models import MusicalAuthority
    context = SongContext(
        version=0, bpm=120,
        authorities=[MusicalAuthority("drums", "rhythm", .99)],
        analyses={"drums": AudioAnalysis("drums", onset_beats=(0.0, .75, 1.5, 2.75))},
    )
    candidate = build_candidates(context, "make a rhythm")[0]
    sequence = generate_sequence(candidate_to_render_request(candidate, context, kind="midi"))
    assert tuple(n.start_beat for n in sequence.notes) == (0.0, .75, 1.5, 2.75)


def test_melody_authority_pushes_generated_line_out_of_lead_register() -> None:
    from core.analysis.contracts import AudioAnalysis
    from core.music_context.models import MusicalAuthority
    context = SongContext(
        version=0,
        bpm=120,
        key="C",
        scale="major",
        chord_progression=["C", "Am", "F", "G"],
        authorities=[MusicalAuthority("vocal", "melody", .99)],
        analyses={"vocal": AudioAnalysis("vocal", note_pitches=(72, 74, 76, 77))},
    )
    candidate = build_candidates(context, "make something")[0]
    sequence = generate_sequence(candidate_to_render_request(candidate, context, kind="midi"))
    assert all(note.pitch < 72 for note in sequence.notes)


def test_low_end_authority_pitches_drive_bass() -> None:
    from core.analysis.contracts import AudioAnalysis
    from core.music_context.models import MusicalAuthority
    context = SongContext(
        version=0,
        bpm=120,
        key="C",
        scale="major",
        chord_progression=["C", "F", "G", "C"],
        authorities=[MusicalAuthority("bass", "low_end", .99)],
        analyses={"bass": AudioAnalysis("bass", note_pitches=(36, 40, 43, 36))},
    )
    candidate = build_candidates(context, "make a bass line")[0]
    sequence = generate_sequence(candidate_to_render_request(candidate, context, kind="midi"))
    assert tuple(n.pitch for n in sequence.notes) == (36, 40, 43, 36)
