from pathlib import Path

from audio.midi.generator import DeterministicMidiGenerator
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


def test_four_directions_produce_distinct_midi_cache_keys(tmp_path: Path) -> None:
    context = SongContext(version=0, bpm=128, key="A", scale="minor")
    candidates = build_candidates(context, "make a bass idea")
    generator = DeterministicMidiGenerator(tmp_path)
    results = [generator.render(candidate_to_render_request(c, context, kind="midi")) for c in candidates]
    assert len({result.cache_key for result in results}) == 4
    assert all(Path(result.artifact_ref).exists() for result in results)


def test_preview_candidates_attaches_midi_and_advances_context(tmp_path: Path) -> None:
    context = SongContext(version=0, bpm=120, key="C", scale="major")
    candidates = build_candidates(context, "make a melody")
    previewed = preview_candidates(context, candidates, DeterministicMidiGenerator(tmp_path))
    assert previewed.version == 1
    assert all(candidate.status == "preview_ready" for candidate in previewed.candidates)
    assert all(candidate.midi_refs for candidate in previewed.candidates)
    assert len(previewed.candidate_history) == 4
