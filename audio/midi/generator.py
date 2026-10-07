"""Deterministic, role-aware MIDI generation for candidate previews."""
from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from audio.cache.keys import build_cache_key
from audio.midi.harmony import nearest_pitch, parse_chord, voice_chord
from audio.rendering.contracts import RenderRequest, RenderResult


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    start_beat: float
    duration_beats: float
    velocity: int = 80
    channel: int = 0


@dataclass(frozen=True)
class MidiSequence:
    notes: tuple[MidiNote, ...]
    tempo_bpm: float
    time_signature: tuple[int, int] = (4, 4)


_NOTE_NAMES = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A#": 10,
    "BB": 10, "B": 11,
}
_SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
}


def _key_pitch(key: str | None) -> int:
    if not key:
        return 60
    token = key.strip().upper().replace("♯", "#").replace("♭", "B").split()[0]
    return 60 + _NOTE_NAMES.get(token, 0)


def _scale(scale: str | None) -> tuple[int, ...]:
    name = (scale or "major").strip().lower().replace(" ", "_")
    return _SCALE_INTERVALS.get(name, _SCALE_INTERVALS["major"])


def _role(intent: str) -> str:
    text = intent.lower()
    if any(token in text for token in ("bass", "베이스", "sub bass", "low end", "저음")):
        return "bass"
    if any(token in text for token in ("chord", "pad", "harmony", "화음", "코드", "패드")):
        return "harmony"
    if any(token in text for token in ("drum", "rhythm", "groove", "드럼", "리듬", "그루브")):
        return "rhythm"
    return "melody"


def _groove_offset(direction: str, index: int) -> float:
    patterns = {
        "identity": (0.0, 0.0, 0.0, 0.0),
        "natural": (0.0, 0.02, 0.0, 0.01),
        "bold": (0.0, -0.04, 0.03, -0.02),
        "experimental": (0.0, 0.08, -0.03, 0.11),
    }
    return patterns.get(direction, patterns["identity"])[index % 4]


def _scale_pitch_classes(root: int, scale_name: str | None) -> tuple[int, ...]:
    return tuple((root + interval) % 12 for interval in _scale(scale_name))


def _nearest_scale_tone(pc: int, scale_pcs: tuple[int, ...], target: int) -> int:
    candidates = [n for n in range(target - 12, target + 25) if n % 12 in scale_pcs]
    return min(candidates, key=lambda n: (abs(n - target), n))



def _authority_data(changes: Mapping[str, object], dimension: str) -> Mapping[str, object] | None:
    raw = changes.get("authority_analysis", {})
    if not isinstance(raw, Mapping):
        return None
    for value in raw.values():
        if isinstance(value, Mapping) and str(value.get("dimension")) == dimension:
            return value
    return None


def _authority_chords(changes: Mapping[str, object]) -> tuple[str, ...]:
    data = _authority_data(changes, "harmony")
    return tuple(str(x) for x in data.get("chords", ())) if data else ()


def _authority_onsets(changes: Mapping[str, object]) -> tuple[float, ...]:
    data = _authority_data(changes, "rhythm")
    return tuple(float(x) for x in data.get("onset_beats", ())) if data else ()


def _authority_notes(changes: Mapping[str, object], dimension: str) -> tuple[int, ...]:
    data = _authority_data(changes, dimension)
    return tuple(int(x) for x in data.get("note_pitches", ())) if data else ()


def _melody_notes(chord_symbols: tuple[str, ...], root: int, scale_name: str | None, direction: str) -> list[MidiNote]:
    if not chord_symbols:
        scale = _scale(scale_name)
        degrees = [root + i for i in scale]
        pattern = (0, 1, 2, 4, 2, 1, 3, 4)
        return [MidiNote(degrees[d % 7] + 12, i * 0.5 + _groove_offset(direction, i), 0.5, 76) for i, d in enumerate(pattern)]
    patterns = {
        "identity": (0, 1, 2, 1, 0, 2, 1, 0),
        "natural": (0, 1, 2, 3, 2, 1, 2, 3),
        "bold": (0, 2, 3, 2, 4, 2, 3, 1),
        "experimental": (0, 2, 1, 3, 0, 4, 2, 1),
    }
    pattern = patterns.get(direction, patterns["identity"])
    notes: list[MidiNote] = []
    beat = 0.0
    previous = 64
    for i, index in enumerate(pattern):
        chord = parse_chord(chord_symbols[i % len(chord_symbols)])
        tones = chord.pitch_classes
        target = previous if direction in ("identity", "natural") else 67 + (i % 2) * 7
        chord_tone = nearest_pitch(tones[index % len(tones)], target)
        scale_pcs = _scale_pitch_classes(root, scale_name)
        pitch = chord_tone
        if direction == "natural" and i % 3 == 2:
            pitch = _nearest_scale_tone(chord_tone + (-1 if i % 2 else 1), scale_pcs, chord_tone)
        if direction == "experimental" and i in (3, 6):
            pitch += 1
        pitch = max(48, min(84, pitch))
        duration = 0.5 if direction != "identity" else 1.0
        notes.append(MidiNote(pitch, beat + _groove_offset(direction, i), duration, min(118, 76 + (i % 3) * 7)))
        beat += duration
        previous = pitch
    return notes


def _bass_notes(chord_symbols: tuple[str, ...], root: int, direction: str) -> list[MidiNote]:
    if not chord_symbols:
        return [MidiNote(root - 24, i * 1.0, 1.0, 82, channel=1) for i in range(8)]
    notes: list[MidiNote] = []
    for i, symbol in enumerate(chord_symbols[:8]):
        chord = parse_chord(symbol)
        pc = chord.bass_pc if chord.bass_pc is not None else chord.root_pc
        if direction == "bold" and i % 2:
            pc = chord.pitch_classes[2]
        elif direction == "experimental" and i % 4 == 3:
            pc = chord.pitch_classes[min(1, len(chord.pitch_classes) - 1)]
        pitch = nearest_pitch(pc, 36 + (i % 2) * 12)
        notes.append(MidiNote(pitch, float(i) + _groove_offset(direction, i), 0.9, 84 + (i % 3) * 4, channel=1))
    return notes


def _harmony_notes(chord_symbols: tuple[str, ...], direction: str) -> list[MidiNote]:
    notes: list[MidiNote] = []
    previous: tuple[int, ...] | None = None
    for i, symbol in enumerate(chord_symbols[:8]):
        chord = parse_chord(symbol)
        center = 60 if direction in ("identity", "natural") else 64
        if direction == "bold":
            center += 7
        if direction == "experimental" and i % 2:
            center += 5
        target = voice_chord(chord, center=center, spread=3)
        if previous is not None and len(previous) == len(target):
            options = [tuple(p + shift for p in target) for shift in (-12, 0, 12)]
            voicing = min(options, key=lambda v: sum(abs(a - b) for a, b in zip(previous, v)))
        else:
            voicing = target
        for pitch in voicing:
            notes.append(MidiNote(pitch, float(i) + _groove_offset(direction, i), 0.9, 70 + (i % 2) * 8, channel=2))
        previous = voicing
    return notes

def _rhythm_notes(direction: str, authority_onsets: tuple[float, ...] = ()) -> list[MidiNote]:
    starts = authority_onsets[:16] if authority_onsets else {
        "identity": (0.0, 1.0, 2.0, 3.0),
        "natural": (0.0, 0.5, 1.5, 2.0, 3.0),
        "bold": (0.0, 0.75, 1.5, 2.75, 3.5),
        "experimental": (0.0, 0.75, 1.75, 2.25, 3.25),
    }.get(direction, (0.0, 1.0, 2.0, 3.0))
    return [MidiNote(36, start, 0.2, 92, channel=9) for start in starts]


def _role_from_constraints(changes: Mapping[str, object]) -> str | None:
    raw = changes.get("constraints", ())
    for item in raw:
        text = str(item)
        if text.startswith("role:"):
            dimension = text.split(":", 1)[1]
            return {"low_end": "bass", "rhythm": "rhythm", "harmony": "harmony", "melody": "melody"}.get(dimension, "melody")
    return None


def _authority_ids(changes: Mapping[str, object], role: str) -> tuple[str, ...]:
    dimension = {"bass": "low_end", "rhythm": "rhythm", "harmony": "harmony", "melody": "melody"}.get(role, role)
    prefix = f"authority:{dimension}:"
    raw = changes.get("constraints", ())
    return tuple(str(item) for item in raw if str(item).startswith(prefix))


def _apply_authority_register(notes: list[MidiNote], role: str) -> list[MidiNote]:
    if role == "melody":
        return [MidiNote(max(60, note.pitch), note.start_beat, note.duration_beats, note.velocity, note.channel) for note in notes]
    if role == "bass":
        return [MidiNote(min(55, note.pitch), note.start_beat, note.duration_beats, note.velocity, note.channel) for note in notes]
    return notes


def generate_sequence(request: RenderRequest) -> MidiSequence:
    changes: Mapping[str, object] = request.parameter_changes
    root = _key_pitch(str(changes["key"])) if "key" in changes else _key_pitch(None)
    direction = str(changes.get("direction", "identity"))
    bpm = float(changes.get("bpm", 120.0))
    chords = tuple(str(x) for x in changes.get("chord_progression", ()))
    authority_harmony = _authority_chords(changes)
    if authority_harmony:
        chords = authority_harmony
    scale_name = str(changes.get("scale", "major"))
    role = _role_from_constraints(changes) or _role(request.intent)
    if role == "bass":
        notes = _bass_notes(chords, root, direction)
        # A low-end authority owns the foundation; follow its observed pitches
        # when available instead of inventing a competing root movement.
        authority_bass = _authority_notes(changes, "low_end")
        if authority_bass:
            notes = [
                MidiNote(min(55, p), float(i), 0.9, 84 + (i % 3) * 4, channel=1)
                for i, p in enumerate(authority_bass[:8])
            ]
    elif role == "harmony":
        notes = _harmony_notes(chords, direction)
    elif role == "rhythm":
        notes = _rhythm_notes(direction, _authority_onsets(changes))
    else:
        notes = _melody_notes(chords, root, scale_name, direction)
        # Melody authority is the lead: generated accompaniment stays outside
        # its observed register and pitch classes rather than competing with it.
        authority_notes = _authority_notes(changes, "melody")
        if authority_notes:
            low = min(authority_notes)
            notes = [
                MidiNote(min(n.pitch, low - 1), n.start_beat, n.duration_beats, n.velocity, n.channel)
                for n in notes
            ]
    notes = _apply_authority_register(notes, role)
    return MidiSequence(tuple(notes), bpm)


def _vlq(value: int) -> bytes:
    value = max(0, value)
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def write_midi(path: Path, sequence: MidiSequence, ticks_per_beat: int = 480) -> None:
    events: list[tuple[int, int, bytes]] = []
    tempo = max(1, round(60_000_000 / max(1.0, sequence.tempo_bpm)))
    events.append((0, 0, bytes((0xFF, 0x51, 0x03)) + tempo.to_bytes(3, "big")))
    numerator, denominator = sequence.time_signature
    events.append((0, 0, bytes((0xFF, 0x58, 0x04, numerator, denominator.bit_length() - 1, 24, 8))))
    for note in sequence.notes:
        channel = note.channel & 0x0F
        pitch = max(0, min(127, note.pitch))
        velocity = max(1, min(127, note.velocity))
        events.append((round(note.start_beat * ticks_per_beat), 2, bytes((0x90 | channel, pitch, velocity))))
        events.append((round((note.start_beat + note.duration_beats) * ticks_per_beat), 1, bytes((0x80 | channel, pitch, 0))))
    events.sort(key=lambda x: (x[0], x[1]))
    track = bytearray()
    previous = 0
    for tick, _, message in events:
        track.extend(_vlq(tick - previous))
        track.extend(message)
        previous = tick
    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    body = b"MTrk" + struct.pack(">I", len(track)) + track
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)


class DeterministicMidiGenerator:
    """Generate context-aware MIDI without an external MIDI dependency."""

    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)

    def render(self, request: RenderRequest) -> RenderResult:
        if request.kind != "midi":
            raise ValueError("DeterministicMidiGenerator only renders MIDI requests.")
        cache_key = build_cache_key(
            request.candidate_id, request.context_version, request.kind, request.parameter_changes
        )
        output = self.cache_dir / f"{cache_key}.mid"
        sequence = generate_sequence(request)
        if not output.exists():
            write_midi(output, sequence)
        return RenderResult(
            candidate_id=request.candidate_id,
            kind="midi",
            artifact_ref=str(output),
            cache_key=cache_key,
            duration_seconds=sum(n.duration_beats for n in sequence.notes) * 60.0 / sequence.tempo_bpm,
            metadata={"renderer": "deterministic-v0.3", "note_count": len(sequence.notes), "role": _role_from_constraints(request.parameter_changes) or _role(request.intent), "authorities": _authority_ids(request.parameter_changes, _role_from_constraints(request.parameter_changes) or _role(request.intent))},
        )
