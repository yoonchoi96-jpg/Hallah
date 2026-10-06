"""Deterministic, dependency-free MIDI generation for candidate previews."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Mapping
from audio.cache.keys import build_cache_key
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

_NOTE_NAMES = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,"G":7,"G#":8,"AB":8,"A#":10,"BB":10,"B":11}
_SCALE_INTERVALS = {"major":(0,2,4,5,7,9,11),"minor":(0,2,3,5,7,8,10),"natural_minor":(0,2,3,5,7,8,10)}

def _key_pitch(key: str | None) -> int:
    if not key: return 60
    token = key.strip().upper().replace("♯","#").replace("♭","B")
    return 60 + _NOTE_NAMES.get(token.split()[0], 0)

def _scale(scale: str | None) -> tuple[int, ...]:
    return _SCALE_INTERVALS.get((scale or "major").strip().lower().replace(" ","_"), _SCALE_INTERVALS["major"])

def _chord_root(chord: str, fallback: int) -> int:
    token = chord.strip().upper().replace("♯", "#").replace("♭", "B")
    return fallback + _NOTE_NAMES.get(token.split(":")[0].split("/")[0].rstrip("0123456789"), 0)


def generate_sequence(request: RenderRequest) -> MidiSequence:
    changes: Mapping[str, object] = request.parameter_changes
    root = _key_pitch(str(changes["key"])) if "key" in changes else _key_pitch(None)
    degrees = [root + interval for interval in _scale(str(changes.get("scale", "major")))]
    direction = str(changes.get("direction", "identity"))
    bpm = float(changes.get("bpm", 120.0))
    chords = tuple(str(x) for x in changes.get("chord_progression", ()))
    patterns = {
        "identity": ((0, 2, 4, 0, 4, 2, 0, 0), (1.0,) * 8),
        "natural": ((0, 1, 2, 4, 2, 1, 3, 4), (0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)),
        "bold": ((0, 4, 6, 4, 7, 4, 6, 2), (0.5,) * 8),
        "experimental": ((0, 3, 1, 5, 2, 6, 4, 1), (0.75, 0.25, 0.5, 0.5, 0.75, 0.25, 0.5, 0.5)),
    }
    pattern, lengths = patterns.get(direction, patterns["identity"])
    notes = []
    beat = 0.0
    for i, degree in enumerate(pattern):
        if chords:
            chord_root = _chord_root(chords[i % len(chords)], root)
            chord_intervals = (0, 4, 7) if "MIN" not in chords[i % len(chords)].upper() else (0, 3, 7)
            pitch = chord_root + chord_intervals[(i + (1 if direction == "bold" else 0)) % 3]
        else:
            pitch = degrees[degree % len(degrees)] + 12 * (degree // len(degrees))
        if direction == "experimental" and i in (3, 6):
            pitch += 1
        notes.append(MidiNote(pitch, beat, lengths[i], min(120, 72 + (i % 4) * 8 + (10 if direction == "bold" else 0))))
        beat += lengths[i]
    return MidiSequence(tuple(notes), bpm)

def _vlq(value: int) -> bytes:
    value = max(0,value)
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0,(value & 0x7F)|0x80)
        value >>= 7
    return bytes(out)

def write_midi(path: Path, sequence: MidiSequence, ticks_per_beat: int = 480) -> None:
    events = []
    tempo = max(1,round(60_000_000/max(1.0,sequence.tempo_bpm)))
    events.append((0,0,bytes((0xFF,0x51,0x03))+tempo.to_bytes(3,"big")))
    numerator, denominator = sequence.time_signature
    events.append((0,0,bytes((0xFF,0x58,0x04,numerator,denominator.bit_length()-1,24,8))))
    for note in sequence.notes:
        channel = note.channel & 0x0F
        pitch = max(0,min(127,note.pitch))
        velocity = max(1,min(127,note.velocity))
        events.append((round(note.start_beat*ticks_per_beat),2,bytes((0x90|channel,pitch,velocity))))
        events.append((round((note.start_beat+note.duration_beats)*ticks_per_beat),1,bytes((0x80|channel,pitch,0))))
    events.sort(key=lambda x:(x[0],x[1]))
    track = bytearray()
    previous = 0
    for tick,_,message in events:
        track.extend(_vlq(tick-previous)); track.extend(message); previous=tick
    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd"+struct.pack(">IHHH",6,0,1,ticks_per_beat)
    body = b"MTrk"+struct.pack(">I",len(track))+track
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(header+body)

class DeterministicMidiGenerator:
    """Generate context-aware MIDI without an external MIDI dependency."""
    def __init__(self, cache_dir: str | Path = ".hallah-cache") -> None:
        self.cache_dir = Path(cache_dir)
    def render(self, request: RenderRequest) -> RenderResult:
        if request.kind != "midi": raise ValueError("DeterministicMidiGenerator only renders MIDI requests.")
        cache_key = build_cache_key(request.candidate_id,request.context_version,request.kind,request.parameter_changes)
        output = self.cache_dir/f"{cache_key}.mid"
        sequence = generate_sequence(request)
        if not output.exists(): write_midi(output,sequence)
        return RenderResult(candidate_id=request.candidate_id,kind="midi",artifact_ref=str(output),cache_key=cache_key,
            duration_seconds=sum(n.duration_beats for n in sequence.notes)*60.0/sequence.tempo_bpm,
            metadata={"renderer":"deterministic-v0","note_count":len(sequence.notes)})
