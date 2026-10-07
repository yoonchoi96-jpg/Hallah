"""Deterministic chord-symbol parsing and voicing helpers for Hallah MIDI."""
from __future__ import annotations

import re
from dataclasses import dataclass

NOTE_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

@dataclass(frozen=True)
class ChordSymbol:
    root_pc: int
    quality: str = "major"
    bass_pc: int | None = None

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        intervals = {
            "major": (0, 4, 7), "minor": (0, 3, 7),
            "dominant7": (0, 4, 7, 10), "major7": (0, 4, 7, 11),
            "minor7": (0, 3, 7, 10), "dim": (0, 3, 6),
            "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
            "add9": (0, 4, 7, 14),
        }
        return tuple((self.root_pc + i) % 12 for i in intervals.get(self.quality, intervals["major"]))

_ROOT = re.compile(r"^([A-Ga-g])([#♯b♭]?)(.*)$")

def parse_chord(symbol: str) -> ChordSymbol:
    text = symbol.strip()
    if not text:
        raise ValueError("Chord symbol cannot be empty.")
    root_match = _ROOT.match(text)
    if not root_match:
        raise ValueError(f"Unsupported chord symbol: {symbol!r}")
    letter, accidental, suffix = root_match.groups()
    root_token = letter.upper() + accidental.replace("♯", "#").replace("♭", "b").upper()
    root_pc = NOTE_TO_PC[root_token]
    bass_pc = None
    if "/" in suffix:
        suffix, bass = suffix.split("/", 1)
        bass_match = _ROOT.match(bass.strip())
        if not bass_match:
            raise ValueError(f"Invalid slash bass in {symbol!r}")
        bletter, bacc, brest = bass_match.groups()
        if brest:
            raise ValueError(f"Invalid slash bass in {symbol!r}")
        bass_pc = NOTE_TO_PC[bletter.upper() + bacc.replace("♯", "#").replace("♭", "b").upper()]
    normalized = suffix.lower().replace(" ", "")
    if normalized in ("", "maj", "major"):
        quality = "major"
    elif normalized.startswith(("min", "m")) and not normalized.startswith("maj"):
        quality = "minor7" if normalized in ("m7", "min7", "minor7") else "minor"
    elif normalized in ("7", "dom7", "dominant7"):
        quality = "dominant7"
    elif normalized in ("maj7", "major7", "Δ7"):
        quality = "major7"
    elif normalized in ("dim", "o", "°"):
        quality = "dim"
    elif normalized in ("aug", "+"):
        quality = "aug"
    elif normalized in ("sus2",):
        quality = "sus2"
    elif normalized in ("sus", "sus4"):
        quality = "sus4"
    elif normalized in ("add9",):
        quality = "add9"
    else:
        raise ValueError(f"Unsupported chord quality: {symbol!r}")
    return ChordSymbol(root_pc=root_pc, quality=quality, bass_pc=bass_pc)

def nearest_pitch(pc: int, target: int) -> int:
    """Return the pitch of pc nearest to target, preferring the lower octave on ties."""
    base = target - ((target - pc) % 12)
    candidates = (base, base + 12)
    return min(candidates, key=lambda p: (abs(p - target), p))

def voice_chord(chord: ChordSymbol, center: int = 60, spread: int = 4) -> tuple[int, ...]:
    tones = list(chord.pitch_classes)
    if chord.bass_pc is not None:
        tones = [chord.bass_pc] + [pc for pc in tones if pc != chord.bass_pc]
    notes: list[int] = []
    target = center
    for index, pc in enumerate(tones):
        note = nearest_pitch(pc, target)
        while notes and note <= notes[-1]:
            note += 12
        notes.append(note)
        target += spread
    return tuple(notes)
