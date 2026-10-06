import math
import wave
from pathlib import Path
import numpy as np
from core.analysis.wav import WavAnalyzer

def write_tone(path: Path, hz: float, seconds: float = 1.0, sr: int = 44100):
    t = np.arange(int(sr * seconds)) / sr
    samples = (0.5 * np.sin(2 * math.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(samples.tobytes())

def test_wav_analysis_extracts_pitch(tmp_path):
    path = tmp_path / "a.wav"
    write_tone(path, 440.0, 1.0)
    result = WavAnalyzer().analyze(str(path))
    assert result.sample_rate == 44100
    assert abs(result.fundamental_hz - 440.0) < 2.0
    assert result.pitch_confidence > .1
    assert result.peak > .4
    assert result.duration_seconds == 1.0

def test_wav_analysis_extracts_frequency_bands(tmp_path):
    path = tmp_path / "bass.wav"
    write_tone(path, 110.0, 1.0)
    result = WavAnalyzer().analyze(str(path))
    assert result.low_energy_ratio > .5
    assert result.spectral_centroid_hz < 500

def test_wav_analysis_extracts_envelope_features(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, 440.0, 1.0)
    result = WavAnalyzer().analyze(str(path))
    assert result.transient_ratio >= 0
    assert result.attack_seconds >= 0
    assert result.release_seconds >= 0


def test_event_fields_exist_and_are_aligned(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, 440.0, 1.0)
    result = WavAnalyzer().analyze(str(path))
    assert isinstance(result.onset_beats, tuple)
    assert isinstance(result.note_pitches, tuple)
    assert isinstance(result.note_durations_beats, tuple)
    assert len(result.note_pitches) == len(result.note_durations_beats)


def test_chord_extractor_labels_simple_harmony(tmp_path):
    sr = 44100
    bpm = 120.0
    beat = 60.0 / bpm
    length = int(sr * beat * 4)
    t = np.arange(length) / sr
    signal = np.zeros(length, dtype=np.float64)
    for hz in (261.63, 329.63, 392.00):
        signal += 0.3 * np.sin(2 * math.pi * hz * t)
    chords, confidence = WavAnalyzer._chords(signal, sr, bpm)
    assert chords
    assert chords[0].startswith("C")
    assert confidence >= 0
