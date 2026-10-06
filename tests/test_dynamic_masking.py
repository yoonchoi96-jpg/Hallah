from audio.processing.dynamic_masking import apply_dynamic_masking, build_dynamic_envelope
from audio.processing.masking import build_relationship_map


def test_fundamental_and_centroid_change_masking_ranges():
    authority = {
        "bass": {
            "role": "bass", "dimension": "low_end",
            "fundamental_hz": 55.0, "spectral_centroid_hz": 420.0,
            "low_energy_ratio": 0.60, "mid_energy_ratio": 0.25, "high_energy_ratio": 0.05,
        },
        "kick": {
            "role": "kick", "dimension": "rhythm",
            "fundamental_hz": 48.0, "spectral_centroid_hz": 1800.0,
            "low_energy_ratio": 0.65, "mid_energy_ratio": 0.25, "high_energy_ratio": 0.10,
        },
    }
    resolved = build_relationship_map(authority)
    decision = resolved["bass"]["decisions"][0]
    assert decision["reference_id"] == "kick"
    assert decision["ranges"]["low"][0] > 20.0
    assert decision["ranges"]["low"][1] <= 220.0


def test_kick_event_ducks_bass_only_near_event():
    sr = 1000
    data = __import__("numpy").ones((2000, 1), dtype="float32")
    bass = {"role": "bass", "dimension": "low_end", "bpm": 60.0}
    kick = {"role": "kick", "dimension": "rhythm", "bpm": 60.0, "onset_beats": (0.5,)}
    rendered, meta = apply_dynamic_masking(data, sr, bass, kick, 0.25)
    assert meta["applied"] is True
    assert meta["events"] == 1
    assert rendered[500, 0] < 0.8
    assert rendered[1500, 0] > 0.99


def test_vocal_note_duration_holds_ducking_then_recovers():
    sr = 1000
    data = __import__("numpy").ones((3000, 1), dtype="float32")
    guitar = {"role": "guitar", "dimension": "harmony", "bpm": 60.0}
    vocal = {
        "role": "vocal", "dimension": "melody", "bpm": 60.0,
        "onset_beats": (0.5,), "note_durations_beats": (0.8,),
    }
    rendered, meta = apply_dynamic_masking(data, sr, guitar, vocal, 0.30)
    assert meta["applied"] is True
    assert rendered[600, 0] < 0.75
    assert rendered[1200, 0] < 0.75
    assert rendered[1600, 0] > 0.98


def test_dynamic_masking_has_ramped_edges():
    envelope, meta = build_dynamic_envelope(
        2000, 1000, {"role": "bass", "bpm": 60.0},
        {"role": "kick", "bpm": 60.0, "onset_beats": (0.5,)}, 0.25,
    )
    assert meta["applied"] is True
    assert envelope[490] > envelope[500]
    assert envelope[620] < envelope[700]
    assert __import__("numpy").max(__import__("numpy").abs(__import__("numpy").diff(envelope))) < 0.03


def test_frequency_dynamic_masking_ducks_only_selected_band():
    sr = 8000
    t = np.arange(4000) / sr
    data = np.column_stack([
        0.4 * np.sin(2 * np.pi * 80 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t),
    ]).astype("float32")
    bass = {"role": "bass", "dimension": "low_end", "bpm": 60.0}
    kick = {"role": "kick", "dimension": "rhythm", "bpm": 60.0, "onset_beats": (0.5,)}
    from audio.processing.dynamic_masking import apply_frequency_dynamic_masking
    rendered, meta = apply_frequency_dynamic_masking(
        data, sr, bass, kick, 0.30, bands=("low",), ranges={"low": (50.0, 120.0)}
    )
    assert meta["applied"] is True
    near = slice(390, 610)
    far = slice(1500, 1900)
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))
    assert rms(rendered[near]) < rms(data[near])
    assert abs(rms(rendered[far]) - rms(data[far])) < 0.02
