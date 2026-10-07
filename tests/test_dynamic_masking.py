import numpy as np

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
    t = np.arange(8000) / sr
    data = np.column_stack([
        0.4 * np.sin(2 * np.pi * 80 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t),
    ]).astype("float32")
    bass = {"role": "bass", "dimension": "low_end", "bpm": 60.0}
    kick = {"role": "kick", "dimension": "rhythm", "bpm": 60.0, "onset_beats": (0.0625,)}
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


def test_mix_gain_plan_caps_cumulative_ducking():
    from audio.processing.masking import build_mix_gain_plan
    authority = {
        "bass": {"role": "bass", "low_energy_ratio": 0.7, "mid_energy_ratio": 0.2, "high_energy_ratio": 0.1},
        "kick": {"role": "kick", "dimension": "rhythm", "low_energy_ratio": 0.7, "mid_energy_ratio": 0.2, "high_energy_ratio": 0.1},
        "vocal": {"role": "vocal", "dimension": "melody", "low_energy_ratio": 0.1, "mid_energy_ratio": 0.7, "high_energy_ratio": 0.2},
    }
    relationships = {
        "bass": {"decisions": (
            {"reference_id": "kick", "priority": 92, "amount": 0.24, "bands": ("low",)},
            {"reference_id": "vocal", "priority": 100, "amount": 0.24, "bands": ("mid",)},
        )}
    }
    plan = build_mix_gain_plan(authority, relationships, max_total_duck=0.30)
    assert sum(float(x["allocated_amount"]) for x in plan["bass"]) <= 0.30 + 1e-9
    assert plan["bass"][0]["reference_id"] == "vocal"


def test_spectral_curve_dynamic_masking_has_localized_curve():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    data = np.zeros((8000, 1), dtype=np.float32)
    time = np.arange(8000, dtype=np.float32) / sr
    data[:, 0] = np.sin(2 * np.pi * 1000 * time)
    source = {"role": "guitar", "bpm": 60.0}
    reference = {"role": "vocal", "bpm": 60.0, "onset_beats": (0.25,), "spectral_centroid_hz": 1000.0, "_source_id": "vocal"}
    out, meta = apply_spectral_curve_dynamic_masking(data, sr, source, reference, 0.30, bands=("mid",), ranges={"mid": (500.0, 1800.0)})
    assert meta["applied"] is True
    assert meta["center_hz"] == 1000.0
    assert float(np.sqrt(np.mean(out[1800:3200, 0] ** 2))) < float(np.sqrt(np.mean(data[1800:3200, 0] ** 2)))


def test_spectral_curve_uses_source_peak_near_reference():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    time = np.arange(8000, dtype=np.float32) / sr
    data = np.column_stack([np.sin(2 * np.pi * 1200 * time)]).astype(np.float32)
    source = {"role": "guitar", "bpm": 60.0}
    reference = {"role": "vocal", "bpm": 60.0, "onset_beats": (0.25,), "spectral_centroid_hz": 1000.0, "_source_id": "vocal"}
    _, meta = apply_spectral_curve_dynamic_masking(
        data, sr, source, reference, 0.30, bands=("mid",), ranges={"mid": (500.0, 1800.0)}
    )
    assert 1100.0 <= float(meta["center_hz"]) <= 1300.0
    assert float(meta["sigma_hz"]) < 400.0


def test_spectral_curve_detects_multiple_collision_peaks():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    time = np.arange(8000, dtype=np.float32) / sr
    data = np.column_stack([0.8 * np.sin(2 * np.pi * 900 * time) + 0.7 * np.sin(2 * np.pi * 1400 * time)]).astype(np.float32)
    source = {"role": "guitar", "bpm": 60.0}
    reference = {"role": "vocal", "bpm": 60.0, "onset_beats": (0.25,), "spectral_centroid_hz": 1100.0, "_source_id": "vocal"}
    _, meta = apply_spectral_curve_dynamic_masking(
        data, sr, source, reference, 0.30, bands=("mid",), ranges={"mid": (600.0, 1700.0)}
    )
    centers = tuple(float(x) for x in meta["center_hz_all"])
    assert any(abs(x - 900.0) < 60.0 for x in centers)
    assert any(abs(x - 1400.0) < 60.0 for x in centers)


def test_spectral_collision_threshold_is_reference_guided():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    time = np.arange(8000, dtype=np.float32) / sr
    data = np.column_stack([np.sin(2 * np.pi * 900 * time)]).astype(np.float32)
    source = {"role": "guitar", "bpm": 60.0}
    reference = {"role": "vocal", "bpm": 60.0, "onset_beats": (0.25,), "spectral_centroid_hz": 900.0, "_source_id": "vocal"}
    _, meta = apply_spectral_curve_dynamic_masking(
        data, sr, source, reference, 0.25, bands=("mid",), ranges={"mid": (500.0, 1500.0)}
    )
    assert meta["collision_mode"] == "source_peaks_reference_guided"


def test_spectral_curve_tracks_collision_frequency_over_time():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    n = 8000
    time = np.arange(n, dtype=np.float32) / sr
    first = time < 0.5
    source = np.where(
        first,
        np.sin(2 * np.pi * 800 * time),
        np.sin(2 * np.pi * 1400 * time),
    ).astype(np.float32)
    reference = np.where(
        first,
        np.sin(2 * np.pi * 800 * time),
        np.sin(2 * np.pi * 1400 * time),
    ).astype(np.float32)
    out, meta = apply_spectral_curve_dynamic_masking(
        source[:, None], sr,
        {"role": "guitar", "bpm": 60.0},
        {"role": "vocal", "bpm": 60.0, "onset_beats": (0.0,), "note_durations_beats": (1.0,), "_source_id": "vocal"},
        0.30, bands=("mid",), ranges={"mid": (500.0, 1800.0)},
        reference_data=reference[:, None],
    )
    centers = tuple(float(x) for x in meta["tracked_centers_hz"])
    assert meta["tracking"] is True
    assert any(abs(x - 800.0) < 80.0 for x in centers)
    assert any(abs(x - 1400.0) < 80.0 for x in centers)
    assert len(out) == n


def test_spectral_collision_strength_controls_ducking_depth():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking
    sr = 8000
    time = np.arange(8000, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    strong_ref = source.copy()
    weak_ref = (0.05 * source).astype(np.float32)
    common = {
        "role": "vocal", "bpm": 60.0, "onset_beats": (0.25,),
        "note_durations_beats": (0.5,), "spectral_centroid_hz": 1000.0,
        "_source_id": "vocal",
    }
    strong, sm = apply_spectral_curve_dynamic_masking(
        source[:, None], sr, {"role": "guitar", "bpm": 60.0},
        common, 0.30, bands=("mid",), ranges={"mid": (700.0, 1300.0)},
        reference_data=strong_ref[:, None],
    )
    weak, wm = apply_spectral_curve_dynamic_masking(
        source[:, None], sr, {"role": "guitar", "bpm": 60.0},
        common, 0.30, bands=("mid",), ranges={"mid": (700.0, 1300.0)},
        reference_data=weak_ref[:, None],
    )
    strong_rms = float(np.sqrt(np.mean(strong[2200:3800, 0] ** 2)))
    weak_rms = float(np.sqrt(np.mean(weak[2200:3800, 0] ** 2)))
    original_rms = float(np.sqrt(np.mean(source[2200:3800] ** 2)))
    assert sm["tracking_strength_max"] > wm["tracking_strength_max"]
    assert strong_rms < weak_rms
    assert strong_rms < original_rms


def test_spectral_collision_tracking_smooths_frequency_jumps():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking

    sr = 8000
    n = 12000
    time = np.arange(n, dtype=np.float32) / sr
    first = time < 0.75
    source = np.where(
        first,
        np.sin(2 * np.pi * 800 * time),
        np.sin(2 * np.pi * 1400 * time),
    ).astype(np.float32)
    reference = source.copy()
    _, meta = apply_spectral_curve_dynamic_masking(
        source[:, None], sr,
        {"role": "guitar", "bpm": 60.0},
        {"role": "vocal", "bpm": 60.0, "onset_beats": (0.0,),
         "note_durations_beats": (2.0,), "_source_id": "vocal"},
        0.30, bands=("mid",), ranges={"mid": (500.0, 1800.0)},
        reference_data=reference[:, None],
    )
    smoothed = tuple(float(x) for x in meta["smoothed_centers_hz"])
    assert len(smoothed) > 3
    assert any(850.0 < x < 1350.0 for x in smoothed)
    assert float(meta["tracking_max_center_jump_hz"]) < 220.0


def test_spectral_collision_strength_attack_release_is_smoothed():
    from audio.processing.dynamic_masking import apply_spectral_curve_dynamic_masking

    sr = 8000
    n = 12000
    time = np.arange(n, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    reference = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    reference[(time >= 0.55) & (time < 0.75)] *= 0.1
    _, meta = apply_spectral_curve_dynamic_masking(
        source[:, None], sr,
        {"role": "guitar", "bpm": 60.0},
        {"role": "vocal", "bpm": 60.0, "onset_beats": (0.0,),
         "note_durations_beats": (2.0,), "_source_id": "vocal"},
        0.30, bands=("mid",), ranges={"mid": (700.0, 1300.0)},
        reference_data=reference[:, None],
    )
    strengths = tuple(float(x) for x in meta["smoothed_strengths"])
    assert len(strengths) > 3
    assert max(strengths) > min(strengths)
    assert max(abs(b - a) for a, b in __import__('itertools').pairwise(strengths)) < 0.5
