import math
import wave

import numpy as np

from audio.rendering.contracts import build_render_request
from audio.rendering.source import SourceAudioGenerator


def _write_tone(path, hz):
    sr = 22050
    t = np.arange(sr * 2) / sr
    x = 0.12 * np.sin(2 * math.pi * hz * t)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes((x * 32767).astype("<i2").tobytes())


def test_candidate_eq_signatures_are_distinct(tmp_path):
    src = tmp_path / "source.wav"
    _write_tone(src, 440)
    generator = SourceAudioGenerator(tmp_path / "cache")
    values = []
    for direction in ("identity", "natural", "bold", "experimental"):
        result = generator.render(build_render_request(
            "eq-" + direction, 1, "audition", {"direction": direction},
            source_asset_ids=(str(src),),
        ))
        values.append(tuple(sorted(result.metadata["candidate_eq_masking"].items())))
    assert len(set(values)) == 4


def test_candidate_eq_uses_spectral_overlap(tmp_path):
    src = tmp_path / "source.wav"
    ref = tmp_path / "reference.wav"
    _write_tone(src, 440)
    _write_tone(ref, 440)
    generator = SourceAudioGenerator(tmp_path / "cache")
    authority = {
        str(src): {
            "low_energy_ratio": 0.05,
            "mid_energy_ratio": 0.80,
            "high_energy_ratio": 0.15,
        },
        str(ref): {
            "low_energy_ratio": 0.05,
            "mid_energy_ratio": 0.80,
            "high_energy_ratio": 0.15,
        },
    }
    result = generator.render(build_render_request(
        "masking", 1, "avoid masking", {
            "direction": "natural",
            "authority_analysis": authority,
        },
        source_asset_ids=(str(src),),
    ))
    assert result.metadata["candidate_eq_masking"]["masking_reduction"] > 0.0
