from audio.rendering.contracts import build_render_request
from audio.rendering.source import SourceAudioGenerator
import math
import wave
import numpy as np


def _write_stereo_source(path):
    sr = 22050
    t = np.arange(sr * 2) / sr
    left = 0.18 * np.sin(2 * math.pi * 220 * t)
    right = 0.08 * np.sin(2 * math.pi * 330 * t)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        data = np.column_stack((left, right))
        wav.writeframes((data * 32767).astype("<i2").tobytes())


def test_candidate_space_transient_signatures_are_distinct(tmp_path):
    src = tmp_path / "stereo.wav"
    _write_stereo_source(src)
    generator = SourceAudioGenerator(tmp_path / "cache")
    settings = {}
    for direction in ("identity", "natural", "bold", "experimental"):
        request = build_render_request(
            "space-" + direction,
            1,
            "candidate audition",
            {"direction": direction},
            source_asset_ids=(str(src),),
        )
        result = generator.render(request)
        settings[direction] = result.metadata["candidate_space_transient"]
    assert len({tuple(sorted(v.items())) for v in settings.values()}) == 4


def test_candidate_space_transient_changes_stereo_source(tmp_path):
    src = tmp_path / "stereo.wav"
    _write_stereo_source(src)
    generator = SourceAudioGenerator(tmp_path / "cache")

    identity = generator.render(build_render_request(
        "identity", 1, "audition", {"direction": "identity"},
        source_asset_ids=(str(src),),
    ))
    bold = generator.render(build_render_request(
        "bold", 1, "audition", {"direction": "bold"},
        source_asset_ids=(str(src),),
    ))

    identity_data, _ = generator._read(__import__("pathlib").Path(identity.artifact_ref))
    bold_data, _ = generator._read(__import__("pathlib").Path(bold.artifact_ref))
    identity_side = np.mean(np.abs(identity_data[:, 0] - identity_data[:, 1]))
    bold_side = np.mean(np.abs(bold_data[:, 0] - bold_data[:, 1]))
    assert bold_side > identity_side
