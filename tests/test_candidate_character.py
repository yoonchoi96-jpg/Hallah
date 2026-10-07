import math
import wave

import numpy as np

from audio.rendering.contracts import build_render_request
from audio.rendering.source import SourceAudioGenerator


def test_candidate_tone_signatures_are_distinct(tmp_path):
    src = tmp_path / "source.wav"
    sr = 22050
    t = np.arange(sr * 2) / sr
    x = 0.2 * np.sin(2 * math.pi * 220 * t)
    with wave.open(str(src), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes((x * 32767).astype("<i2").tobytes())

    generator = SourceAudioGenerator(tmp_path / "cache")
    settings = {}
    for direction in ("identity", "natural", "bold", "experimental"):
        request = build_render_request(
            "tone-" + direction,
            1,
            "candidate audition",
            {"direction": direction},
            source_asset_ids=(str(src),),
        )
        result = generator.render(request)
        settings[direction] = result.metadata["candidate_tone"]
    assert len({tuple(sorted(v.items())) for v in settings.values()}) == 4
