"""Musical Authority inference primitives."""

from core.analysis.contracts import AudioAnalysis
from core.music_context.models import AuthorityDimension, MusicalAuthority


ROLE_TO_DIMENSIONS: dict[str, tuple[AuthorityDimension, ...]] = {
    "drums": ("rhythm",),
    "percussion": ("rhythm",),
    "vocal": ("melody",),
    "lead": ("melody",),
    "bass": ("low_end",),
    "guitar": ("harmony", "texture"),
    "piano": ("harmony", "texture"),
}


def infer_authority(analyses: list[AudioAnalysis]) -> list[MusicalAuthority]:
    """Infer initial authority from explicit/recognized musical roles."""
    authorities: list[MusicalAuthority] = []
    for analysis in analyses:
        role = (analysis.role or "").strip().lower()
        for dimension in ROLE_TO_DIMENSIONS.get(role, ()):
            confidence = analysis.confidence.get(dimension, 0.5)
            authorities.append(
                MusicalAuthority(
                    source_id=analysis.asset_id,
                    dimension=dimension,
                    confidence=max(0.0, min(1.0, confidence)),
                    rationale=f"Role '{role}' is a strong prior for {dimension} authority.",
                )
            )
    return authorities
