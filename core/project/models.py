"""Project and asset models."""
from __future__ import annotations

from dataclasses import dataclass
from core.music_context.models import SongContext


@dataclass(frozen=True)
class AudioAsset:
    id: str
    path: str
    role_hint: str | None = None


@dataclass
class MusicProject:
    id: str
    context: SongContext
    assets: list[AudioAsset]

    def __post_init__(self) -> None:
        """Validate the project asset registry deterministically at its boundary."""
        seen: set[str] = set()
        for asset in self.assets:
            if not asset.id or not asset.id.strip():
                raise ValueError("AudioAsset id must not be empty.")
            if not asset.path or not asset.path.strip():
                raise ValueError(f"AudioAsset {asset.id} path must not be empty.")
            if asset.id in seen:
                raise ValueError(f"Duplicate AudioAsset id: {asset.id}")
            seen.add(asset.id)

    def asset_map(self) -> dict[str, str]:
        """Return the canonical logical-ID -> filesystem-path registry."""
        return {asset.id: asset.path for asset in self.assets}

    def asset_ids(self) -> tuple[str, ...]:
        """Return project asset IDs in deterministic project order."""
        return tuple(asset.id for asset in self.assets)

    def resolve_asset_path(self, asset_id: str) -> str:
        """Resolve a logical asset ID or fail explicitly."""
        paths = self.asset_map()
        try:
            return paths[asset_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown asset id {asset_id!r} in project {self.id!r}."
            ) from exc
