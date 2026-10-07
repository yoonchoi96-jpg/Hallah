"""Deterministic cache-key helpers for rendered candidates."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def build_cache_key(
    candidate_id: str,
    context_version: int,
    kind: str,
    parameters: Mapping[str, object],
    source_asset_ids: tuple[str, ...] = (),
) -> str:
    payload = {
        "candidate_id": candidate_id,
        "context_version": context_version,
        "kind": kind,
        "parameters": parameters,
        "source_asset_ids": source_asset_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
