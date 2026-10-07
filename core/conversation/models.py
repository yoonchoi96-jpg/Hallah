"""Conversation models used by the Executive Producer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserUtterance:
    text: str
    context_version: int

@dataclass(frozen=True)
class ExecutiveProducerIntent:
    summary: str
    affected_dimensions: tuple[str, ...] = ()
