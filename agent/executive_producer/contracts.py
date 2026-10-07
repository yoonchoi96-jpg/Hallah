"""Executive Producer orchestration contracts."""
from dataclasses import dataclass

from core.candidates.models import Candidate
from core.conversation.models import ExecutiveProducerIntent
from core.music_context.models import SongContext


@dataclass(frozen=True)
class ProductionRequest:
    utterance: str
    context: SongContext

@dataclass(frozen=True)
class ProductionPlan:
    intent: ExecutiveProducerIntent
    candidates: tuple[Candidate, ...]

class ExecutiveProducer:
    def plan(self, request: ProductionRequest) -> ProductionPlan:
        raise NotImplementedError
