"""Provider-neutral Executive Producer V0 implementation."""

from agent.executive_producer.contracts import ProductionPlan, ProductionRequest
from core.candidates.engine import build_candidates
from core.conversation.models import ExecutiveProducerIntent


class DefaultExecutiveProducer:
    """Minimal deterministic EP used before an LLM provider is connected."""

    def plan(self, request: ProductionRequest) -> ProductionPlan:
        intent = ExecutiveProducerIntent(
            summary=request.utterance,
            affected_dimensions=(),
        )
        candidates = build_candidates(request.context, request.utterance)
        return ProductionPlan(intent=intent, candidates=candidates)
