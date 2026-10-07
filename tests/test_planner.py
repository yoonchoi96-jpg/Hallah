from agent.executive_producer.planner import (
    apply_patch,
    parse_production_intent,
    plan_context_patch,
)
from core.music_context.models import SongContext


def test_natural_language_production_plan():
    intent=parse_production_intent("드럼은 그대로 유지하고 기타 루프 기준으로 키랑 코드 맞춰. BPM은 드럼 따라가.")
    patch=plan_context_patch(SongContext(),intent)
    new=apply_patch(SongContext(),patch)
    assert ("drums","rhythm") in intent.authority_requests
    assert ("guitar","harmony") in intent.authority_requests
    assert any(a.source_id=="guitar" and a.dimension=="harmony" for a in new.authorities)
    assert any(c.target_id=="drums" and c.type=="fixed" for c in new.constraints)
    assert any("BPM follows drums" in x for x in new.pending_decisions)

def test_explicit_bpm_is_applied():
    intent=parse_production_intent("BPM 128로 맞춰")
    new=apply_patch(SongContext(),plan_context_patch(SongContext(),intent))
    assert new.version==1 and new.bpm==128

def test_candidate_carries_context_constraints():
    from core.candidates.engine import build_candidates
    ctx=SongContext(version=3)
    ctx.constraints=[]
    candidates=build_candidates(ctx,"make bass")
    assert len(candidates)==4
    assert all(c.parent_context_version==3 for c in candidates)


def test_explicit_pitch_reference_is_preserved():
    intent=parse_production_intent("드럼 피치를 기타에 맞춰")
    assert ("drums","pitch","guitar") in intent.adaptation_requests
    new=apply_patch(SongContext(),plan_context_patch(SongContext(),intent))
    assert new.adaptation_overrides["drums"]["pitch"] == "guitar"
