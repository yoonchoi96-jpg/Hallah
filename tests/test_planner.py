from agent.executive_producer.planner import parse_production_intent, plan_context_patch, apply_patch
from core.music_context.models import SongContext

def test_natural_language_production_plan():
    intent=parse_production_intent("드럼은 그대로 유지하고 기타 루프 기준으로 키랑 코드 맞춰. BPM은 드럼 따라가.")
    patch=plan_context_patch(SongContext(),intent)
    values=dict(patch.operations)
    assert ("drums","rhythm") in intent.authority_requests
    assert ("guitar","harmony") in intent.authority_requests
    assert values["bpm"]=="follow:drums"
    assert values["tonality"]=="follow:guitar"

def test_explicit_bpm_is_applied():
    intent=parse_production_intent("BPM 128로 맞춰")
    new=apply_patch(SongContext(),plan_context_patch(SongContext(),intent))
    assert new.version==1
    assert new.bpm==128
