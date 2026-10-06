"""Deterministic V0 natural-language production planner."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from core.music_context.models import SongContext

@dataclass(frozen=True)
class ContextPatch:
    operations: tuple[tuple[str,str], ...]
    rationale: str = ""

@dataclass(frozen=True)
class ProductionIntent:
    raw: str
    fixed_sources: tuple[str,...] = ()
    authority_requests: tuple[tuple[str,str], ...] = ()
    constraints: tuple[str,...] = ()
    bpm: float | None = None
    tonal_source: str | None = None

def parse_production_intent(utterance: str) -> ProductionIntent:
    text=utterance.strip()
    fixed=[]
    if re.search(r"(드럼|drum).*(그대로|유지|건드리지)", text, re.I):
        fixed.append("drums")
    if re.search(r"(기타|guitar).*(중심|기준|맞춰)", text, re.I):
        tonal_source="guitar"
    else:
        tonal_source=None
    bpm_match=re.search(r"(?:bpm|템포)\s*(?:을|를|=|:)?\s*(\d+(?:\.\d+)?)", text, re.I)
    bpm=float(bpm_match.group(1)) if bpm_match else None
    authorities=[]
    if re.search(r"(드럼|drum).*(리듬|그루브).*(따라|기준)",text,re.I):
        authorities.append(("drums","rhythm"))
    if tonal_source:
        authorities.append((tonal_source,"harmony"))
        authorities.append((tonal_source,"texture"))
    constraints=[]
    if fixed:
        constraints.extend(f"{s}:fixed" for s in fixed)
    if re.search(r"(다른 악기|나머지|여러 악기).*(기타|guitar).*(맞춰|따라)",text,re.I):
        constraints.append("other_tracks:adapt_to:guitar")
    return ProductionIntent(text,tuple(fixed),tuple(authorities),tuple(constraints),bpm,tonal_source)

def plan_context_patch(context: SongContext, intent: ProductionIntent) -> ContextPatch:
    ops=[]
    for source,dimension in intent.authority_requests:
        ops.append(("authority",f"{source}:{dimension}"))
    for source in intent.fixed_sources:
        ops.append(("constraint",f"{source}:fixed"))
    for constraint in intent.constraints:
        ops.append(("constraint",constraint))
    if intent.bpm is not None:
        ops.append(("bpm",str(intent.bpm)))
    elif any(source=="drums" and dimension=="rhythm" for source,dimension in intent.authority_requests):
        ops.append(("bpm","follow:drums"))
    if intent.tonal_source:
        ops.append(("tonality",f"follow:{intent.tonal_source}"))
    return ContextPatch(tuple(ops), "Derived from the user's production direction.")

def apply_patch(context: SongContext, patch: ContextPatch) -> SongContext:
    new=context.next_version()
    for op,value in patch.operations:
        if op=="bpm" and not value.startswith("follow:"):
            new.bpm=float(value)
        elif op=="tonality":
            new.pending_decisions.append(f"Resolve tonality from {value.removeprefix('follow:')}.")
        elif op=="authority":
            new.pending_decisions.append(f"Set musical authority: {value}.")
        elif op=="constraint":
            new.pending_decisions.append(f"Apply constraint: {value}.")
    new.decisions.append(f"[planner] {patch.rationale}")
    return new
