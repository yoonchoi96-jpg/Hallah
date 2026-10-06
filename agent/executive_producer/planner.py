"""Deterministic V0 natural-language production planner."""
from __future__ import annotations
from dataclasses import dataclass
import re
from core.music_context.models import SongContext, MusicalAuthority
from core.music_context.constraints import ContextConstraint, MusicalRelationship

@dataclass(frozen=True)
class ContextPatch:
    operations: tuple[tuple[str,str,str|None], ...]
    rationale: str = ""

@dataclass(frozen=True)
class ProductionIntent:
    raw: str
    fixed_sources: tuple[str,...] = ()
    authority_requests: tuple[tuple[str,str], ...] = ()
    constraints: tuple[str,...] = ()
    bpm: float | None = None
    tonal_source: str | None = None
    adaptation_requests: tuple[tuple[str,str,str], ...] = ()

def parse_production_intent(utterance: str) -> ProductionIntent:
    text=utterance.strip()
    fixed=[]
    if re.search(r"(드럼|drum).*(그대로|유지|건드리지)", text, re.I):
        fixed.append("drums")
    tonal_source="guitar" if re.search(r"(기타|guitar).*(중심|기준|맞춰)",text,re.I) else None
    bpm_match=re.search(r"(?:bpm|템포)\s*(?:을|를|=|:)?\s*(\d+(?:\.\d+)?)",text,re.I)
    bpm=float(bpm_match.group(1)) if bpm_match else None
    authorities=[]
    if re.search(r"(드럼|drum).*(리듬|그루브).*(따라|기준)",text,re.I):
        authorities.append(("drums","rhythm"))
    if re.search(r"(기타|guitar).*(리듬|그루브).*(그대로|따라|기준|가져)",text,re.I):
        authorities.append(("guitar","rhythm"))
    if tonal_source:
        authorities += [(tonal_source,"harmony"),(tonal_source,"texture")]
    constraints=[]
    if fixed: constraints += [f"{s}:fixed" for s in fixed]
    if re.search(r"(다른 악기|나머지|여러 악기).*(기타|guitar).*(맞춰|따라)",text,re.I):
        constraints.append("other_tracks:adapt_to:guitar")
    return ProductionIntent(text,tuple(fixed),tuple(authorities),tuple(constraints),bpm,tonal_source)

def plan_context_patch(context: SongContext, intent: ProductionIntent) -> ContextPatch:
    ops=[]
    for source,dimension in intent.authority_requests: ops.append(("authority",source,dimension))
    for source in intent.fixed_sources: ops.append(("constraint",source,"fixed"))
    for constraint in intent.constraints: ops.append(("constraint",constraint,"adapt"))
    if intent.bpm is not None: ops.append(("bpm",str(intent.bpm),None))
    elif intent.authority_requests:
        rhythm_sources=[source for source,dimension in intent.authority_requests if dimension=="rhythm"]
        if rhythm_sources: ops.append(("bpm",rhythm_sources[0],"follow"))
    if intent.tonal_source: ops.append(("tonality",intent.tonal_source,"follow"))
    return ContextPatch(tuple(ops),"Derived from the user's production direction.")

def apply_patch(context: SongContext, patch: ContextPatch) -> SongContext:
    new=context.next_version()
    for op,target,dimension in patch.operations:
        if op=="bpm":
            if dimension is None: new.bpm=float(target)
            else: new.pending_decisions.append(f"BPM follows {target}.")
        elif op=="tonality":
            new.pending_decisions.append(f"Resolve tonality from {target}.")
        elif op=="authority":
            auth=MusicalAuthority(target,dimension,1.0,"Explicit user direction.")
            new.authorities=[a for a in new.authorities if not (a.source_id==target and a.dimension==dimension)]
            new.authorities.append(auth)
        elif op=="constraint":
            if dimension=="fixed":
                new.constraints.append(ContextConstraint(target,"fixed",reason="Explicit user direction."))
            elif dimension=="adapt":
                parts=target.split(":")
                if len(parts)>=3: new.constraints.append(ContextConstraint(parts[0],"adapt",parts[2],reason="User adaptation direction."))
    return new
