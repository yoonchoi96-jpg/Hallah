# Music Context

Hallah's Song Context is the single source of truth for a song.

## Dimension-specific authority

An analyzed asset can be authoritative for only the musical dimensions it is good at:

- drums/percussion → rhythm
- guitar/piano → harmony and texture
- vocal/lead → melody
- bass → low end
- texture → texture

Authority is confidence-scored. When several assets disagree, Hallah does **not** silently average them. The strongest authority for the relevant dimension wins the initial context value, while the disagreement remains visible as a conflict/pending decision.

## Cross-asset relationship analysis

V0 compares assets without assuming every difference is a problem.

- **BPM match** — near-equal tempos are compatible.
- **Half/double-time** — 60↔120 and 120↔240 are recognized as related rhythmic grids rather than hard conflicts.
- **Tonal match** — same key and scale.
- **Relative major/minor** — shared pitch collection is treated as compatible.
- **Compatible tonal centers** — closely related centers (for example C↔F) are flagged as compatible, not contradictory.
- **Tonal conflict** — materially different centers remain unresolved.
- **Low-end overlap** — substantial shared low-frequency energy is surfaced as a possible masking conflict.
- **Spectral competition** — moderate band overlap becomes a dependency rather than a hard conflict.

The system records both the relationship and its explanation so a later Executive Producer can decide how to adapt the non-authoritative source.

## Adaptation constraints

Authority becomes actionable through first-class constraints:

- rhythm authority → other tracks **adapt** to the authority
- harmony authority → other tracks **adapt** to the authority
- melody authority → other tracks **avoid** conflicting with the melody
- low-end authority → other tracks **avoid** competing in the low end
- texture authority → other tracks **adapt** to the texture authority

These are guidance for candidate generation, not irreversible edits.

## Example

If drums analyze at 120 BPM and guitar at 124 BPM:

- rhythm authority: drums
- context BPM: 120
- relationship: rhythm conflict (120 vs 124)
- constraint: guitar adapts to drums
- pending decision: user may keep drums, adapt guitar, or choose another strategy

If guitar is A minor while drums are C major:

- harmony authority: guitar
- context key/scale: A minor
- tonal relationship: unresolved conflict remains recorded
- later candidate generation can adapt the non-authoritative asset

## Design rule

Analysis produces evidence. Authority resolves responsibility. Relationships explain interaction. Constraints turn authority into actionable guidance. Song Context stores the resulting state and unresolved decisions.
