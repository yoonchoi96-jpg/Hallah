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

## Relationships

Hallah records explicit relationships between analyzed assets:

- authority: source controls a dimension for another asset
- conflict: measurements disagree materially
- dependency: reserved for later adaptation constraints

These relationships are deterministic and explainable in V0.

## Example

If drums analyze at 120 BPM and guitar at 124 BPM:

- rhythm authority: drums
- context BPM: 120
- conflict: 120–124 BPM disagreement
- pending decision: user may keep drums, adapt guitar, or choose another strategy

If guitar is A minor while drums are C major:

- harmony authority: guitar
- context key/scale: A minor
- tonal conflict remains recorded
- later candidate generation can adapt the non-authoritative asset

## Design rule

Analysis produces evidence. Authority resolves responsibility. Song Context stores the resulting state and unresolved decisions.
