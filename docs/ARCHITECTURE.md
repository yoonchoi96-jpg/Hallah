# Hallah Architecture

## High-level system

```
UI / Conversation
       │
       ▼
Executive Producer
       │
       ├──► Song Context
       ├──► Analysis Tools
       ├──► Candidate Engine
       └──► Production Tools
                  │
                  ▼
          Audio / MIDI Engine
                  │
                  ▼
            Render / Cache
                  │
                  ▼
              A / B / C / D
```

## Boundaries

### Core
Pure domain models and deterministic logic. No DAW/UI dependency.

### Analysis
Turns audio/MIDI into structured musical observations: tempo, meter, key/scale, harmony, rhythm/groove, onset/transients, melody/pitch, timbre/spectrum, dynamics, stereo/spatial profile, frequency occupation, and musical role.

### Song Context
Persistent state for one song. It is the single source of truth shared by agents, generators, UI, and future plugins.

### Executive Producer
Interprets natural language against Song Context, decides which tools to call, generates candidate plans, and explains tradeoffs. It must not directly mutate the song before a user-approved candidate is selected.

### Candidate Engine
Every meaningful transformation can produce alternatives. Candidates are immutable until applied.

### Audio engine
Real-time playback and offline rendering are separate concerns. Heavy operations should be asynchronous and cached.

### Plugins
Native production engines are callable by the Executive Producer. External VST3/AU hosting is a later integration layer.

## Dependency rule

```
core  ←  analysis
core  ←  candidates
core  ←  agent
audio ←  plugins
audio ←  rendering
ui    → core/agent
```


## Candidate lifecycle

Candidates are immutable proposals tied to the SongContext version from which they were generated. Their lifecycle is `proposed → preview_ready → selected → applied`, with `rejected` and `superseded` terminal states. Lifecycle transitions return new candidate values rather than mutating the existing candidate.

Candidate validation is deterministic and checks context-version lineage, fixed constraints, and dimension-specific musical authorities. Applying a candidate always creates a new SongContext version; the parent context remains unchanged. A candidate generated for an older context cannot be reused on an unrelated newer context.

Selection and application are separate transitions so future audio/MIDI preview generation can occur between them. The candidate history and decision log preserve what was selected, rejected, and applied.
