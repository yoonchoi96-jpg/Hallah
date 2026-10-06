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
