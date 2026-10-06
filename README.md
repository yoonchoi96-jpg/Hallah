# Hallah

> **Anyone can be the Executive Producer.**

Hallah is an AI-native music production environment built around a conversational Executive Producer.

Hallah is not a one-click AI music generator. The user owns taste, direction, judgment, and final decisions; Hallah listens, analyzes, executes, generates alternatives, and keeps the song's evolving context.

## Core loop

```
오늘은 무슨 음악을 만들어볼까요?
        ↓
Conversation
        ↓
Sample Import
        ↓
Audio Analysis
        ↓
Song Context / Music DNA
        ↓
Executive Producer
        ↓
Candidate Engine
        ↓
A / B / C / D
        ↓
User chooses
        ↓
Apply + update Song Context
        ↺
```

## V0 goal

Build a real **Music Agent V0** that can:

1. accept a conversation and musical direction
2. import multiple audio/MIDI sources
3. analyze tempo, key, harmony, rhythm, timbre, dynamics and musical role
4. infer relationships between sources
5. assign **Musical Authority** by dimension
6. maintain a persistent **Song Context**
7. generate multiple candidate transformations
8. render real audio/MIDI candidates
9. let the user choose and apply a candidate
10. continue developing the same song with memory of prior decisions

## Architecture principles

- Song Context is the single source of truth.
- Analysis, reasoning, generation, rendering, and UI are separate layers.
- Candidate generation is non-destructive.
- Expensive analysis/rendering is asynchronous and cacheable.
- Real-time playback stays lightweight.
- Future plugin engines and external VST3/AU integrations remain modular.

## Project map

- `core/` — song state, conversation, analysis, candidates, project model
- `audio/` — audio analysis, processing, rendering and cache
- `agent/` — Executive Producer orchestration and tools
- `plugins/` — native production engines, beginning with Austin
- `apps/` — future desktop/DAW application
- `docs/` — product and architecture specifications
- `tests/` — deterministic tests and fixtures

## Status

**Foundation / Music Agent V0**

See [docs/ROADMAP.md](docs/ROADMAP.md).
