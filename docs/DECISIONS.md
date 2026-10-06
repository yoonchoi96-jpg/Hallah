# Architecture Decisions

## 001 — Song Context is the source of truth
One canonical context model is shared by analysis, agents, candidate generation and future UI/plugins.

## 002 — Candidate-first mutation
AI operations create candidates before changing project state. User selection is the commit point.

## 003 — Offline-first heavy processing
Expensive analysis and rendering are asynchronous. Real-time playback should consume cached results whenever possible.

## 004 — Separate domain from UI
The core must be usable without a DAW interface so Music Agent V0 can be tested independently.

## 005 — Modular plugin engines
Native effects and Austin remain independently testable production engines callable by the Executive Producer.
