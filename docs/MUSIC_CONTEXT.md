# Music Context

Song Context is Hallah's persistent representation of a song.

## Required domains
- song identity: title, genre, mood, references, musical identity, energy curve
- structure: BPM, time signature, key, scale, chord progression, sections, bar/beat positions
- performance: groove, swing, microtiming, velocity/dynamics, melody, phrasing
- sound: sound palette, timbre, spectral occupation, transient/head/tail, stereo/spatial relationships
- track relationships: harmonic/rhythmic/melodic authority, masking/conflict, dependencies, adaptation constraints
- production state: selected/rejected candidates, pending decisions, processing chain, automation, arrangement
- user model: explicit preferences, learned preferences, explanation level, decisions made during this song

## Versioning
Every accepted change advances a context version. Candidates are evaluated against a parent context version. Applying a candidate creates a new context version.
