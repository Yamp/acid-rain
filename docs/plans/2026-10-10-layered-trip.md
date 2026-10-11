# Layered psychedelic light and echoes

**Goal:** Make the default image more intricate and psychedelic, with longer layered trails, gentle self-similar light and additional adjustable strength, while keeping soft color, smooth motion and bounded CPU work.

**Architecture:** Three convex temporal reservoirs in linear light create distinct short/medium/long chromatic echoes. Smooth spatial drift decorrelates them without changing water dynamics. A small, finite set of repeated directional light structures is evaluated in the cached environment, so existing reflections and refractions carry it naturally. Default intensity becomes 2 (the former maximum), range becomes 0–4; brightness/saturation have separate soft bounds. Physics and camera clocks stay unchanged.

**Choices:** Increasing global saturation alone would lose the soft palette. Recursive full-scene ray tracing would multiply frame cost. Use bounded temporal layers and cached finite-octave illumination instead. The user's request authorizes this design and implementation in the existing dirty workspace; no new commit or isolated checkout is needed.

1. Freeze current source and native baseline provenance. Add failing functional regressions for a longer echo tail and distinct response beyond intensity 2, then implement finite, convex, resize-safe histories in `src/retina.rs`.
2. In `src/scene.rs` and `src/colors.rs`, add bounded self-similar illumination that remains smooth at environment-map resolution. Test finiteness, soft color, zero-dose behavior and that intensity 4 differs from 2 without faster camera motion.
3. In `src/cli.rs`, `src/main.rs`, `src/help.rs` and `src/term.rs`, propagate default 2 and maximum 4, show the selected intensity in a runtime HUD, preserve `e` and +/- controls and history continuity. Check actual terminal behavior, including paused changes.
4. Extend comparison tooling with explicit intensity values. Capture matched old/default/overdrive clips, inspect actual output and adjust only if the requested extra structure/trails are not visible.
5. Run cargo check before Rust execution, full meaningful tests, strict Clippy, formatting and native build. Run PTY checks, save sequential native CPU A/B data and display a table. Verify source hashes across builds, captures and measurements. Update controls and optical-model documentation.

## Execution evidence

Completed all five steps. Visual review rejected an overly blurred first pass; the final version preserves at least 58% current light at default intensity and 45% at maximum, and modulates existing world light without filling dark pockets. A phase-wrap regression also caught and fixed a discontinuity in independently drifting echoes.

Final checks: 170 Rust tests, strict Clippy, formatting, native builds, 80 PTY control checks and 150 deterministic stress operations passed. Eighteen sequential CPU cases were saved and displayed; the new default adds 0.68–1.74 ms per frame at 200×60 in this run. Frozen source hashes match the gates, final captures and protected benchmark build. See the [final verification report](../../benchmarks/trip-after/verification.md), [CPU table](../../benchmarks/trip-after/cpu/comparison.md) and [three-level animation](../../benchmarks/trip-after/high-levels.gif).
