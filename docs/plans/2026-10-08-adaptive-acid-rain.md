# Adaptive Acid Rain implementation plan

**Goal:** Make the terminal water animation smoother, richer, more varied, and cheaper to render, with measured performance and robust resize/input behavior.

**Architecture:** Keep the CPU renderer and half-block terminal presentation. Evaluate analytic water once per displayed frame; build scene parameters once per frame; render into reusable HDR buffers, downsample bloom, and encode changed terminal runs. Adapt quality using measured frame work, including terminal writes, with hysteresis. Keep time independent of frame count.

**Tech stack:** Rust 2024, existing ndarray/crossterm/clap/rand dependencies. No GPU, web frontend, or new runtime dependencies.

## Design choices

- Retain the existing optical model and enrich lighting, water motion, and procedural environments. A new backend would expand deployment requirements; color filters alone would not provide the requested variety.
- Default to a journey through distinct acid, aurora, storm, abyss, lava, and prism environments with smooth transitions. Preserve manual scene selection and intensity controls.
- Reduce sampling and postprocessing cost before using lower render resolution. Target 60 FPS by default, with a configurable cap and bounded catch-up after stalls.

## Tasks and verification

1. Preserve a detached baseline of `2d0c3a5`; add a deterministic headless measurement harness there. Save frame times, bytes, summary, environment, and sample frames under `benchmarks/`.
2. Add regression tests for texel-center sampling and zero/singleton dimensions, then repair samplers and temporal reprojection. Normalize interpolated normals and grid derivatives.
3. Test analytic time partitioning, ripple causality/expiry and bounded input, then optimize wave evaluation and add subtle directional swells.
4. Test scene continuity and finite colors, then add smoothly blended scene parameters and procedural sky structures.
5. Test ANSI reconstruction, initial black frames, dirty runs, and resize; then encode runs with cached colors and synchronized output.
6. Test adaptive quality hysteresis and argument validation. Replace the fixed-step catch-up loop with one analytic evaluation per visible frame, responsive event waits, pause, bounded rain, safe terminal cleanup, and reproducible benchmark/snapshot modes.
7. Run `cargo check --locked`, tests, formatting, Clippy, release build, matched release benchmarks, snapshot inspection, and a PTY resize/input/cleanup smoke test. Record measured limits and update README.

All requested changes are implemented in the user's current clean checkout. The detached worktree is only for preserving and measuring the original implementation.

## Follow-up: slower, less literal, more stable optics

User observation: the surface remains too recognizably water, rotation is hard to notice,
some frames become coarse pixel noise, and the image evolves too quickly.

Confirmed causes: sinusoidal roll stops and reverses; automatic Low halves resolution and
duplicates columns; frame jitter changes even a frozen scene; unresolved wave harmonics and
sharp highlights alias. Surface, lighting and camera shared one clock.

- Preserve the delivered source and measurements under `benchmarks/refinement-before/`.
- Use continuous camera roll, independent 0.35x surface time and 0.30x material time.
- Extend scene dwell/fade to 70/20 seconds, with spectral chrome/plasma material responses.
- Filter unresolved wave carriers before sampling; account for normal variance and pixel
  footprint in specular filtering. Remove frame jitter; preserve history on quality changes.
- Keep automatic quality at full resolution, trading sustained overload for a slower frame
  cap. Explicit Low retains smooth reconstruction.
- Verify frozen-scene stability, continuous camera motion, spectral filtering, pacing,
  side-by-side moving previews, performance and terminal lifecycle.
