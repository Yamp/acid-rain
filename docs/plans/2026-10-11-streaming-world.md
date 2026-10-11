# Streaming water world implementation plan

**Goal:** Travel over a stable procedural bed and spatially varied flowing water while retaining camera rotation, approved colors, controls and bounded work.

**Architecture:** Keep a finite 2 m water window, shift its pigment grid by whole cells, and use double precision world origins with small local rendering coordinates. Generate deterministic bed strips and compact local phenomena only around that window. Independent camera and physical clocks separate translation, rotation and current; Drain/Rapids retain their local hydraulic domains.

**Tech stack:** Rust 2024, existing CPU renderer and finite-volume dye transport, no additional runtime dependencies.

The existing dirty workspace and protected accepted binary are preserved. The requested implementation is already authorized; work proceeds here without a commit, reset or worktree migration.

1. Add failing tests for camera translation relative to a fixed water patch and correct history reprojection across origin shifts. Separate camera look target from sampling origin, use small local coordinates, preserve rotation and the existing guarded frustum.
2. Add a cadence-independent non-closing travel controller. Pause and travel toggles preserve position. Wire `c` and `--no-travel`; keep the requested visual defaults.
3. Test open dye boundaries and exact whole-cell window shifts, accounting for pigment leaving the active domain. Preserve overlap and wave/wake locations; initialize only fresh strips. Add a separately toggleable uniform current (`w`, `--no-current`) with continuous accumulated displacement.
4. Generate seeded compact world vortices and wave packets around the active patch. Reuse their streamfunction/current in pigment and surface motion. Retain legacy anchored Drain/Rapids solvers and freeze travel during their transitions.
5. Replace patch-carried bed texture with a bounded rolling world cache including a refraction halo. Generate seeded mineral, cellular, spiral and emissive patterns; stable samples, seamless cache movement, bounded albedo/emission. Keep color grading and retinal optics unchanged.
6. Integrate interactive and headless entry points, help and documentation. Verify positive/negative origins, long travel, pause/resume, resize, mode changes, constant cache capacity and preserved overlap.
7. Run `cargo check --locked --all-targets` before tests/builds; run release tests, strict Clippy and formatting. Capture actual output, inspect movement and colors, run PTY control/resize checks, save and display matched performance tables. Build and install only the verified native executable.

Validation must distinguish actual simulated transport from prescribed local wave/current approximations; discarded off-window dynamic state is not retained indefinitely. Static terrain regenerates exactly from world seed and coordinates. Finite captures and workload measurements do not prove every possible frame or terminal performance.
