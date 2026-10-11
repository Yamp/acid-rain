# Pastel vortices implementation plan

**Goal:** Restore soft rose/lilac imagery, slower evolution, translucent overlapping forms, stronger smoothing, manual quality controls and recognizable vortex structure.

**Architecture:** Extend the existing analytic surface with smooth finite-core rotational flow and pressure bowls, retaining bounded per-frame work and grid-aware filtering. Use a related pastel palette across worlds and gradual manual scene transitions. Keep full resolution in Auto and provide direct quality selection; all changes remain in the current working tree containing the previously delivered implementation.

**Tech stack:** Rust 2024, existing ndarray/crossterm/clap; uv for artifact scripts.

1. Preserve the current source, binary, previews and measurements under `benchmarks/pastel-before`.
2. Add physically inspired finite-core vortex deformation and spiral advection in `src/water.rs` and a small dedicated module. Test finite cores, rotation direction, bounded heights, resize and frame-partition independence before implementation; retain wave filtering.
3. Retune `src/scene.rs` and `src/colors.rs`: related rose, pearl, lavender and restrained cool accents; less saturation, emission and interference; more transmission, less mirror reflectivity. Slow material time and journey transitions.
4. Improve `src/term.rs` spatial smoothing, bounded highlight response and smooth manual scene changes without discarding temporal history. Preserve still-scene stability and quality-switch continuity; test transitions.
5. Add `k` cycling and direct `0/1/2/3` Auto/High/Balanced/Low selection through `src/user_input.rs`, `src/cli.rs`, `src/main.rs`. Add a brief on-screen indication. Slow default global motion further, keeping explicit speed controls.
6. Run fmt/check/clippy/tests, build native release, exercise quality selection and transitions in a PTY, inspect all scene previews and a short animation, then save sequential CPU measurements with a comparison table.

Alternatives considered: an image-space spiral only would not affect physical normals/refraction; a full fluid grid solver would add timestep and stability costs. A bounded analytic vortex field fits the current renderer and supplies geometry for the optical layers. These remain real-time physical approximations, not a fluid solver.
