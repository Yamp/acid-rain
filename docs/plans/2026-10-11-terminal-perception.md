# Terminal perception implementation plan

**Goal:** Port the native renderer's stable camera and controllable subjective post layer to the terminal renderer.

**Architecture:** Keep the existing Rust water/optics and terminal presentation pipeline. Replace reactive frustum fitting, adapt bounded HDR histories and spectral effects to terminal half-cell pixels, and drive perception from active wall time independently of simulation speed. Reuse current buffers and bounded per-frame work.

**Tech Stack:** Rust 2024, existing crossterm/ndarray, deterministic headless rendering, PTY control tests.

The user approved this port after the native implementation. No additional design approval, dependency, commit or rollback of the dirty working tree is needed. Native source and the built macOS application remain untouched.

## 1. Camera

Files: `src/term.rs` Camera block and camera tests.

1. Reproduce normal-wave and patch-rebase aperture changes with a failing test.
2. Port the analytic all-angle cone fit and nominal height envelope with emergency release.
3. Smooth world-space target, inclination/FOV and roll residual; preserve zero-time redraws.
4. Test geometric coverage, invariant aperture, cadence, rebasing and existing surface hits.

## 2. Subjective post layer

Files: `src/retina.rs`, Renderer integration in `src/term.rs`.

1. Test configured RGB displacement bounds and neutral-history protection before changing code.
2. Use three unwarped HDR reservoirs with 0.10/0.35/0.80-second half-lives; remove large affine zoom/rotation echoes.
3. Add smoothly modulated dose, bounded drifting channels and cached spatial chroma patterns.
4. Apply normalized spectral diffraction and recombine RGB before one tone mapping step.
5. Drive all phases and histories using active wall time; freeze on pause/help and reject hidden/stalled catch-up.
6. Test constants/black, channel bounds, half-lives, energy, gamut, pause, quality and integration.

## 3. Controls

Files: `src/cli.rs`, `src/user_input.rs`, `src/help.rs`, `src/main.rs`, `src/benchmark.rs`, `README.md`.

1. Add `--lsd 0..2`, default 1; `n/m` change by 0.1, `N` resets.
2. Keep Trip, speed, RGB radius and quality independent; show LSD in help/status.
3. Add CLI/key tests and PTY exercise including pause/help, resize and normal terminal cleanup.

## 4. Verification

Preserve baseline source/binary under `benchmarks/terminal-perception/before`.
Run cargo check before execution/build, then formatting, strict Clippy, all Rust tests and release build.
Compare sequential matched CPU preparation and PTY output cadence at 200x60 and 320x100, save and print tables.
Inspect real generated frames and verify runtime controls. PTY byte cadence is not terminal-emulator FPS.
Save report, commands and source hashes under `benchmarks/terminal-perception`.
