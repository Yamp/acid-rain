# Smooth highlights, persistent trails and frame cadence

**Goal:** Remove grid-shaped highlights in the original Metal application and terminal, make colored afterimages longer and clearer, and improve measured frame cost without reducing the physics or preferred quality settings.

**Architecture:** Keep the shared water model and unwarped linear-HDR history. Repair spatial reconstruction at the source and temporal reconstruction at presentation. Mirror optical and retinal changes in the portable desktop renderer. Preserve Classic/Journey/High, speed 2.4, Trip 4, RGB separation 2 logical pixels and hidden falling drops.

**Tech stack:** Rust 2024, Swift/Metal, WGSL/wgpu; deterministic CPU/GPU numerical fixtures and frozen executable A/B measurements.

## Evidence and baseline

Baseline commit: `1a542c21be747e4817c4c3746838a5218b0b0c8d`. Frozen binaries, full shader resources, captures, source archive and exact commands live in ignored `artifacts/smooth-optics/baseline/`.

- Terminal normal variance omits mixed curvature; an axis-aligned saddle is incorrectly smooth, while its rotated equivalent broadens GGX. A separate normal-length term introduces interpolation-phase roughness.
- Balanced spatial AA has an abrupt threshold. The bright-only 2x2 bloom reduction loses one-pixel translation phase.
- Metal normal and environment reconstruction is bilinear on small source grids. Screen-space direct specular filtering ignores the varying half-vector.
- Capped retinal history uses one bilinear reading even when reducing by more than two pixels per axis. A slow symmetric EMA barely captures brief moving highlights.
- Quality changes clear retinal history; a slow active terminal frame can be treated as an interruption. The terminal presenter's 150 ms delay cap creates holds when render age plus source interval exceeds it.
- Native baseline simulation takes about 20–23 ms, GPU 9–14 ms for the measured fixtures. A separate CPU sample identifies pressure-impact evaluation and radial Hankel reconstruction as the dominant simulation work. Terminal High takes roughly 30/31/79 ms for 120x40, 200x60 and 320x100 cells. These are offline work measurements, not display FPS.

## Implementation and validation sequence

1. **Terminal optical reconstruction** — `src/term.rs`.
   Add failing saddle-rotation and moving-glint regressions; include the complete normal gradient and remove phase-dependent pseudo-variance. Add threshold-continuity and bloom centroid/energy regressions before changing those filters. Preserve High supersampling and constant/affine fields. Slow active frames must advance retinal wall time; explicit resets still invalidate history.

2. **Native and portable optical reconstruction** — `native/Sources/Water.metal`, `desktop/shaders/water.wgsl`, new native optical tests.
   Compare smooth low-resolution surfaces and glints against dense references before implementation. Reconstruct a shared interpolating Catmull-Rom height field with analytic derivatives for optical intersections, shading, caustics and native contacts; enlarge camera bounds by the proven 9/32 sample-range overshoot envelope. Keep positive cubic B-spline reconstruction for environment magnification only. Integrate the direct GGX lobe over the pixel with footprint-adaptive quadrature of both normal and view. Do not increase the global pixel budget or alter caustic flux transport. Check real production shaders, subpixel continuity, mean energy and retained resolvable contrast.

   The stronger geometry requirement follows additional failing cases: normal-only B-splines attenuate a resolved three-cell wave, a fixed 2x2 glint quadrature loses about half the energy at a 0.2 radian normal footprint, and a cubic crest can occlude a ray that a bilinear bracket incorrectly lets escape. Cubic normals alone do not repair those inconsistencies.

3. **Visible, bounded retinal persistence** — `src/retina.rs`, native and desktop retinal shaders and policies, their real GPU tests.
   Three independent HDR reservoirs use luminance-selected scalar RGB attack/release: attack half-lives 0.025/0.050/0.090 s; release 0.24/0.72/1.65 s. Test a fixed-duration pulse across cadences against the analytic response, plus black, neutral, saturated, pause/reset and colored-on-dark contracts. Keep the neutral-history rejection and existing RGB radius ceiling. Never feed distorted output back into storage.

4. **Retinal GPU work and allocation** — `native/Sources/MetalRenderer.swift`, `desktop/src/gpu.rs`, pipeline/resource layouts.
   Fuse three updates into one compute dispatch; compute the current radiance's source-pixel area average once, including fractional reduction ratios. Test impulse flux and phase at 2x, 4x and noninteger scales. Keep reservoir dimensions tied to the drawable and the existing memory budget so quality changes retain history. Prove texture identity/content across High/Low changes and keep explicit invalidation for real discontinuities.

5. **Terminal cadence** — `src/presentation.rs`.
   Reproduce source period/publication age 100/100 ms at 60 Hz display ticks. Cover the measured age and interval with bounded adaptive delay, retain monotonically advancing presentation and queue bounds. Avoid redundant color conversions for identical endpoints with exact parity checks.

6. **Measured physics optimization** — `src/waves.rs`, `src/water.rs`, `native/core/src/lib.rs` only where profile-supported.
   Preserve the complete spectral basis, pressure impulse timing, conservation and grid resolution. Establish numerical equivalence against the current scalar path before batching/caching repeated work. Keep a simple reference in tests, not a second runtime model. Profile the changed path and compare the same frozen fixtures.

7. **Integration and acceptance**.
   Run `cargo check --locked --workspace --all-targets` before Rust test/build commands; focused red/green tests first, then workspace tests, strict Clippy, nightly fmt, actual Metal and WGSL fixtures. Run original native policy and optical suites, terminal PTY controls and a representative GUI check. Build after shaders and host bindings are synchronized. Run timed A/B jobs sequentially with identical settings, save raw samples and print tables; inspect full-resolution captures and report work-time versus presentation-time honestly. Use the existing four-platform CI if publishing a follow-up; never overwrite v0.2.0 assets.

No approval gate is needed: the user explicitly requested implementation and identified the original Metal app and terminal as the main acceptance targets.
