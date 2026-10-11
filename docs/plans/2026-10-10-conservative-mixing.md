# Conservative filament mixing implementation plan

**Goal:** Make colored liquid stretch into coherent threads, curls and overlapping folds while preserving pigment, realistic flow coupling, soft optical color and bounded terminal rendering cost.

**Architecture:** Retain the existing conservative passive-species grid. Use less diffusive MC reconstruction and its stricter positivity CFL; replace the implicit mass fade with a separately controlled first-order bleaching reaction and converted-mass ledger. Drive classic transport with a periodic streamfunction and exact discrete-curl face velocities. Add persistent, compact, counter-rotating impact wakes as an explicitly reduced near-surface stirring model. Integrate flow, wake evolution and pigment on shared integer physical ticks, independent of terminal resolution and frame cadence for matched input timestamps.

**Tech stack:** Rust 2024, existing ndarray/crossterm renderer, Python via uv for saved diagnostic images and benchmarks. No new runtime library required.

The user's request authorizes implementation in the current working tree. Existing uncommitted features are preserved; no commit, publication or workspace reset is needed.

## Design choices

1. Improve conservative transport and add bounded coherent stirring (chosen). This directly changes dissolved concentration and keeps per-frame work bounded.
2. Add a surface/bulk exchange model. Useful for vertically stratified plumes, but introduces extra flow/depth assumptions and doubles scalar transport. The present rain is already miscible, so chemical dissolution or solid-particle reactions would be unsupported.
3. Replace the solver with full 3-D fluid/solute dynamics. This would need a different computational budget and surface representation. This change instead documents its two-dimensional scope and empirical wake closure.

Transport and diffusion conserve pigment. The user additionally requested controlled decoloration so sustained rain does not fill the view with old color. Add an explicit first-order reaction of optically active pigment into colorless products under uniform effective exposure: default multiplier 1, half-life 8 physical seconds; adjustable from 0 (off) to 4. Accumulate converted pigment separately in the reaction ledger. Beer–Lambert absorption responds to the remaining optically active column mass. With bleaching disabled, a closed periodic patch accumulates pigment; an open channel can carry it out.

The user also requested default speed **3×**, preserving live speed controls and resetting to 3 with `s`. The default physical-water rate becomes 0.75 seconds per wall second. Bleaching follows this physical clock. Runtime `v`/`b` decrease/increase bleaching by 0.25 and `B` resets it; `--bleach` sets its initial multiplier.

## Tasks and validation

1. Freeze all current Rust and Cargo files plus native binary provenance in `benchmarks/mixing-before`. Reuse the protected previous native build only after exact source/compiler/flags verification.
2. In `src/dye.rs`, write failing closed-mass and thin-filament translation/shear regressions. Replace minmod with MC, update the CFL reconstruction factor from 1.5 to 2, remove age decay and mass-discard threshold. Initially retain the 96×64 grid and diffusion coefficient; change them only if diagnostics justify the cost. Preserve positivity, open-boundary outflow and source normalization.
3. Add exact streamfunction face fluxes to `src/dye.rs`, with source/shape validation and tests for uniform concentration under periodic incompressible flow, conservation and positivity. Share vertex-grid indexing with `src/stirring.rs`.
   Column transport uses the rapids' depth-averaged momentum velocity, leaving the empirical surface return flow for foam. Diffusive flux is `-D h_face grad(M/h)`, with harmonic interface depth and the corresponding per-cell positivity bound; equal true concentrations must not diffuse simply because water depth differs.
4. Add periodic finite-core base flow in `src/vortex.rs` and bounded compact wake evolution in `src/stirring.rs`. Wakes use a smooth compact streamfunction, finite cores, paired signs, midpoint advection, core spreading and explicit damping. A symmetric impact in motionless water must not invent horizontal swirl. Existing current/shear supplies orientation and amplitude; document this as a reduced impact-wake closure, not a resolved three-dimensional vortex ring or a turbulence cascade.
5. In `src/water.rs`, advance rapids, wakes, transition blend and dye on shared 8 ms ticks; keep continuous analytic-wave time. Add integrated classic/rapids frame-partition tests with pigment and resize/pause/empty-grid checks. Register the new module in `src/main.rs`.
6. Save direct concentration-field before/after animations and metrics with `scripts/compare_mixing.py`; inspect coherent deformation and diffusion with fixed sources and identical physical times. Also capture actual renderer output at the requested defaults: before scene speed 1.1×, final after speed 3×, both with surface scale 0.25×, Balanced quality and intensity 2. These renderer captures compare eight wall seconds, not equal physical histories or doses; label this explicitly. Preserve earlier first-pass captures separately and use the final bleaching-enabled archive for delivery. Artifacts must retain source hashes and exact commands.
7. Run `cargo check --locked --all-targets` before Rust execution, release tests, strict Clippy, formatting and native build. Run existing PTY checks and resize/rain stress. Save and display eighteen sequential CPU measurements: two water modes × three fixed qualities × before at 1.1× / after at matched 1.1× / after at default 3×, all at intensity 2. Use protected builds so unrelated workspace rebuilds cannot change measurements.
8. Update `docs/DYE.md`, relevant README/physics descriptions and a final verification report. State what is solved, what is an empirical closure, mass/outflow behavior, measured cost and actual visual evidence.

9. Implement the requested first-order bleaching with exact exponential half-steps around each transport tick; verify analytical half-life, zero-rate conservation, shape preservation, reacted-mass accounting and bounded pigment under repeated inputs. Expose CLI, help and paused runtime controls. Update speed default/reset to 3 and the headless benchmark to honor explicit speed so an unchanged-speed comparison remains possible.

10. Re-run final gates and captures after bleaching. Retain the initial 191-test conservative implementation only as `no-bleach-intermediate`, never as final delivery evidence. Benchmark old/new at matched 1.1×, and separately measure the requested new 3× default.

Independent algorithm review applies to both the face-flux conservation/positivity proof and the stirring model. Regression tests should demonstrate behavior and analytical invariants, not merely repeat formulas from the implementation.

## Completed verification

The final source passed 203 release tests, all-target Cargo check, strict Clippy and formatting. Its protected native build passed 94 PTY checks and 150 stress operations, including six zero-size recoveries. Final field and renderer captures match all source hashes; both modes also passed 64 physical seconds of default continuous rain. Eighteen sequential CPU cases separate matched 1.1× cost from the new 3× default. Results, limits, exact commands and raw artifacts are in [the final verification report](../../benchmarks/mixing-after/verification.md).
