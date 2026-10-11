# Rain controls, transported pigment and visible rapids

**Goal:** Repair the reported index panic, make rapids visibly active, expose rain rate and actual drop diameter, and transport colored drop pigment through the water.

**Architecture:** Keep the conservative shallow-water solver, with explicitly empirical jump-powered short waves and foam entrainment for unresolved surface activity. Add a persistent finite-volume passive pigment field driven by the current; its three nonnegative species affect Beer attenuation in the existing linear-light optics. Drop diameter changes the physical source footprint, independently of rainfall frequency. Rust 2024, no new runtime dependency.

1. Freeze accepted source and executable provenance. Reproduce periodic-index failure, add failing endpoint regression and wrap palette indices safely. Audit other periodic samplers and stress resize/quality controls.
2. Add rate/diameter CLI, runtime actions and help state. Up/Down changes rate; comma/period changes diameter; R resets. Integrate burst and automatic rain with the same diameter.
3. Test actual source footprint changes at constant strength; add radius scaling to classic spectral impulses and shallow-flow kicks. Preserve defaults and bounded inputs.
4. Implement persistent conservative pigment advection/diffusion, positivity and open/periodic boundaries, documented surface-layer dilution. Test mass, translation, spreading and resize persistence; tint primary and secondary transmitted light with zero-dye identity.
5. Make dissipation at rapids produce advected short-wave geometry and patchy foam, with bounded local forcing. Test both conservation of underlying hydraulics and visible fixed-camera evolution after startup.
6. Check Rust before execution, run focused red/green tests, full tests, Clippy/fmt and native release. Exercise new controls and a longer mixed resize/quality/rain stress in a real PTY.
7. Compare actual pixels with a fixed camera and unchanged physical clock; inspect several time samples and colors in both modes. Save matched CPU measurements and source fingerprints. Document empirical turbulence/entrainment limits and exact controls.

The user explicitly requested these edits and colored dissolving drops. Implementation proceeds in the existing dirty workspace; no commit or external publication is requested.

## Implementation evidence

- Reproduced palette seam panic, wrapped the palette index, and added endpoint regressions. The original user's stack trace is unavailable; this is a confirmed reproducible crash candidate.
- Added independent rain rate/impact diameter controls and current values in help/HUD. Empty viewports now advance water together with ongoing rain.
- Added bounded, persistent pigment transport and passive absorption through water and foam. Numerical units, boundaries and empirical closures are documented in [DYE.md](../DYE.md).
- Changed the rapids bed, local breaking detail and foam to produce visible fixed-camera motion; hydraulic variables remain separate from bounded optical detail. See [RAPIDS.md](../RAPIDS.md).
- Final source passed 154 unit tests, strict Clippy, formatting and native compilation. The native executable passed 68 PTY checks and 150 mixed stress operations.
- Matched source fingerprints and full validation/performance evidence are indexed in [verification.md](../../benchmarks/rain-after/verification.md).
