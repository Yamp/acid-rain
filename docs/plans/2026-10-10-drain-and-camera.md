# Drain, camera coverage and pigment optics implementation plan

**Goal:** Hide no world edge during camera rotation, make a sustained draining whirlpool the startup mode, and give pigment slightly brighter color without losing soft optics or replacing absorption with a brightness floor.

**Architecture:** Keep the finite water domain and fit the full camera frustum within its dynamic height slab with a guard margin. Add a persistent 96×64 shallow-water drain using existing HLL/MUSCL numerical building blocks; co-evolve pigment with exactly the accepted hydraulic face fluxes and source fractions. Retain Classic and Rapids as selectable modes. Preserve Snell/Fresnel, spectral dispersion, thin-film interference and normalized aperture diffraction.

**Tech stack:** Rust 2024, existing dependencies; Python via uv for diagnostic captures and saved benchmarks.

The user explicitly chose Drain as the startup mode. Existing dirty work is preserved in place, with a verified source/native baseline in `benchmarks/whirlpool-before`. No commit, reset or publication is requested. Implementation proceeds within the authorized task.

## Alternatives and selected scope

1. Camera fit versus an infinite repeated world: fitting the whole frustum and dynamic wave slab removes the reproduced edge without introducing tiling seams, reflected-normal flips or duplicating a small drain everywhere. It retains the orbit and adjusts framing when the aspect ratio or roll needs it. Reflection continuation remains a bounded environment approximation.
2. Cosmetic funnel versus numerical draining flow: choose the numerical flow so radial motion, surface depression, angular momentum, transported pigment and removed mass agree. Model a pumped periodic water patch, not a fully resolved 3D bathtub, air core or bubbles.
3. Output brightness floor versus pigment optics: evaluate a modest passive extinction recalibration and the correct integrated attenuation of distributed volume light. Keep background transmission under Beer-Lambert, preserve clear-water identity, and never normalize pigment concentration per frame.

## Tasks and validation

1. Preserve current 203-test source and native baseline with exact provenance. The before binary remains speed 3, bleaching 1, intensity 2 and Classic by default.
2. Reproduce camera misses for normal and narrow/tall viewports with saved RGB/masks and source hashes. Add failing regressions for rotated corners at both height slab faces. Fit camera FOV with a safety margin and verify no primary downward ray sees sky solely through the finite patch boundary. Include displaced surfaces and tiny viewports. Preserve Snell, multi-reflection and diffraction invariants.
3. Implement Drain's positive-depth, conservative periodic hydraulic grid, finite initial rotating core and matching pressure bowl. Use a smooth central outflow and distributed clear-water feed that returns exactly accepted withdrawn volume with prescribed tangential momentum. Resolve shear from nonuniform feed and rain impulses. Maintain fixed time ticks, CFL rejection and persistent state; test water volume, depth positivity, finite velocities, no spin from an unforced symmetric source, long-run activity and frame partitioning.
4. Couple Dye to accepted hydraulic stages: use shared face discharges multiplied by reconstructed volume concentration, and remove the same local fraction of pigment as water. Record drained pigment separately. Bleaching keeps its exact exponential reaction and ledger. Test uniform concentration consistency, source/sink accounting, positivity, disabled bleaching, retries and same-grid indexing. Existing Classic/Rapids transport remains available.
5. Integrate three modes in Water with smooth visible transitions, persistent hydraulic state and no dye reset. Expose startup selection and runtime keys; keep existing Rapids shortcut, help, pause, resize, speed 3 and bleaching controls. Test startup default and all mode transitions.
6. Implement and compare a modest pigment-optics adjustment. Validate passive transmittance bounds, species color ordering, clear-water identity, no light from a dark source and dense-mixture response. Shared helper coefficients must be used by optical diagnostics. Primary and secondary volume attenuation must agree.
7. Run `cargo check --locked --all-targets` before executing Rust, full release tests, strict Clippy and formatting. Build a protected final native binary; run controls/PTY and resize/rain/mode stress. Do not reuse before measurements as after evidence.
8. Capture matched-camera/source before/after evidence for the edge and pigment optics plus actual default Drain animation. Verify sustained water/pigment flow and no opaque collapse at defaults. Save source hashes, commands and lossless images; inspect actual output. Run sequential native CPU comparisons only after other heavy work ends, distinguish same-mode cost from new-default workload, save and display tables.
9. Update README, physics/dye/optics documentation, and the final report with controls, model limits, tests, visuals and measured cost. Verify final source hashes match all final artifacts.

## Physical limits

The drain is a depth-averaged shallow-water approximation with prescribed pumping. Resolved horizontal shear and vorticity are not a solved 3D turbulence cascade. Do not claim a physically resolved air core, molecular dissolution or wave-optical scene transport. Aperture diffraction remains a normalized image-space approximation; the existing geometric refraction and finite reflection budget remain explicit.
