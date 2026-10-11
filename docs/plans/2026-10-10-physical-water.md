# Physically consistent water implementation

**Goal:** Make the resolved wave dynamics and surface geometry physically consistent, preserving pastel psychedelia, runtime controls and bounded CPU rendering.

**Architecture:** Retain analytic evaluation, replacing independent wave speeds and decorative impact harmonics with a finite-depth gravity-capillary spectral response. Shared SI parameters drive swell frequencies, viscous attenuation, vortex pressure and rendering. Rays intersect the displaced surface and normals differentiate the same heights. The optical environment remains deliberately artistic.

**Audit:** Existing wave packets independently specify envelope speed, carrier frequency and temporal phase; nonlinear height compression changes superposition and volume; a fixed vortex coordinate warp relocates fresh impacts; surface normals are resolution-calibrated rather than metric derivatives; primary rays intersect z=0 instead of the displayed heights. Runtime is O(cells × live impacts) plus bounded lookup preparation; preserve that bound and cache radial basis functions.

**Steps:**
1. Archive delivered source/binary and measurements in `benchmarks/physics-before`.
2. Add shared gravity, density, surface tension, viscosity, domain length and depth. Test shallow/deep dispersion limits, capillary behavior, group velocity and dissipated oscillator energy.
3. Replace radial impact profiles with a filtered Hankel/Bessel superposition of damped gravity-capillary modes driven by a compensated momentum impulse. Test reference agreement, smooth initial response, no hard artificial front and bounded lookup error. Keep lifetime retirement smooth and explicit as numerical truncation.
4. Derive directional swell frequencies and weak second-order bound harmonics from physical wavelengths. Preserve zero mean without nonlinear height clipping; cap excitation into the small-steepness regime. Correct fresh-impact placement and trace its center through the prescribed vortex flow using a bounded midpoint integrator.
5. Make vortex pressure use the same gravity/domain units; test divergence and centripetal balance. Keep spiral ornament confined to a documented weak perturbation of prescribed flow.
6. Intersect primary rays with the height field, calculate normals in meters, use dielectric Fresnel from water IOR and a compatible roughness model. Test known planar intersections, normals, Fresnel limits and serial/parallel equality.
7. Run fmt/check/strict Clippy/tests/native build, PTY controls and visual review; save deterministic timing and numerical evidence. Document the model's scope and unresolved phenomena.

**Alternatives:** A full 3-D two-phase Navier–Stokes solver would model overturning, spray and bubbles but exceeds this CPU terminal renderer's real-time budget. A shallow-water solver alone would lose the gravity-capillary dispersion requested here. A spectral linear/weakly nonlinear surface model gives verifiable resolved waves with explicit limits; vortex-wave coupling remains approximate.

**Primary references:** MIT Akylas/Mei wave modules, chapter 3 (finite-depth dispersion and group velocity); NIST DLMF 10.17 (Bessel asymptotics); PBRT 4th edition, Specular Reflection and Transmission (dielectric Fresnel). Links and equations will be retained in the physics documentation.
