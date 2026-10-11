# Water color and highlight separation implementation plan

**Goal:** Make transported pigment visible in saturated rainbow water, with a
textured fractal bed, dark/light regions and sparse emissive veins. Reassemble
independently drifting R/G/B channels directly, add bounded retinal diffraction,
and preserve smooth motion and bounded cost.

The user's latest feedback supersedes the initial highlight-mask approach:
the milky candidate was rejected, and a direct full-frame channel displacement
was requested explicitly. Bottom emission is now an authorized material effect.

**Architecture:** Diagnose illumination, pigment transport, postprocessing,
perception and display grade independently against the protected Clarity build.
Change the optical/perceptual stages supported by those measurements. Keep the
hydraulic and pigment simulation, display interpolation and runtime controls.

**Tech stack:** Rust 2024, existing CPU renderer and crossterm, isolated Rust
diagnostic drivers, Python via uv for artifact analysis.

## Investigation and implementation

1. Verify current sources against `benchmarks/clarity-after/source.tar.gz`.
   Preserve that archive and executable as the comparison reference.
2. Reproduce early Drain and several Journey worlds using identical water,
   camera and exposure. Measure primary transmission/reflection/volume/glint
   contributions, actual pigment absorption, and post/retina/display effects.
   Isolate the directional background using an equal-mean constant backplate;
   do not treat a global saturation slider as a root-cause correction.
3. Replace the highlight detector with direct R/G/B samples from independently
   drifting positions, including broad and dim image features. Verify neutral
   constant fields, display-pixel scaling, finite bounds and stable pause.
   Add a normalized wavelength-dependent diffraction PSF before display mapping.
4. Select a supported transmission/illumination correction that reveals actual
   pigment. Retain passive Beer attenuation, Fresnel/Snell reflection/refraction
   and documented material approximations. Do not manufacture pigment mass or
   erase extinction with a brightness floor. Use saturated incident light and
   cached fractal bed material. Keep explicitly emissive veins separate from
   diffuse reflection: emission has only the outgoing Beer/Fresnel path.
5. Implement narrowly scoped optical/perceptual changes with focused failing
   regressions before the fix. Inspect matched image sheets across worlds,
   rain/dye age and bright/dark cases, including final output after all effects.
6. Run cargo check before Rust execution; then complete release tests, strict
   Clippy and format checks on an immutable snapshot. Build its native binary.
   Run PTY controls and matched performance checks; retain/display the table.
7. Verify hashes and install the gated executable. Document visible behavior,
   physical approximation limits and measured rather than assumed performance.

No commit, reset or branch change is needed in this existing working tree. The
user has requested these corrections directly; routine design choices proceed
within that scope without another approval step.

## User correction: clean acid color and favorite defaults

The user also rejected the compact-RGB `73cd655` version as pastel/milky.
Keep its protected binary and evidence as a comparison, not an accepted result.
The requested startup preset is now Journey / Classic / High, speed 2.4,
intensity 4, rain 32 per physical second, drop diameter 1.5, bleaching 1,
RGB enabled with a 2-pixel nominal shift, interpolation enabled, running.

Two specific mechanisms require correction: Classic's deeper column magnifies
the old absorption-derived, unlit volume veil; the display transform forces
bright saturated colors toward white by preserving near-white luminance during
gamut compression. Separate absorption and scattering with incident-light and
energy bounds, and use a chromaticity-preserving display shoulder. Evaluate
selective reduction of conflicting temporal color histories while preserving
long trails against dark backgrounds. Do not change pigment mass or transport
to manufacture color, or add fictitious bathymetry to a constant-depth solver.

Capture the complete pipeline at the user's exact preset, with chronological
rain/history and several Journey positions. Compare against the protected
rejected version using matched scene/physical time and random impacts. Inspect
actual images, not only a saturation score. Only after the combined appearance
is reviewed: freeze, run full gates and PTY checks, benchmark the matched preset,
verify source/binary identities, and install the tested executable.
