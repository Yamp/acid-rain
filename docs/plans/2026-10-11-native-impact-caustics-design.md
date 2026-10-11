# Native impact, caustics and living bottom

The user requests physically credible drop-driven surface motion, real refractive
highlights on the bottom, a more complex moving psychedelic bottom, and an independent
airborne-drop visibility toggle which defaults to off. Existing rain, pigment, camera,
speed, palettes and terminal controls remain active.

## Model and implementation

1. **Drop contact:** retain ballistic contact against the moving water. Replace only the
   native legacy broad kick with a localized gravity-capillary pressure impulse derived
   from drop radius, mass and incoming speed. Use the true sub-frame contact time.
   Resolve the wave spectrum on the existing surface lattice; apply the same displacement
   to height and depth, preserving bed elevation. Keep terminal impulse behavior unchanged.
   This is a linear free-surface response; it does not resolve spray or overturning crowns.
2. **Bottom caustics:** GPU forward refraction of a bounded incident grid, with Snell,
   Fresnel and wavelength-dependent IOR. Additive normalized footprints conserve incoming
   flux; finite light-source and receiver footprints regularize folds. Directional light
   and ambient light remain separate in bed shading. Missing exterior source coverage
   falls back smoothly to the flat-water solution rather than creating dark world edges.
3. **Bottom material:** continuous world-anchored mineral/fractal texture, moving sparse
   acid organisms, bounded reflective albedo and explicit luminous emission. This is
   authored material/animation, not a claim to simulate living organisms. Optical lighting
   is applied to that material, not embedded as painted caustics. Observation mode keeps its
   neutral bed to isolate pigment dynamics.
4. **Visibility:** `a` switches only the instanced airborne draw. Hidden by default,
   `--show-drops` opts in and `--hide-drops` opts out. Rain sources, contact physics,
   pigment transport and camera picking are independent of this switch. Existing `v/b`
   bleaching controls remain intact; physical-key routing supports Russian layouts.

The bounded GPU grid approach is preferred to a per-screen-pixel iterative caustic solve
or full volumetric fluid/path simulation: it adds focused wave optics at a predictable
cost while retaining the existing real-time native architecture. Finite-resolution and
linear-wave limitations must be documented honestly.

## Verification

- Save the currently built application as the matching pre-change baseline.
- Physics tests: source scaling/energy/zero mode, contact timing, wave propagation,
  consistent bed elevation and all three water modes.
- Real GPU tests: flat/slanted refractive transport, integrated flux, focusing,
  finite outputs, dispersive paths and continuous movement.
- Real rendered paired drop/control images with empty pigment isolate visible surface
  deformation; a separate normal scene tests bottom material, caustics and color.
- Run native build, appropriate Rust/Swift/Metal tests, actual controls/resize smoke,
  and save/print matched timing tables. Do not confuse offline GPU duration with
  displayed frame cadence.
- Verify the terminal release binary is unchanged.

Implementation is authorized by the user's requests. Work proceeds in the current dirty
workspace without resetting or committing the existing accumulated changes.
