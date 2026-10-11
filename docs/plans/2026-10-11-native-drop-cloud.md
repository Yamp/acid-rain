# Visible drops and volumetric pigment (2026-10-11)

The native Metal edition will show a ballistic colored droplet, its actual contact with the simulated water, and a sinking, folding pigment cloud. Normal defaults remain unchanged; click and O add an individual drop, and --drop-demo isolates one drop for observation.

## Physics and representation

The surface solver remains authoritative for wave height and contact. Native droplets carry persistent IDs and world coordinates; pigment enters only on contact. Native pigment uses a bounded 96x64x32 three-dimensional concentration field. Conservative finite-volume transport, bounded fluxes, normalized sources, diffusion and physical-clock bleaching preserve a meaningful mass budget. A prescribed descending vortex-ring wake and smooth strain approximate unresolved three-dimensional motion; this is not a fully resolved Navier-Stokes solver. The terminal edition retains its existing two-dimensional pigment path.

The cloud window follows the existing integer-texel world recentering. GPU updates use snapshot physical time, never display-frame count. Retained impact records bridge skipped display frames. Pause/help/repeated snapshots cannot age or reinject sources.

## Rendering

Integrate Beer-Lambert extinction and colored single scattering along refracted rays through the three-dimensional field. Render airborne spheres as instances into the linear HDR target, before existing exposure, RGB separation and sRGB output. Pigment absorption never becomes white emission. Limit ray steps and wake support to keep GPU cost predictable.

## Implementation and acceptance

1. Rust lifecycle, uncolored surface impulse, ABI2 event journal and tests.
2. Conservative GPU transport and real Metal mass/positivity/rebase/bleach tests.
3. Swift scheduling, immutable uploads, picking, demo/capture controls and tests.
4. Volume optics and instanced drops, real GPU shader regression tests.
5. Build and capture one drop before contact, at impact and during dispersal; inspect the images and motion. Validate actual native controls, pause, resize and performance. Run affected Rust regressions. Save evidence in native/artifacts/drop-cloud.
