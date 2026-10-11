# Water model and validation

This document describes the shared water models and the terminal renderer's
column pigment and optics. The [native renderer](../native/README.md) and
[desktop renderer](../desktop/README.md) use separate GPU volume and optical passes.

The renderer models a resolved free surface in a resident 2 m × 2 m patch.
Default Classic mode streams that patch through a procedural world and uses
dispersive waves at a 0.70 m reference depth. `--water drain` selects a pump-fed,
depth-averaged whirlpool; `--water rapids` / `--rapids` selects
an open shallow channel over uneven terrain. `d` toggles Drain and Classic;
`f` toggles Rapids and Classic. Heights, optical depth and camera distances use
meters. The UI scales physical time; `--speed`, brackets and left/right arrows
do not change the equations. Up/down arrows control rain intensity.
The default scene speed is 2.4×; water advances at 25% of scene time, or 0.6 physical
seconds per wall second. `s` resets speed to 2.4×.
The sections on dispersive waves and prescribed vortices below describe classic mode.
[DRAIN.md](DRAIN.md) and [RAPIDS.md](RAPIDS.md) specify the finite-volume flow
solvers and their respective forcing and closures.

## Shared parameters

| Parameter | Value |
| --- | ---: |
| Gravity | 9.81 m/s² |
| Water density | 998.2 kg/m³ |
| Surface tension | 0.0728 N/m |
| Kinematic viscosity | 1.004 × 10⁻⁶ m²/s |
| Refractive index, green band | 1.333 |

The material values represent approximately room-temperature clean water. Molecular viscosity
and surface tension enter the classic wave equations; Drain and Rapids use
hydrostatic shallow-water dynamics and numerical dissipation. The pastel colors,
fluorescence, thin-film tint, environment, bloom and visual grading are artistic.

## Waves and impacts

Every free wave mode follows the finite-depth gravity-capillary dispersion law:

`ω² = (g k + σ k³ / ρ) tanh(k h)`.

Phase speed is `ω/k`; group speed is `dω/dk`. They cannot be selected independently.
Background waves comprise ten crossing directions. Small-steepness second-order
Stokes harmonics use twice the carrier phase, with finite-depth coefficient
`k a² (3 − tanh²(kh)) / (4 tanh³(kh))`. Their unresolved frequencies are filtered
separately. The conservative sum of background modal steepness is below 0.3.
Background swells are maintained incident waves, not a simulated wind atmosphere.

A drop applies a compensated Gaussian vertical momentum impulse. Its spectral
density vanishes at zero wavenumber; a downward central impulse is accompanied
by displaced surrounding water. The radial response is a Hankel superposition
of `J0(k r)` modes. Each mode solves
`η̈ + 2γ η̇ + ω² η = 0`, with `γ = 2ν k²`. Thus short wavelengths lose amplitude
faster. The impulse oscillator starts at zero displacement with nonzero velocity.

The numerical integral uses 384 midpoint spectral bins up to 192 rad/m and a
769-point shared radial basis. Bilinear/lookup interpolation is tested against
direct sums, independent Bessel angular quadrature and an eight-times-finer
spectral integral. Gravity and capillary terms apply to all resolved modes;
terminal resolution cannot display millimeter-scale capillary ripples. Frequencies
fade before aliasing instead of being folded into false coarse motion.

Impacts are retired through a smooth final window over a ten-second history.
This is a bounded-cost numerical truncation, distinct from viscosity. Finite-patch
zero-mode projection preserves the sampled mean height. It is not a local finite-volume
continuity solver. Heights superpose without nonlinear compression.

## Vortices and transport

Each prescribed vortex has finite-core tangential velocity
`vθ = Ω a² r / (r² + a²)` and pressure height
`η = −Ω² a⁴ / (2g(r² + a²))`. Their derivative satisfies radial pressure balance
`dη/dr = vθ²/(g r)`. The sum of horizontal vortex velocities is divergence-free.
The cores have opposite circulation and stay finite at the origin.

Impact centers follow bounded RK4 trajectories in locally frozen prescribed flow;
at age zero they remain at the actual drop location. Background swell distortion
uses a finite-residence flow map, spatial reconstruction and a weak diffused spiral
perturbation. The two-vortex pressure heights are superposed. Cross-pressure terms,
time-dependent wave-current scattering and fluid feedback on the vortices are
approximations rather than a coupled Navier–Stokes solution.

The non-streaming Classic test model uses a separate periodic version of the
finite-core current so pigment velocity joins at the closed patch seams.
Interactive Classic instead has an open resident window: entering cells are clear,
departing pigment is recorded as exported mass, and overlapping cells retain their
state when the window moves. Disabling camera travel does not close these boundaries.
A vertex streamfunction
supplies face velocities through a discrete curl, whose cell divergence cancels
up to floating-point error. This changes far-field mixing transport, not the
existing analytic surface geometry. Compact current-aligned impact wakes add
empirical two-dimensional stirring. They have smooth finite support, advected
and damped cores, and a bounded source budget; a vertical impact in still water
does not invent horizontal swirl. Wake decay removes stirring activity; pigment
bleaching is a separately controlled reaction.

## Optional pump-fed whirlpool

Drain evolves depth `h` and horizontal momentum `h u` on a fixed 96×64 grid,
shared with pigment. The flat-bottom 2×2 m domain is periodic. A finite central
suction disk removes water and its local momentum; an outer annular feed replaces
the same accepted water volume with specified tangential momentum. External
pump forcing, localized Darcy intake resistance, passive outer reservoir
resistance and quadratic bed resistance are explicit momentum exchanges. The
feed-kernel-weighted mean input velocity is removed so the feed supplies torque
without a net horizontal jet. The outer resistance reaches a constant before
the periodic seams, damping momentum without adding walls or changing mass.
The model therefore conserves water volume up to numerical roundoff without claiming
conservation of angular momentum or mechanical energy in the forced patch.

Initialization integrates the radial pressure balance of a smooth finite-core
swirl and shifts the surface to preserve reference volume. Both pressure bowl
and circulation subsequently evolve with the water state. Limited MUSCL/HLL
finite-volume fluxes and SSP-RK2 advance the flow. Fixed 4 ms physical ticks
contain CFL-controlled substeps; both intermediate water states and the coupled
pigment candidate must be accepted before commitment. Feed is matched to actual
accepted suction at each stage, including its shallow-column reduction.

Pigment shares the water grid, actual face discharges, two accepted RK stages
and feed/suction sources. No independent sampled-current advection, empirical
impact wakes or extra eddy-diffusion pass is added in this mode. Rain deposits
color and a bounded horizontal momentum impulse without adding water volume.
The renderer uses the evolved height and depth; it adds no synthetic Drain foam.
[DRAIN.md](DRAIN.md) details numerical coupling, forcing, wet-column safeguards
and the reduced model's limitations.

## Optional river rapids

Rapids evolve `h`, `hu` and `hv` on a persistent 64×48 grid using the nonlinear
Saint-Venant equations with bottom-slope forcing. Hydrostatic reconstruction balances
bed pressure against a stationary free surface; conservative HLL interface fluxes,
limited MUSCL reconstruction and two-stage SSP-RK2 advance the flow. Fixed 4 ms ticks
contain CFL-controlled substeps with rejection of invalid depth states. The inlet
supplies discharge and weak correlated perturbations; the downstream boundary sets
stage in subcritical flow. The transverse direction is periodic.

The resulting current and shear respond to the evolving depth and momentum. Hydraulic
jumps appear as dissipative depth transitions. Near-surface backward motion inside
a roller and the creation/decay of foam use explicit empirical closures, described
in [RAPIDS.md](RAPIDS.md). They do not resolve a three-dimensional overturning barrel.
The full rapids state replaces the classic waves after the mode transition; fixed-depth
dispersive waves are not superposed onto the shallow numerical solution. The transition
itself is a visual blend, not a coupled physical boundary-value problem.

## Conservative pigment mixing

The persistent 96×64 pigment grid stores three passive areal masses `q_i`.
With water-column depth `h`, its volume concentration is `c_i = q_i/h`.
Drain advances `∂t q_i + ∇·(h u c_i) = F c_feed,i - D c_i - λ q_i` with MC-limited
concentration and the actual hydraulic face discharges. Clear feed dilutes the
column without adding pigment; suction removes `D c_i`. The scalar and water
share accepted stages and timesteps. Internal face fluxes are equal and opposite;
pump losses and optional test-feed supplies have separate cumulative ledgers.

Classic/Rapids instead evolve
`∂t q_i + ∇·(u q_i) = ∇·[κ h ∇(q_i/h)] - λ q_i`, using MC-limited face
reconstruction and SSP-RK2. Harmonic face depth makes diffusion follow the
volume-concentration gradient. Its positivity step bound accounts for MC
reconstruction and each cell's depth-weighted diffusive outflow.
[DYE.md](DYE.md) gives units, equations and closures for both transport paths.

Rapids contribute interpolated depth-averaged momentum divided by depth; their
empirical surface roller velocity transports foam rather than the full dissolved
column. The added Classic/Rapids compact wakes are a reduced stirring velocity,
not resolved momentum feedback or three-dimensional turbulent entrainment. Pigment remains
passive and does not change the water equations.

Transport conserves pigment apart from explicit pump removal/feed and open-boundary
outflow. Independently,
`λ = ln(2)/(8 physical seconds) × bleach` converts colored species into optically
untracked colorless products. `--bleach` ranges from 0 to 4, default 1; `v` / `b`
adjust it by 0.25 and `B` resets it. Uniform exposure and first-order decay are an
explicit phenomenological photodecoloration model, not a calibrated substance or
resolved chemistry. The default half-life is eight physical seconds, about 13.33
wall seconds at default speed. Exponential reaction half-steps surround transport;
cumulative converted mass and hydraulic/boundary exchanges have separate ledgers.
Positive reaction converts subnormal floating-point residues to the colorless
ledger to avoid a quantization stall; zero reaction bypasses that cleanup exactly.

In closed periodic Classic transport tests with bleaching disabled, the deposited
species inventory is conserved up to floating-point error, so ongoing input increases
mean absorption. Interactive Classic can export pigment through its open moving
window; Drain suction and Rapids outflow also remain removal paths with reaction
disabled. Positive bleaching bounds mean active color under bounded
continuous input. Transport has no arbitrary small-concentration discard rule;
the reaction's subnormal cleanup is explicitly accounted for.
Classic and Rapids use 8 ms coupling ticks for wake evolution, transport and
reaction, with Rapids hydraulic steps inside them. Drain consumes 4 ms coupled
ticks directly and owns the scalar
update while selected; there is no second mixing update. Switching away freezes
its hydraulic state and stops suction while its surface visually fades. Mode
transitions retain pigment but are visual blends, not physically coupled flows.
Coupled cadence invariance assumes identical input
timestamps; automatic interactive rain is still placed at display updates.

## Geometry and optics

Primary rays bracket the actual height slab and finite patch, then solve
`z_ray(t) = η(x_ray(t), y_ray(t))`. A bounded grazing-ray scan and residual check
prevent an unconverged hit from being shaded. Reflected secondary rays use the
same height field; no fixed offset plane substitutes for a second surface.

Normals are metric Scharr derivatives of these heights, with correct sample spacing
in both axes and one-sided boundary spans. Neither scene intensity nor terminal
dimensions scales the physical slope. The microfacet lobe uses exact unpolarized
dielectric Fresnel and height-correlated Smith-GGX masking. Refraction uses Snell's
law with a three-band water IOR approximation; absorption uses Beer–Lambert along
the transmitted path. Classic column depth increases at a crest over a fixed bottom;
Drain and Rapids supply their evolved `h` directly for primary and secondary optical paths.

Paraxial caustics use the full local Hessian in a refractive Jacobian determinant.
Focal singularities are regularized, and the finite-patch irradiance field is
normalized to redistribute incident flux. This is a normal-incidence/local mapping
approximation using the local water-column depth, not photon tracing through arbitrary bathymetry. Caustics affect
scattered irradiance; refracted environment radiance is not multiplied by a lens
gain. Volume scattering, film color and the luminous environment remain visual
approximations, so the whole image is not a calibrated radiometric measurement.
Rapids foam attenuates water radiance by coverage and contributes neutral diffuse
scattering from cached environment irradiance and the daylight source. Its albedo
is 0.86; it is not a white emitter. Secondary reflection paths apply the same
coverage split. Coverage-weighted surface-velocity backtracking helps temporal
reconstruction follow transported foam; clear water retains geometric reprojection.
Mixed pixels still share one history rather than separate optical/material layers.

## Numerical checks

Tests cover shallow/deep dispersion limits, group-speed derivatives, capillary
speed behavior, monotonically decreasing modal mechanical energy, spectral
quadrature convergence, impulse continuity, volume projection, vortex pressure
balance/divergence, frame-partition independence, metric plane normals, actual
surface-hit residuals, Fresnel/TIR limits, caustic flux normalization and deterministic
serial/parallel output. Runtime/PTY tests separately cover controls, resize and cleanup.
Rapids tests cover a lake at rest over uneven terrain, mass change versus integrated
boundary flux, positive finite depths, display-partition independence, momentum-only
impacts, bounded foam/decay, surface return flow and sustained late-time motion.
Material tests cover shallow-column attenuation/focusing, normalized diffuse lighting
and zero-foam identity. These checks establish specific numerical properties, not
full validation against laboratory measurements.
Pigment checks include closed species mass, integrated open outflow, filament
translation/shear, uniform-mixture preservation under periodic discrete-curl
flow, diffusion across unequal depths, dilute-mass retention and coupled frame
partitioning with matched inputs. Reaction checks cover half-life and rate scaling,
exact disable behavior, normalized transported shape, and active plus converted
plus exported species mass. Drain tests additionally exercise a finite pressure
bowl, matched pump volume accounting, inward current, absence of unforced spin,
shared-concentration transport, suction/bleaching mass accounting and rejection
without committing pigment. Subnormal reaction checks cover both scalar paths
and exact disable behavior. Wake tests cover quiescent impacts, periodic
seams, open-end tapering and bounded activity. These contracts live alongside the
implementation in [src](../src), including [water.rs](../src/water.rs),
[dye.rs](../src/dye.rs) and [drain.rs](../src/drain.rs).

Run the terminal package's checks from the repository root:

```sh
cargo check --locked -p acid-rain --all-targets
cargo test --locked --release -p acid-rain
```

Historical benchmark captures and generated reports are not distributed. A test
run verifies the checked-out implementation; older measurements do not establish
its current appearance, performance or long-run stability.

Classic mode assumes constant reference depth and weak wave nonlinearity. Strong
overlapping bursts can exceed its small-amplitude validity range even while numerical
values remain finite. Drain and Rapids add nonlinear depth-averaged flow, with
variable bathymetry in Rapids, but omit nonhydrostatic wave dispersion, molecular
viscosity and resolved three-dimensional turbulence. Drain does not resolve an
air core, drain pipe, vertical suction jet or pipe turbulence; its periodic domain
has no tank walls. All modes retain a single-valued surface and do not resolve
overturning crests, spray, individual bubbles, air entrainment, contact/wetting or
three-dimensional two-phase fluid motion. Numerical and empirical dissipation
must not be interpreted as a calibrated turbulent energy cascade.

## Primary sources

- [MIT, Akylas and Mei: Dispersion of Surface Water Waves](https://web.mit.edu/fluids-modules/waves/www/material/chap-3.pdf).
- [NIST DLMF 10.17: Bessel asymptotic expansions](https://dlmf.nist.gov/10.17).
- [PBRT, Specular Reflection and Transmission](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
- [Audusse et al. (2004), hydrostatic reconstruction for shallow-water flows](https://publications.imp.fu-berlin.de/478/).
- [Clawpack, approximate Riemann solvers for the shallow-water equations](https://www.clawpack.org/riemann_book/html/Shallow_water_approximate.html).
- [Clawpack, MC-limited advection example](https://www.clawpack.org/gallery/_static/notebooks/classic/advection_1d/advection_1d.html).
- [LeVeque, periodic advection method comparisons](https://depts.washington.edu/clawpack/clawpack-4.3/book/chap6/compareadv/www/).
