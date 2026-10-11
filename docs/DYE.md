# Colored rain and passive pigment

This document describes the terminal renderer's depth-integrated pigment model.
The [native](../native/README.md) and [desktop](../desktop/README.md) GPU renderers
use a separate three-dimensional tracer volume.

Every new impact carries a mixture of cyan, magenta and amber pigments. Currents
stretch those mixtures into folds and filaments. Drain mode transports
pigment with the same accepted face discharges as water; Classic and Rapids add
reduced impact wakes and strain-dependent unresolved mixing. Pumped-out pigment
and open-boundary outflow have separate mass ledgers. An adjustable reaction
converts active color to unrendered colorless products. Color comes from selective
absorption of light passing through water. The pigment does not emit light or
paint a blurred trail over the screen.

```sh
./run.sh --rain 12 --drop-size 1.5
./run.sh --rapids --rain 16 --drop-size 1.25
./run.sh --rain 12 --bleach 2
./run.sh --water drain --bleach 0 # Disable reaction; suction still removes pigment.
./run.sh --water classic --bleach 0 # Disable reaction; the open window can export pigment.
```

| Control | Effect |
| --- | --- |
| `Up` / `Down` | Raise/lower rain by 2 impacts per physical water second, within 0–60 |
| `,` / `.` | Decrease/increase relative impact diameter by 0.25, within 0.25–3 |
| `R` | Reset rain to 32 impacts/s and diameter to 1.5× |
| `r` | Add a burst of 12 drops at the current diameter |
| `v` / `b` | Decrease/increase bleaching by 0.25×, within 0–4; 0 disables it |
| `B` | Reset bleaching to 1×: an eight-physical-second half-life |
| `d` | Enter Drain; return to Classic when already in Drain |
| `f` | Enter Rapids; return to Classic when already in Rapids |
| `h` / `?` | Show current settings and all controls |

`--water drain|classic|rapids` selects the initial model; Classic is the default.
`--rapids` remains shorthand for `--water rapids`. Mode changes retain pigment.

Rain rate counts impacts per **physical water second**, not wall-clock second.
At the default scene speed of 2.4×, physical water time runs at 0.6× wall time. Rate zero
stops automatic impacts; existing waves and pigment continue evolving. Diameter
changes affect new drops. Existing drops keep their original sizes. While paused,
the controls update their settings and status without advancing pigment transport.
Minimizing the terminal to an empty viewport does not pause physical time: rain,
water and pigment keep advancing together, and resizing back resamples their state.

Classic wave rendering retains at most 128 live impact responses. Heavy rain or
repeated bursts can evict the oldest responses before their normal 10-second
decay finishes; at a sustained 60 impacts per physical second the queue spans
about 2.13 seconds. This bounds work but can cut an old classic ripple early.
Deposited pigment persists in its independent transport grid and is not removed
when its originating ripple leaves that queue.

## Impact diameter and source mass

The diameter setting scales the impact footprint, not a calibrated falling
raindrop measured in millimetres. With strength `J` bounded to `[0,1]` and diameter
multiplier `d`, the Gaussian width parameters `sigma` in metres are:

```text
classic momentum impulse: (0.035 + 0.045 sqrt(J)) d
drain/rapids momentum:    (0.055 + 0.035 sqrt(J)) d
pigment source:             0.025 (0.65 + sqrt(J)) d
```

The classic spectral response uses the changed spatial support. Drain and Rapids
receive an outward horizontal momentum kick of the changed width. Rain adds no
water volume in any mode. The pigment amount supplied by `Water::add_drop_sized` is
`A = 0.006 (J / 0.2) d³` in normalized pigment-mass units. The volume scaling is
a source model for larger colored drops, not a calibrated impact/entrainment law.
The deterministic sequence chooses six mixtures of the three species.

`Dye::deposit` normalizes its Gaussian so that the grid integral is exactly the
requested mass up to floating-point error, regardless of radius or clipping at
an open end. Species weights are nonnegative and sum to one after normalization.
Periodic seams wrap the kernel. Pigment footprints narrower than half the smaller
cell width use that minimum radius, approximately 1.04 cm, to avoid subcell
aliasing. This is a resolution limit, not a claim that tiny real drops have that size.

## Transport and numerical scheme

[src/dye.rs](../src/dye.rs) stores three nonnegative areal pigment densities
`q_i` on a persistent 96×64 grid covering 2×2 m. Let `P` denote the arbitrary
pigment-mass unit: `q_i` has units `P/m²`, its area integral has units `P`, and
the local volume concentration is `c_i = q_i/h` in `P/m³`, and transport velocity
`u` has units m/s. All three species are passive: pigment does not feed back on
water momentum, density, viscosity or surface tension. Water modes use different
transport paths, described below; they share the independently controlled reaction

```text
λ = [ln(2) / (8 seconds)] × bleach
```

This is one spatially resolved column with three passive colored species. Internal
transport conserves their mass; the first-order reaction reduces the active
colored inventory. `--bleach` sets its multiplier in `[0,4]`, default 1.
For a positive multiplier, the reaction half-life is `8/bleach` physical seconds;
at 1× that is about 13.33 wall seconds with the default speed. Setting it to zero
disables reaction exactly, without changing water transport or pump removal.

The reaction represents phenomenological photodecoloration under uniform effective
exposure. It is not calibrated to a particular substance, illumination or chemical
reaction network. Colorless products are not spatially resolved or rendered. A
cumulative species ledger records converted mass, rather than attributing the
decrease to mixing. Exponential half-steps surround each 8 ms Classic/Rapids
transport tick or each accepted Drain hydraulic substep. The ledger uses the
actual stored-field decrease, including floating-point rounding. With positive
bleaching, subnormal residues below `f32::MIN_POSITIVE` (about `1.175×10⁻³⁸`)
are converted to the colorless ledger instead of becoming stuck at quantized
values. Zero bleaching bypasses this cleanup exactly.

Active mass plus cumulative converted, pumped-out and open-boundary exported
mass accounts for initial and deposited mass plus any pigment supplied by feed,
up to transport roundoff. Production Drain feed is clear; the supplied-pigment
ledger supports explicit colored-feed transport tests.

With bleaching disabled, continuing colored rain in the closed periodic Classic
test model increases mean column mass and absorption. Interactive Classic uses
an open resident window: outflow and departing cells export pigment, while newly
entered cells start clear. `--no-travel` fixes the camera's travel but does not
make transport periodic. Stopping rain stops new mass; mixing alone does not make
pigment disappear. Positive bleaching bounds the long-term mean active
inventory under bounded continuous input, although stronger rain can still darken
the water. Drain removes pigment through suction and supplies clear replacement
water; Rapids export pigment through open channel ends. Both can reduce active
inventory even when bleaching is disabled.

## Drain: shared hydraulic transport

[Drain](DRAIN.md) evolves water and pigment on the same 96×64 periodic
grid. With depth rates `F` for feed and `D` for suction, it advances

```text
∂t q_i + ∇·(h u c_i) = F c_feed,i - D c_i - λ q_i
species_face_flux = water_face_discharge × c_upwind
c_i = q_i/h
```

Monotonized-central (MC) reconstruction limits volume concentration at each
face. The face discharge comes directly from the water HLL solver; both scalar
SSP-RK2 stages use the matching water depths, fluxes and sources. This makes
uniform concentration follow the same evolving water volume. It also avoids
inventing scalar compression from an independently interpolated velocity field.

Clear feed adds depth but no species mass. Suction removes the local concentration
times the accepted removed water volume; its integrated species loss is tracked
in `f64`. Feed and drain use the same accepted water-volume rate at each stage.
Both hydraulic stages and the scalar candidate must be valid before either is
committed; rejection and retry do not alter committed pigment or its ledgers.
The outgoing material-rate bound includes suction and margin for MC face values.

Drain does not add molecular/eddy diffusion or the empirical wakes described
below. Resolved advection stretches color, while limited finite-volume transport
introduces numerical diffusion. This is reduced horizontal transport, not a
resolved three-dimensional turbulent mixing cascade.

## Classic/Rapids: advection and unresolved diffusion

The other two modes retain the separate transport equation

```text
∂t q_i + ∇·(u q_i) = ∇·[κ h ∇(q_i/h)] - λ q_i
```

Rapids supply depth-averaged velocity, computed as interpolated horizontal
momentum divided by interpolated depth. The empirical roller return velocity
continues to move surface foam; it is not applied to the entire dissolved column.
The non-streaming Classic test model uses a periodic finite-core current,
described below; interactive Classic samples currents and local flow structures
in world coordinates through its open window. Their
transitions blend these currents and column depths; bounded impact wakes add
reduced stirring. Entering Drain switches to its shared hydraulic transport.

Face fluxes combine upwind advection from monotonized-central (MC) linear
reconstruction with centered concentration-gradient diffusion. The slope is
`minmod(2 delta_left, (delta_left + delta_right)/2, 2 delta_right)`.
It retains more resolved filament structure than the previous minmod limiter,
while suppressing spurious oscillations at steep fronts. The role of MC limiting
and the diffusion/oscillation tradeoff are illustrated by the official
[Clawpack advection notebook](https://www.clawpack.org/gallery/_static/notebooks/classic/advection_1d/advection_1d.html)
and [LeVeque's periodic advection comparisons](https://depts.washington.edu/clawpack/clawpack-4.3/book/chap6/compareadv/www/).

Two-stage SSP-RK2 advances all species. Each internal face contributes equal and
opposite mass flux to its two cells. The diffusive face conductance is
`K_face = (κ_left + κ_right)/2 × harmonic_mean(h_left, h_right)`;
its flux is `-K_face × (c_right - c_left)/distance`. Thus a uniform volume
concentration over a varying stationary depth does not spuriously diffuse simply
because its column mass varies. The adaptive substep uses

```text
a_cell = [max(u_right,0) + max(-u_left,0)]/dx
       + [max(v_top,0) + max(-v_bottom,0)]/dy
d_cell = (K_left + K_right)/(h_cell dx²)
       + (K_bottom + K_top)/(h_cell dy²)
dt <= 0.8 / [2 max_cells(a_cell) + max_cells(d_cell)]
```

The factor 2 bounds outgoing MC-reconstructed column density. The condition
accounts for every outgoing advective and diffusive face rate, including shallow
cells beside deeper ones. It keeps both stages nonnegative without clipping
negative concentrations and silently changing mass. A converging transport field
can raise local column density above its initial maximum.

The strain-based subgrid diffusivity, in m²/s, is

```text
κ = 10⁻⁹ + min[(0.3 sqrt(dx dy))² |S|, 0.0015]
|S| = sqrt[2 ((∂x u)² + (∂y v)²) + (∂y u + ∂x v)²]
```

This Smagorinsky-style closure responds to deformation, not rigid-body rotation.
Interior gradients use centered differences; open-end x gradients use the
actual one-sided spacing. The coefficient and cap are graphics-model choices,
not a turbulence calibration for a particular dye or river. Numerical transport
also introduces diffusion, especially near limited, steep concentration fronts.

Classic mode uses periodic boundaries. Rapids use periodic transverse boundaries,
clear-water inflow and advective outflow at either open x end, with zero diffusive
flux through those ends. Thus outgoing pigment is removed rather than recycled.
The implementation accumulates species outflow separately from bleaching for
its mass-balance checks. Transport has no arbitrary low-concentration discard
threshold. Exactly empty fields skip transport; nonzero dilute fields remain
represented, subject to the configured reaction and its subnormal-residue cleanup.
With bleaching disabled, that cleanup is also disabled.

## Classic/Rapids currents and compact impact wakes

These additional stirring fields are used only outside Drain. Classic pigment
transport uses a periodic streamfunction whose chord distances
match Euclidean distance near each original finite core. This changes the distant
mixing current to join correctly across the patch seams; the analytic wave
geometry and its pressure bowls remain the existing model.

The base current and wakes are assembled into one vertex-grid streamfunction
`psi`, in m²/s. Face velocities are its discrete curl:

```text
u_face = -(psi_top - psi_bottom)/dy
v_face =  (psi_right - psi_left)/dx
```

Adjacent differences cancel in the cell divergence, up to floating-point error.
Periodic edge vertices represent the same values. This avoids seam compression
from wrapping a nonperiodic infinite-plane vortex field, and preserves a uniform
mixture under the periodic incompressible component. The rapids mean current
remains an additional, generally compressible depth-averaged velocity.

[src/stirring.rs](../src/stirring.rs) stores at most 16 pairs of oppositely signed
compact blobs. Each has streamfunction
`psi_blob = A [max(1 - r²/R², 0)]³`. It is C2 at the support boundary and its
velocity is C1. Pair orientation follows the pre-impact column current; support
scales with the pressure footprint, and amplitude depends on local current speed
and impact strength. A symmetric vertical impact in quiescent water seeds no
horizontal swirl.

All centers advance from a common midpoint state under the mean current and the
other wake contributions. Cores spread, amplitudes ramp in and then damp, and
old wake activity retires smoothly. A spatial upper bound scales the entire
streamfunction when needed, rather than clipping velocity components and breaking
incompressibility. Wake-only speed is bounded at 0.36 m/s. Open-channel wakes
taper to zero at the x ends. Only compact support regions are written to the
vertex grid; heavy bursts keep the bounded wake budget by replacing sufficiently
weak pairs or omitting an additional unresolved wake. Every valid pigment deposit
still enters the conservative species field.

These are empirical two-dimensional mixing wakes, not a resolved splash vortex
ring, a momentum-conserving fluid feedback model or a three-dimensional turbulent
cascade. Wake damping removes stirring activity; the independently configured
bleaching reaction controls the conversion of colored pigment.

## Shared physical time

Water coupling, transition progress, wake evolution, pigment transport and bleaching
share integer 8 ms physical ticks and a retained fractional remainder. Drain and
Rapids take their own 4 ms hydraulic ticks inside each coupled tick. Selected
Drain owns the pigment update through its accepted hydraulic stages; it receives
no second advection/diffusion update or empirical wake forcing. Leaving Drain
freezes its saved hydraulic state and stops its suction immediately, even while
its last surface fades visually. Mode transitions are visual blends, not coupled
physical boundary problems.

In Classic/Rapids, the current is updated before that tick's wake and pigment
transport, so a long display frame does not apply its final current retroactively
to older transport ticks. Analytic wave time follows accumulated physical duration.

Equivalent display-frame partitions produce the same coupled result when input
events occur at the same physical timestamps. Automatic interactive rain is
still scheduled at display updates, so changing display cadence can change the
actual impact timestamps; it is not an identical forcing history. Resizing only
resamples stored fields and introduces no extra transport or diffusion. Pause
does not advance physical ticks. Nonfinite inputs are rejected or sanitized,
and the transport API retains finite velocity/depth guards.

## Optical absorption

The renderer samples pigment mass at the water hit. For local hydraulic depth
`h`, the corresponding uniform column concentration is `q_i/h`. If the optical
path inside that column is `ell`, its species transmittance is

```text
tau_dye,rgb = 0.85 (ell/h) Σ_i alpha_rgb,i q_i
T_rgb = exp(-tau_dye,rgb)
```

The base extinction coefficients `alpha` have units `m²/P` under the normalized
mass convention. A shared 0.85 scale modestly reduces absorption. Their broad
RGB bands are deliberately soft artistic spectra:

| Species | Red | Green | Blue |
| --- | ---: | ---: | ---: |
| Cyan | 2.00 | 0.65 | 0.30 |
| Magenta | 0.30 | 1.80 | 0.45 |
| Amber | 0.25 | 0.65 | 1.80 |

For a locally uniform column, `ell/h = 1/cos(theta_t)` from the refracted angle.
This ratio prevents a shallow hydraulic column from losing pigment opacity merely
because it is shallower: the same areal mass occupies a higher volume concentration.
Zero pigment gives unit transmission exactly. Nonnegative pigment can only
attenuate light; it adds no emissive energy. Background radiance at primary and
secondary water encounters receives the full-path Beer factor. Existing volume
illumination distributed through the column instead receives

```text
volume_factor = E(tau_water + tau_dye) / E(tau_water)
E(tau) = (1 - exp(-tau))/tau, with E(0) = 1
```

Light generated or scattered nearer the exit traverses less absorbing pigment
than background light, so the volume factor is generally larger than full-path
Beer transmission. Both factors remain in `[0,1]` and approach zero for increasing
pigment at fixed water optical depth. This introduces no brightness floor or
light source: zero original volume illumination stays zero. It is a homogeneous,
uniform-source approximation, not a full three-dimensional scattering solve.
Direct surface reflection and dielectric Fresnel retain their own response.
[src/pigment_optics.rs](../src/pigment_optics.rs) shares these factors between
primary and secondary shading; [OPTICS.md](OPTICS.md) describes the light paths.

Diffuse foam also passes through a pigment absorption factor: the liquid films
around bubbles contain the same dissolved species. Both primary and secondary
foam shading use the sampled pigment and an effective normalized path factor
`ell/h = 1.5`, then combine that attenuated diffuse radiance with water according
to foam coverage. This keeps drops visible in an aerated roller without making
the pigment luminous. The factor is an empirical optical closure, not a measured
bubble-film thickness or a separate film-concentration solver. Zero pigment
preserves the original foam response. Mixing occurs before the existing
linear-light color pipeline.

This is a sampled-column approximation. It does not integrate heterogeneous dye
along a full bent ray through three-dimensional water, resolve chemical kinetics
beyond the explicit first-order reaction, model individual dissolved molecules
or solve three-dimensional turbulent mixing.

## Checks and evidence

Tests cover deposited species mass and radius, periodic seams, closed mass
conservation, integrated open-boundary outflow, velocity-correct centroids,
translated filament shape, analytic shear covariance, diffusion variance `2 κ t`,
depth-weighted diffusion and nonnegative variable-flow transport. Additional
checks cover uniform concentration under discrete-curl transport, retention of
arbitrarily dilute species, strain versus rigid rotation and invalid inputs.
Reaction checks cover the eight-second default half-life, rate scaling and exact
disable behavior, preservation of normalized transported shape, and the combined
active/bleached/exported species ledger. Closed-mass transport checks disable bleaching.
Drain coupling tests additionally exercise concentration consistency with water,
clear-feed dilution, actual face-discharge advection, suction/bleaching accounting,
positive filaments and isolation of rejected candidates. Subnormal reaction tests
cover both transport paths and exact preservation with bleaching disabled.
Wake tests exercise the quiescent-impact limit, coherent paired motion, seams,
open-end tapering and the bounded source budget. Integration tests cover matched
input/frame partitions, pause, resizing, mixed drop sizes, bursts and mode changes;
optical tests cover passive Beer absorption, depth-normalized path length and
attenuation of both primary and secondary diffuse foam lighting. Shared pigment
optics tests cover clear-water identity, passive bounds, monotonic attenuation,
thin/thick limits and absence of emission from a zero source.

Run the contracts in [dye.rs](../src/dye.rs), [stirring.rs](../src/stirring.rs),
[water.rs](../src/water.rs) and [pigment_optics.rs](../src/pigment_optics.rs)
from the repository root:

```sh
cargo check --locked -p acid-rain --all-targets
cargo test --locked --release -p acid-rain dye::tests
cargo test --locked --release -p acid-rain stirring::tests
cargo test --locked --release -p acid-rain water::tests
cargo test --locked --release -p acid-rain pigment_optics::tests
```

Historical concentration-field captures, A/B archives and generated reports are
not distributed. These numerical contracts do not establish laboratory accuracy,
whole-renderer appearance or current performance; those require separate evidence.
