# Experimental river rapids

This describes the shared hydraulic model and its terminal rendering closures.

Start with `./run.sh --rapids`, or press `f` to switch between rapids and classic
water. Help (`h`) shows the selected mode. The six pastel worlds, quality controls,
speed controls and RGB echoes apply to both modes. Classic remains the default.

Rapids evolve a shallow current over a submerged sill and four staggered mounds. Changes in water
depth, currents and horizontal shear come from a persistent numerical flow. Foam
and colored rain make transport and surface return flow visible. The return flow
uses an empirical roller closure: the solver does not resolve the vertical loop
of a real hydraulic roller or an overturning wave tube.

## Variables and geometry

The implementation is [src/rapids.rs](../src/rapids.rs). Its conserved variables
are `U = (h, hu, hv)`, where `h` is column depth in metres and `u`, `v` are
depth-averaged horizontal velocities in m/s. `hu` and `hv` have units m²/s
(discharge per unit width, or depth-integrated momentum divided by density).
The free-surface elevation is `eta = h + z_b`, with fixed bed elevation `z_b`.
The rendered height is `eta - 0.10 m` plus zero-mean short-wave detail from the
breaking closure described below. Neither the datum nor that optical detail
changes conserved hydraulic depth.

| Quantity | Value |
| --- | ---: |
| Domain | 2 m × 2 m |
| Finite-volume cells | 64 × 48 |
| Cell size | 0.03125 m × 0.0416667 m |
| Gravity | 9.81 m/s² |
| Downstream free-surface stage | 0.09 m |
| Base inlet discharge per unit width | 0.045 m²/s |
| Physical tick | 0.004 s |
| Minimum accepted cell depth | 0.00001 m |

The bed combines a broad 0.039 m sill, whose crest position varies across the
channel, with four staggered submerged mounds. Each mound uses
`height exp(-((x-cx)/rx)² - ((y-cy)/ry)²)`; all values below are metres:

| Center x | Center y | Height | x width | y width |
| ---: | ---: | ---: | ---: | ---: |
| 1.02 | 1.12 | 0.0099 | 0.13 | 0.20 |
| 1.14 | 0.36 | 0.075 | 0.15 | 0.21 |
| 1.38 | 1.28 | 0.075 | 0.16 | 0.22 |
| 1.63 | 1.75 | 0.060 | 0.14 | 0.21 |

These smooth features split the current into several local hydraulic transitions;
they are not solid objects with resolved boundary layers. The initial current
uses the critical depth `(q²/g)^(1/3)` and subcritical/supercritical branches of the
specific-energy relation `h + q²/(2gh²) + z_b = E`, followed by the downstream
stage. A small transverse velocity seed breaks exact symmetry.

## Governing equations and update

The two-dimensional hydrostatic Saint-Venant system is

```text
∂t h     + ∂x(hu)             + ∂y(hv)             = 0
∂t(hu)   + ∂x(hu² + g h²/2)   + ∂y(huv)            = -g h ∂x z_b
∂t(hv)   + ∂x(huv)            + ∂y(hv² + g h²/2)    = -g h ∂y z_b
```

Each interface uses one conservative HLL flux. Its left/right signal bounds
enclose both states' normal velocities plus/minus `sqrt(g h)` and zero. The
flux uses `(s_R F_L - s_L F_R + s_L s_R (U_R - U_L))/(s_R - s_L)`.
The shallow-water wave speeds and HLL construction are described in the
[Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water_approximate.html).

Hydrostatic reconstruction uses the larger adjacent bed height, reconstructs
nonnegative interface depths and adds the corresponding bed-pressure corrections
on each side. This follows the approach of
[Audusse et al. (2004)](https://publications.imp.fu-berlin.de/478/).
Primitive reconstruction in `(eta, u, v)` uses limited MUSCL slopes; slope limiting
also keeps reconstructed depths positive. A two-stage SSP-RK2 update advances
both spatial directions together. Internal face mass fluxes cancel, while the
boundary mass flux is accumulated separately for conservation tests.

Within each integer 4 ms tick, the substep satisfies

```text
dt <= 0.38 / max_cells[(|u| + sqrt(g h))/dx + (|v| + sqrt(g h))/dy]
```

Both Runge–Kutta stages must have finite momentum and depth at least `1e-5 m`.
An invalid stage is rejected and retried at half the substep; the code does not
repair it by adding water through a depth clamp. The model is designed for this
wet channel, not a general shoreline wetting/drying problem.

The tick remainder is retained as a `Duration`, so equivalent elapsed time
partitions take the same physical steps. Resizing only resamples the persistent
solver grid. Switching modes blends surface height, optical depth and foam over
0.45 physical seconds; the blend is a presentation transition. Once disabled,
the rapids state waits until enabled again. Pausing holds simulation time.

## Boundaries, forcing and rain

Flow enters at `x = 0`. The base feed is

```text
q0(y) = 0.045 [1 + 0.15 cos(2 pi y / 2 m) + 0.06 sin(4 pi y / 2 m)] m²/s
```

This lateral variation supplies shear. Eight deterministic, bounded correlated
noise values perturb it by `q(y,t) = q0(y) [1 + 0.20 n(y,t)]`.
The values use a recurrence with decay `exp(-0.004/0.8)` and normalized uniform
innovations, then clamp to `[-2.5, 2.5]`; smooth periodic interpolation supplies
the inlet rows. Thus the maximum relative perturbation is 50%, with a 0.8 s
correlation-time parameter. This is prescribed unresolved upstream variability,
not a simulated turbulent cascade or exact Gaussian Ornstein–Uhlenbeck process.
Its random sequence advances on physical ticks, independently of render cadence.

At the subcritical inlet, the prescribed discharge is combined with the outgoing
characteristic `u - 2 sqrt(g h)` to recover boundary depth. Subcritical outflow
uses the fixed downstream stage and `u + 2 sqrt(g h)`; supercritical outflow
extrapolates the interior state. The transverse direction is periodic. These
boundaries maintain an open current and are not walls of a closed basin.

Rain applies a localized outward horizontal momentum kick. It does not add or
remove water. `--drop-size` or `,` / `.` scales its Gaussian footprint independently
of rain rate; `Up` / `Down` changes rate and `R` resets both settings. The RMS kick
radius is `(0.055 + 0.035 sqrt(J)) d` metres for strength `J` and diameter multiplier
`d`. This is an inexpensive disturbance of the shallow flow; the classic mode's
dispersive Hankel impulse response is not added to rapids. Each impact also adds
passive pigment, whose actual surface-current transport, mass normalization and
absorption are documented in [DYE.md](DYE.md).

## Breaking detail

Hydraulic jumps can approach a visually quiet mean shape even while carrying
substantial current. An added stochastic closure uses the resolved compression,
strain and jump source to excite local short-wave packets and patches of foam.
It is driven by the flow, with a deterministic random sequence in physical time;
it is not a global oscillation added to every part of the water.

Packets have Gaussian radii of 0.055–0.10 m and signed, compensated vertical-speed
impulses of 0.50–1.20 m/s. Their event rate is proportional to the resolved source,
with a coefficient of 0.012. A narrow-band wave model centered on wavelength
0.18 m propagates the detail using `c² = g tanh(k h)/k`, with `k = 2 pi / 0.18 m`,
and damps vertical speed with time constant 0.65 s. Semi-Lagrangian advection transports displacement
and vertical speed with the same surface current as the foam. The displacement
and vertical-speed means are removed after each update. A depth limiter caps the
optical displacement at 20% of local hydraulic depth and at 15 mm, dissipating
outward vertical speed when it would continue beyond the bound. Repeated clipping
and mean removal retain the local amplitude bound and zero mean within numerical
projection tolerance. The limiter represents unresolved breaking losses.

This changes rendered height and normals but leaves conservative `h`, `hu` and
`hv` untouched. The hydraulic depth still controls absorption and caustics. The
closure is not a solved overturning wave, spray model or two-way transfer of a
calibrated breaking-energy budget. Its excitation and dissipation are empirical.

## Rollers and foam

The hydraulic jump in `h`, `hu`, `hv` is a dissipative depth transition. Its
vertical overturning cannot be represented by depth averaging. A separate
surface-tracer closure detects compression, a downstream rise in surface height
and a near/supercritical upstream Froude number `|u|/sqrt(g h)`.

The resulting bounded roller strength extends downstream with an e-folding
length of `2.8 h`. Surface velocity blends the mean `u` toward
`-0.55 sqrt(g h)` using weight `min(1.65 × roller, 0.95)`; transverse velocity
remains `v`. This allows foam to return upstream while the bulk current continues
downstream. It does not insert an unresolved lower return branch into the
conservative momentum equations, and its coefficients are not laboratory-fitted.

Foam coverage `C` is a passive visual field in `[0,1]`. Sources depend on the
roller and resolved compression/strain. Each accepted breaking packet adds a
localized coverage `C += (1-C) 0.90 exp(-r²/2)` before advection. The continuous
production rate uses 0.08 times the resolved source, allowing clear gaps between
coherent foam patches. Bilinear semi-Lagrangian backtracking
uses the surface-tracer velocity, with periodic transverse coordinates and
clear incoming foam. With source rate `s` and decay time `tau = 3.2 s`, its
split update is

```text
C_new = 1 - (1 - C_advected exp(-dt/tau)) exp(-s dt)
```

This keeps coverage bounded and gives exponential decay when production stops.
It is not a conservative bubble-number or air-volume solver; interpolation also
introduces numerical diffusion. The renderer blends water radiance by coverage
with passive diffuse foam scattering, using neutral base albedo 0.86 and cached
environment lighting. Dissolved pigment attenuates that diffuse light through
an effective liquid-film path factor of 1.5 in both primary and secondary foam
shading. This empirical absorption closure exposes colored drops in aerated
water; it does not emit light or resolve individual bubble films. Secondary
reflection paths also attenuate their remaining water contribution by foam
coverage. Actual `h` controls primary/secondary optical
path lengths and local paraxial caustic focusing. Temporal backtracking is weighted
by foam coverage: clear water keeps geometric reprojection, while covered regions
follow the tracer velocity. A single mixed-pixel history remains an approximation
to separate material and reflected-light motion. The existing pastel illumination and linear
color pipeline remain in use.

## Scope and checks

The model resolves nonlinear, depth-averaged flow and numerically captured
hydraulic transitions at its grid scale. It omits vertical accelerations,
overturning surfaces, resolved spray/bubbles/air entrainment, bottom boundary
layers, molecular viscosity, capillary dispersion and a calibrated turbulence
closure. HLL dissipation and grid filtering suppress unresolved motion; visual
complexity alone is not evidence of resolved three-dimensional turbulence.

Tests exercise resting water over the bed, water-volume change against boundary
flux, positive finite states, fixed-tick partition invariance, momentum-only
impacts of different sizes, foam bounds/decay, moving zero-mean breaking detail
and surface reverse flow. A long-run test checks that
the prescribed inlet fluctuations maintain changing depth fields. These are
numerical checks of this implementation, not a laboratory validation suite.

Run the contracts in [rapids.rs](../src/rapids.rs) and the water integration
tests from the repository root:

```sh
cargo check --locked -p acid-rain --all-targets
cargo test --locked --release -p acid-rain rapids::tests
cargo test --locked --release -p acid-rain water::tests
```

Historical captures, frozen comparison binaries and generated timing reports are
not distributed. The scripts for historical A/B studies require their matching
archives and provenance; they are not fresh-checkout validation commands. CPU
frame-preparation timing also excludes terminal transport and emulation and does
not establish displayed frame rate.
