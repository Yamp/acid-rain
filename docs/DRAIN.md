# Pump-fed whirlpool

Drain is an optional water mode; Classic is selected at startup. Drain evolves the pressure bowl, inward current
and horizontal circulation together, while colored rain is carried by the same
water-volume fluxes. The pump removes water through a finite central disk and
replaces the accepted volume through an outer annulus with tangential momentum.
This is a forced, depth-averaged shallow-water model, not a simulation of a
three-dimensional drain pipe or an air core.

```sh
./run.sh                         # Default: Classic
./run.sh --water drain --rain 12
./run.sh --water classic         # Dispersive waves and prescribed currents
./run.sh --water rapids          # Experimental open-channel flow
```

`d` enters Drain from another mode and returns to Classic when already in Drain.
`f` similarly toggles Rapids and Classic. `--rapids` remains a shorthand for
`--water rapids`; it cannot be combined with `--water`. Rain, diameter, bleaching,
quality and speed retain their existing controls; `h` / `?` shows their current
values. The three requested water modes are mutually exclusive.

## State and forcing

[src/drain.rs](../src/drain.rs) stores water depth `h` and horizontal momentum
per unit density `m = h u` on the same persistent 96×64 grid as pigment. The
domain is a 2×2 m flat-bottom patch, periodic in both directions. Grid spacing
is approximately 2.08×3.13 cm; terminal dimensions and quality settings do not
change this physical grid.

With feed and drain depth rates `F` and `D` in m/s, the equations are

```text
∂t h + ∇·(h u) = F - D
∂t (h u) + ∇·(h u⊗u + ½ g h² I)
    = F u_feed - D u - (γ_intake + γ_rim) h u - C_drag |u| u
```

Gravity is 9.81 m/s². The last term is a quadratic bottom-stress closure; it
removes momentum. Localized linear Darcy resistance `γ_intake` represents an unresolved
porous intake. It uses the unnormalized compact suction shape,
`γ_intake(r) = 2.5 s⁻¹ × [max(1-r²/R_d²,0)]³`, so it vanishes smoothly outside the
intake. This explicitly removes horizontal momentum near the outlet rather
than clipping circulation or imposing a pressure bowl. Its strength is an
empirical reduced-model parameter, not a measured drain-pipe law.

A second passive resistance represents the rough outer return reservoir:

```text
s = clamp[(r - 0.82 m)/(0.15 m), 0, 1]
γ_rim(r) = 0.5 s⁻¹ × s³ (10 - 15s + 6s²)
```

It ramps smoothly from zero to a constant before the periodic seams, reaching
its full value at `r = 0.97 m`. The constant continuation makes the resistance
and its derivatives match across opposite edges. It damps momentum without
changing water or pigment mass; it does not turn the periodic patch into a
walled tank or prescribe a desired vortex shape. Both linear resistance terms
are dissipative external momentum closures.

Suction removes the local water's momentum, rather than leaving its momentum
behind while reducing its depth. The feed injects its
specified, predominantly tangential momentum. Thus the pump, feed and bed are external
momentum exchanges; water volume conservation does not imply conservation of
the patch's momentum, angular momentum or mechanical energy.

| Parameter | Value |
| --- | ---: |
| Reference mean depth | 0.16 m |
| Nominal pump discharge | 0.008 m³/s |
| Suction/resistance support radius `R_d` | 0.24 m |
| Initial swirl core radius | 0.14 m |
| Initial peak tangential speed | 0.55 m/s |
| Feed annulus center radius / support half-width | 0.74 m / 0.16 m |
| Baseline tangential feed speed | 0.40 m/s |
| Feed fluctuation correlation time | 0.65 physical s |
| Quadratic bed coefficient `C_drag` | 0.008 |
| Maximum intake resistance `γ_intake` | 2.5 s⁻¹ |
| Outer reservoir resistance `γ_rim` | 0.5 s⁻¹ |
| Outer resistance ramp start / end | 0.82 m / 0.97 m |
| Minimum accepted wet depth | 0.005 m |

The compact suction kernel's RMS radius is `R_d/sqrt(5)`, approximately 10.7 cm,
or over three coarse-axis cells. It is a resolved uptake region rather than a
point sink. Nominal pump discharge is an upper bound; local shallow-depth
throttling can reduce the accepted rate.

The central suction kernel is `[max(1-r²/R_d²,0)]³`, normalized by cell area.
Suction weakens smoothly as a column becomes shallow. At **each Runge–Kutta
stage**, the accepted drain rate is summed and used to normalize the clear-water
feed rate:

```text
Q_accepted = Σ_cells D × cell_area
Σ_cells F × cell_area = Q_accepted
```

The feed occupies a smooth annulus with a fixed, mild angular variation. Its
tangential inlet velocity carries bounded, temporally correlated perturbations
in four low angular modes. A fixed random seed and a physical-time update make
that forcing independent of display cadence. These perturbations represent
unsteady incoming momentum; they do not prescribe moving vortex centers or
add a hidden stirring field after transport.

Each forcing update subtracts the feed-kernel-weighted mean velocity from all
feed velocities. Thus opposed feed sectors supply circulation with zero net
horizontal feed momentum up to roundoff, while retaining local shear. The
correction balances the pump input; it neither projects the evolving water
velocity nor recenters the evolved vortex.

The initial state uses a smooth, finite-core tangential velocity, tapered to
zero before the domain edges. Its pressure depression is obtained by radial
integration of `dη/dr = vθ²/(g r)`, then shifted by a constant to preserve the
reference water volume. This initializes a wet bowl rather than a singular
point vortex. The discrete solver subsequently evolves both depth and momentum;
the initial analytic bowl is not reapplied every frame.

## Water discretization and accepted steps

Limited MUSCL reconstruction and conservative HLL face fluxes advance the
flat-bed Saint-Venant equations with two-stage SSP-RK2. Each shared face has
one water-volume flux, including matching copies at periodic seams. The model
uses fixed 4 ms physical ticks, each containing adaptive substeps constrained
by gravity-wave propagation and outgoing material flux.

Both intermediate water states must remain finite and above the model's
minimum wet depth. The second stage must also satisfy its own step bound.
Pigment must accept those same two hydraulic stages before either system is
committed; a rejected step is retried at half the duration. Rejection does not
advance the pigment field or its cumulative ledgers. This is a wet-column
model with a guarded timestep, not a wetting/drying or free-boundary algorithm.

The gravity-wave bound is based on
`(|u_x|+sqrt(g h))/dx + (|u_y|+sqrt(g h))/dy`. A separate material bound includes
every outgoing face discharge and the local suction rate divided by depth,
with margin for the pigment reconstruction. A third bound includes explicit
momentum damping:

```text
dt <= 0.34 / max_cells(wave_rate)
dt <= 0.40 / max_cells[(outgoing_water_rate + D)/h]
dt <= 0.40 / max_cells[D/h + γ_intake + γ_rim + C_drag |u|/h]
```

The source bound prevents an unresolved resistance timescale from reversing
momentum in a single explicit update. Neither terminal frame rate nor
optical quality weakens these hydraulic constraints. Fixed grids bound storage;
the number of accepted CFL substeps still depends on the evolving flow.

## Pigment follows the accepted water flux

For species areal mass `q_i`, volume concentration is `c_i = q_i/h`. Drain uses

```text
∂t q_i + ∇·(h u c_i) = F c_feed,i - D c_i - λ q_i
λ = ln(2)/(8 physical seconds) × bleach
```

The actual numerical species flux is `water_face_discharge × c_upwind`.
Monotonized-central reconstruction limits **volume concentration**, not column
mass. The hydraulic face discharge, both SSP-RK2 stages and the accepted duration
are shared with the water solver. A spatially uniform concentration therefore
follows the same moving water volume; scalar transport does not substitute an
independently sampled current for that volume flux.

Clear feed has `c_feed = 0`. It increases water volume locally without adding
pigment; suction removes local concentration times the accepted removed water
volume. Exact exponential reaction half-steps surround each accepted transport
step. Separate cumulative species ledgers account for pumped-out pigment,
optional supplied pigment in transport tests, and bleaching. Ordinary rain
deposits are independent sources. For each species, up to floating-point error,

```text
active + bleached + drained + open_boundary_export
    = initial + rain_deposited + feed_supplied
```

Drain has no open-boundary export because its boundaries are periodic. Clear
feed and suction can remove color even with `--bleach 0`; that option disables
only the reaction. Positive bleaching also converts subnormal floating-point
residue to the colorless ledger to avoid a quantized decay stall. Zero bleaching
bypasses this cleanup and preserves such dilute values.

There is **no added eddy/molecular diffusion or compact empirical impact wake
in Drain's pigment path**. Mixing comes from resolved advection and the finite
volume scheme's numerical diffusion. Classic and Rapids retain their separate
depth-weighted advection–diffusion transport and reduced impact wakes, detailed
in [DYE.md](DYE.md). All modes keep pigment passive: it does not change density,
viscosity, surface tension or water momentum.

## Impacts, mode transitions and rendering

Rain supplies a bounded, outward Gaussian momentum impulse to Drain and deposits
pigment with the selected footprint. The impulse changes horizontal momentum,
not water depth or volume. The diameter control is a spatial impact multiplier,
not a calibrated falling-drop diameter; [DYE.md](DYE.md) gives source scaling.

Water, rain and pigment share the application's physical clock. Drain consumes
4 ms coupling ticks directly; the other modes retain 8 ms coupling. Equivalent display-frame partitions give
the same state for the same timed inputs; interactive rain is still scheduled
at display updates, so different input timestamps need not give identical runs.
Pause advances no physical state, and resizing only resamples the grids.

Mode changes retain the pigment field and visually blend water surfaces over
0.45 physical seconds. Requested Drain immediately owns pigment transport;
leaving it immediately stops its hydraulic advancement and suction, while its
last surface fades from view. Its saved state can be resumed later. This is a
visual transition between models, not a physical hybrid-fluid boundary problem.

Rendering samples the evolved depth and momentum; sampled velocity is momentum
divided by depth. The same depth feeds optical attenuation at primary and
secondary water encounters. Surface normals, reflection and refraction follow
the evolved height field plus the small-wave optical closure below. Drain contributes no synthetic foam: this wet-core
model does not resolve air entrainment that would justify whitewater.
Pigment changes underwater transmission and distributed volume illumination,
not the dielectric surface's Fresnel response; see [OPTICS.md](OPTICS.md).

### Rain-driven optical detail

A separate 96×64 signed height/vertical-velocity field restores short-wave
surface structure lost by the coarse shallow-water solver. It follows the sampled
hydraulic current and depth on fixed 8 ms ticks. Its characteristic wavelength
is 0.18 m, with finite-depth phase coefficient
`c² = (g/k + surface_tension*k/density) * tanh(k*h)`.
This is a narrow-band, small-amplitude optical approximation, not a fully
dispersive or fully coupled capillary solver; this wavelength is gravity dominated.

Rain supplies a compensated zero-mean pressure footprint. Unresolved tiny
footprints are filtered with matching low-frequency impulse scaling. Cubic
advection retains resolved wave crests; a shared-face `div(c² grad eta)` operator
and velocity-Verlet advance the waves. Physical viscous damping, zero-mean
projection and a common depth-dependent amplitude bound keep this optical field
bounded. The amplitude limit dissipates excess detail; it is not a hydraulic
energy conservation claim. Viewport footprint filtering removes unresolved detail.

The optical layer never changes hydraulic depth/momentum, dye transport or
water/pigment ledgers. Reflections, refractions, normals and caustic structure see
the added geometry; absorption continues using the actual hydraulic depth.

## Validation and limits

Source tests exercise radial pressure initialization, a finite wet core,
matched feed/suction volume accounting, inward current, lack of invented spin
without angular forcing, momentum-only impacts and physical-time partitioning.
Coupled scalar tests cover uniform-concentration consistency, clear-feed
dilution, suction and bleaching ledgers, rejected-stage isolation, positive
filaments and advection at the actual water-volume velocity. Manufactured-stage
scalar checks are separate from whole-model diagnostics.
These are targeted numerical checks, not laboratory validation of a drain or
a turbulent flow.

Run the contracts in [drain.rs](../src/drain.rs) and
[dye.rs](../src/dye.rs) from the repository root:

```sh
cargo check --locked -p acid-rain --all-targets
cargo test --locked --release -p acid-rain drain::tests
cargo test --locked --release -p acid-rain dye::tests::hydraulic
```

Historical 300-physical-second runs, stress captures and generated reports are
not distributed. Their frozen-source measurements are not current acceptance
limits. Any new long-run claim needs its own source revision, forcing history,
water/pigment ledgers and saved depth/velocity diagnostics; passing focused tests
alone does not establish stability for every input or rendered performance.

The depth-averaged hydrostatic approximation loses accuracy when vertical
motion and short-wave dispersion matter. It does not resolve a three-dimensional
air core, pipe flow, pipe turbulence, a vertical suction jet, overturning crests,
spray, individual bubbles, free-surface breakup or contact/wetting. The wet-depth
guard is a numerical model limit, not an air–water interface. The domain has
periodic edges rather than tank walls. Incoming pump forcing, intake and outer
reservoir resistance, and quadratic drag are explicit closures; numerical dissipation is not a calibrated turbulence
cascade. The initial pressure-balance construction does not make the discrete
flow an exact steady-state or well-balanced vortex solution.

The general HLL shallow-water construction is described by
[Clawpack's approximate Riemann solver reference](https://www.clawpack.org/riemann_book/html/Shallow_water_approximate.html).
The shared concentration flux and source accounting above specify this
implementation's coupling. [PHYSICS.md](PHYSICS.md) compares its scope with
the other water models.
