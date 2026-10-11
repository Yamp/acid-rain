# Color and optical processing

The water geometry, dispersion, refractive indices and physical clocks remain described
in [PHYSICS.md](PHYSICS.md). The display is an SDR true-color terminal; it cannot represent
spectral radiance, HDR luminance or a calibrated cinema projection.

Intensity defaults to 4 and accepts 0–4. `+` / `=` increases
it and `-` decreases it; the status label and `h` / `?` help show the current value.
The extra range strengthens artistic illumination, interference and layered
perception. Prescribed illumination and display chroma are artistic controls,
not stronger pigment extinction or extra dye mass. Water parameters, scene and
physical clocks, camera roll/elevation and field-of-view motion do not depend on
this control. A separate `--lsd` setting controls subjective perception from 0 to 2,
default 1: `n` / `m` adjust it by 0.1 and `N` restores 1. It scales the histories,
retinal diffraction approximation, whole-image RGB drift and spatial chroma
adaptation below; it does not change water dynamics or optical pigment mass.
`e` toggles those subjective stages together. Channel displacement also has its
own `--rgb-shift` setting, from 0 to 12 display pixels with a default of 2;
`j` / `l` adjust it by 0.25 and `J` restores 2. This setting is a strict ceiling
on each channel's displacement vector. The instantaneous radius varies smoothly
with subjective dose and never exceeds the ceiling. Zero Trip or zero LSD bypasses
all subjective stages; zero RGB radius bypasses only direct channel displacement.

## Linear-light color pipeline

The working RGB space uses Rec.709 primaries and D65, with luminance weights
`[0.2126, 0.7152, 0.0722]`. Environment illumination mixes equal-luminance pastel sources,
so chromatic changes do not introduce exposure pulses. All defaults use neutral gains.
The grade normalizes gain luminance and removes the hue matrix's luminance error.
Hue/saturation excursions below zero contract toward the same-luminance neutral
until all channels are nonnegative. The maximum of those corrected RGB channels
then sets the filmic shoulder; a single multiplier maps all three channels into
the SDR gamut. Their ratios survive this shoulder instead of being compressed
toward white to retain a near-white output luminance. Neutral ramps retain the
previous scalar curve. This deliberately sacrifices the former high-luminance
matching for saturated colours: SDR cannot preserve both their chromaticity and
the luminance of equally bright neutral light. The lower-gamut contraction can
change chromaticity; ratio preservation applies after that correction, not to
arbitrary hue/saturation excursions. It is not a perceptually uniform colour-space
transform. Whole-image RGB reassembly and the separate subjective chroma
adaptation now operate on linear HDR before this common tone/gamut transform.
The recombined channels pass through the display transform once, then the sRGB
transfer once; channels are not separately tone-mapped before being mixed.

The final sRGB transfer uses its linear toe and 2.4 exponent, cached in 8193 samples
and interpolated. Its tested absolute error is below `1e-5` over 20,001 samples, with
all 256 encoded terminal values round-tripping. Displayed scene crossfades decode
bytes to linear light before blending and encoding again; a black/white midpoint
is sRGB 188 rather than 128. Terminal/OS display calibration remains external.

This is a **custom** SDR transform. It is not ACES, an ICC monitor profile, an exact
CIE spectral integration or an implementation of the ACES reference gamut compressor.
The separation of tone mapping and gamut handling is informed by the
[ACES gamut-compression overview](https://docs.acescentral.com/rgc/overview/) and
[output gamut-mapping description](https://docs.acescentral.com/system-components/output-transforms/technical-details/gamut-compression/).

## Optical paths and spectral effects

Water keeps its existing RGB indices `[1.331, 1.333, 1.337]`; reflection and transmission
use the same channel indices and dielectric Fresnel. Reflected rays intersect the displaced
surface, including rising and horizontal rays that meet a steeper nearby crest.
High allows three secondary encounters, Balanced two, Low one. Each encounter splits
Fresnel reflection and transmitted/absorbed light, propagating RGB throughput. A global
surface-slope bound skips rays proven to escape. Small throughput terminates early;
the unresolved tail always samples the cached environment instead of disappearing.
Escaping paths retain their environment contribution; transmitted paths meet the
illuminated/emissive bed and absorbing column described below.

This is a bounded deterministic height-field approximation, not a Monte Carlo path tracer:
finite grazing scans can miss very narrow occluders, unresolved reflection tails use the
environment, and transmission uses a local diffuse bed plus homogeneous volume-source
approximations.
Microfacet shading remains at the primary encounter; secondary paths use geometric specular
directions. See [PBRT dielectric reflection and transmission](https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission)
for the transport principles.

### Refracted bed and diffuse airlight

Transmission sees a structured bed with neutral diffuse albedo, authored coloured
illumination and sparse emissive veins. It no longer looks up a reflected/flipped
sky direction behind the water. Surface reflections still retain directional sky detail.
Snell refraction projects the view ray to a locally horizontal bed using actual column
depth in metres. The deterministic procedural bed now continues outside the finite
numerical water patch. A fixed 1.5 m halo on each side supports refracted lookups.
For exceptional grazing rays whose lateral travel exceeds that support, the lookup
endpoint is scaled back along its horizontal direction to the halo; the Beer path
length remains the original refracted length. This is a bounded local-plane
approximation, not arbitrary bathymetric ray tracing.

The neutral mineral albedo stays between 0.10 and 0.90. Four smooth noise scales,
50, 25, 12 and 6 cm, have weights `[0.52, 0.27, 0.14, 0.07]` after a 19 cm
warp coefficient. Seeded cellular mineral plates and broken spiral fossils vary
the response across larger regions. Spiral support fades before its generating
cell boundary; the underlying mineral field continues smoothly through it.
The texture approximates surface material, not resolved mineral geometry.

Coloured emissive veins follow warped 56 and 28 cm level sets with a smooth band
and spatial gate, mixed with sparse spiral markings. Emission remains bounded by
0.32 per channel in working radiance units, with the accepted lilac/rose/cyan
spectra and no neutral emissive floor. A positive normalized two-dimensional
`[1, 4, 6, 4, 1] / 16` binomial filter smooths only emission, approximately 3.7 cm
FWHM. Its halo is evaluated from the same seeded world field as its interior.

The material uses 64 samples per metre: a 336 × 336 filtered rolling lattice
and a 340 × 340 raw lattice occupy 3,655,936 bytes (3.49 MiB), independent of
travel distance. World coordinates use `f64` and integer cell addresses; per-ray
sampling uses a precomputed local `f32` offset and four bilinear cache reads.
Crossing a lattice boundary generates only entering rows and columns. Fractional
motion within a cell does not regenerate material; a teleport or seed change
regenerates at most the finite cache. Ring addressing is storage reuse, not
periodic terrain: revisiting a world position regenerates the same material.
Each RGB Snell ray samples that world material at its own refracted position.

A separate 64 × 64 world-aligned lattice at 12 samples per metre filters
environmental irradiance, including the refracted halo. A smooth seeded palette
field interpolates cyan, blue, lilac, rose, amber and mint. Its slow phase drift
moves prescribed illumination over the bed without attaching it to the camera.
With `s = clamp(2 × light_complexity, 0, 1)` (equivalently `min(D / 2, 1)`),

```text
F_c = 1 + s × [0.12 + 0.88 × world_palette_c - 1]
```

At full strength each channel lies between 0.12 and 1; the 0.12 floor is
minimum transmission, not added white radiance. There is no luminance
renormalization after filtering. The filter affects diffuse environmental bed
illumination and the environmental source used for single scattering; the white
directional light is evaluated separately. This is an authored spatial irradiance
boundary, not traced projectors or a solved volumetric lighting field. Refraction
selects visible positions while the world field stays independent of the camera.

The environment cache integrates 32 upper-hemisphere directions with cosine weighting
and entry Fresnel to estimate diffuse bed irradiance divided by pi. It also stores a
radiance-weighted mean refracted secant for the diffuse illumination path. The direct
light uses its refracted path and entry Fresnel separately. Reflected illumination
receives clear-water extinction (absorption plus scattering) and pigment Beer
attenuation on both legs. Emission originates
at the bed, so only its view leg is attenuated. In schematic channel notation,

```text
diffuse return = bed_albedo × filtered_incident_irradiance / pi × T_light × T_view
emitted return = authored_bed_emission × T_view
bed return = diffuse return + emitted return
visible transmission = (1 - F_view) × bed return
T_light × T_view = exp(-tau_light - tau_view)
```

The view leg samples actual transported pigment near its midpoint. The direct light
leg samples its own midpoint; diffuse illumination uses the local bed sample and its
cached mean path. The two integrated optical masses enter the same exponential rather
than unrelated colour filters. The existing normalized paraxial caustic grid modulates
direct bed irradiance, with channel-dependent Snell projections selecting its samples.
It no longer boosts the volume-source terms. The diffuse bed is passive and returns
zero without incident light. The emissive veins are a distinct, explicitly authored
source; they remain visible without illumination, but still pass through clear-water
and actual pigment absorption plus exit Fresnel. Their intensity follows the same
bounded strength `s` as the illumination filter and vanishes at zero intensity.
No fluorescent excitation, conversion efficiency or spectral re-emission is solved.
Neither source changes the passive pigment extinction or transported pigment mass.

This is a local horizontal-bed/column approximation, not ray tracing through arbitrary
bathymetry. Diffuse illumination uses one mean path, pigment is depth averaged, and
neither path integrates a resolved three-dimensional concentration field. Caustics remain
paraxial and grid normalized; this is not a volumetric multiple-scattering solver or
complete underwater light transport.

Atmospheric fog keeps its distance-dependent exponential blend but takes its airlight
colour from a cached upper-hemisphere mean. This isotropic in-scattering approximation
has no view-azimuth dependence at a vertical ray. Flattening a nearly vertical view
direction onto the horizon previously unfolded directional sky colour into radial
spokes; diffuse airlight removes that coordinate singularity without changing the
water's physical dye field. Scene fog density is now 0.00055–0.001 per metre,
100 times lower than the rejected settings. It is not a full atmospheric
phase-function integration.

The primary GGX lobe uses an effective unresolved roughness alpha of 0.05.
Resolved normal variance and the pixel footprint broaden it before shading,
preserving filtered highlights as the camera moves. This is a microfacet closure,
not a measured universal water constant. Compared with the previous 0.12, it
concentrates the same normalized lobe into more localized glints without increasing
emitter radiance; the change is visible only where light and view directions align.

The primary camera uses a roll- and orbit-independent safety cone, rather than
refitting rotated image corners against each small change in wave height.
Its nominal surface envelope is `[-0.16, 0.12]` metres. For patch width `L`, the
cone fits inside a circle of radius
`0.47 L - max(0.04 m, distance(target, patch_center))`, with a positive numerical
floor. The 4 cm reserve accommodates normal patch rebases and presentation lag.
The two major-axis endpoints of the cone's intersection with each limiting
height plane bound the view for every roll. A smooth minimum limits the requested
half-angle while preserving horizontal/vertical aspect ratio. Ordinary rain,
roll and heading changes therefore do not repeatedly pump the lens.

`CameraPresentation` critically damps the world-space target, elevation, FOV and
oscillatory part of roll, using a 0.06-second damping scale and active wall-time
steps capped at 0.1 seconds. World targets use absolute `f64` coordinates before
returning to local ray coordinates, so moving the numerical patch is not treated
as camera motion. The steady 4-degree-per-scene-second roll and orbit heading
remain tied to the shared scene clock; terminal rendering does not predict an
independently published simulation snapshot. Pause and help freeze presentation
motion. Switching coordinate modes or moving the scene clock backward reseeds
the presentation state.

Exceptional heights expand the envelope immediately, with 2 mm clearance, and
release it toward the nominal range with a two-second exponential time constant.
If necessary, the safety fit also raises the orbit elevation or distance so its
central ray remains valid. These exceptional corrections can change framing;
there is no claim of a fixed lens under arbitrary wave heights or window shapes.
The construction keeps primary rays in the finite simulated patch without tiling
water or changing secondary reflection/refraction approximations.

## Dissolved pigment and distributed light

[pigment_optics.rs](../src/pigment_optics.rs) treats pigment as passive spectral
extinction. The cyan, magenta and amber extinction bands retain their relative
spectral shapes, scaled by 0.85 from the previous version for slightly clearer
color. These are artistic broad bands in arbitrary pigment-mass units, not
measured optical constants for a particular dye.

Water absorption and scattering now have separate roles. For RGB channel `c`,

```text
sigma_a,c = [1.2, 1.6, 1.2]_c × scene_absorption_scale  m^-1
sigma_s = 0.015  m^-1
sigma_t,c = sigma_a,c + sigma_s
tau_c = sigma_t,c × view_path_length
T_c = exp(-tau_c)
W_c = sigma_s / sigma_t,c × (1 - T_c)
L_volume,c = (1 - F_view,c) × W_c × J_c × pigment_volume_factor_c
```

These are an authored rose-tinted absorption spectrum and a chosen clear-water
scattering coefficient, not calibrated natural-water data. Absorption removes
light; it cannot itself illuminate an unlit column. The former constant ambient,
diffuse/subsurface body fill and blanket fluorescence are removed. Explicit
emissive bed veins remain a separate source with their own outgoing path.

The isotropic incident source `J` uses actual environment/direct light. A cached
32-direction upper-hemisphere quadrature sums
`L_sky,c × (1-F_entry,c) × cos(theta_i) / [2 N cos(theta_t)]`, including the
combined Snell solid-angle/radiance Jacobian; its RGB is then multiplied by the
prescribed spatial illumination filter. The separate direct source is
`0.18 × (1-F_entry,c) / (4 pi)` when incident from above. Neither term is a fixed
material colour that stays lit after the incident sources are extinguished.

This is a constant-source, homogeneous-column, isotropic single-scattering
approximation. The view-path integral is exact for that assumed source; incoming
light extinction at every point inside the volume is not separately resolved.
It is not a full bidirectional volume-lighting or multiple-scattering solution.
Primary and secondary paths share the same extinction and scattering treatment.

Light reflected from the bed receives full extinction on its incoming and outgoing
paths; the bed's separate emission receives only the outgoing path. The scattering
coefficient appears exactly once in each traversed leg. Pigment remains absorbing
and changes the homogeneous volume return through

```text
E(tau) = (1 - exp(-tau)) / tau, E(0) = 1
volume_factor = E(tau_water + tau_dye) / E(tau_water)
```

Here `tau_water` includes water absorption and scattering. This is the homogeneous
distributed-source integral relative to clear water with the same incident light.
Primary and secondary volume paths share this correction. Surface reflection and
direct specular light remain outside the pigment column; thin foam retains
its full-path pigment attenuation.

Both factors are passive and bounded by one; the volume factor lies between
full-column pigment transmission and one. Clear pigment is exactly transparent,
and zero source radiance stays zero. These pigment factors add no brightness floor;
the diffuse bed is separately illuminated and its authored emission is accounted for
as a distinct source. A thin-depth series
avoids cancellation, and existing water transmission is reused by the volume path.
This corrects
attenuation within the existing homogeneous source approximation; it does not
turn the renderer into a multiple-scattering volume solver. See the
[PBRT equation of transfer](https://pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer).

Bleaching and physical removal through the drain remain separate mass processes,
documented in [DYE.md](DYE.md) and [DRAIN.md](DRAIN.md). They do not depend on
exposure or the optical coefficients.

## Thin-film interference

A cached air / dielectric-film / water Airy response adds a gentle colored coating.
Nine wavelengths per broad RGB band are averaged with normalized sensitivity weights;
2049 optical-path samples replace per-pixel spectral trigonometry. View angle changes the
film's optical path. Film thickness and strength are artistic, and the response uses
normal-incidence interface amplitudes with angular path length rather than a full
polarized oblique multilayer solver. Ordinary clean water does not acquire this coating.
The pastel emitter is a spatial color design, not a measured spectral light source.

## Finite self-similar illumination

The cached environment adds three smoothly folded directional light structures.
Each octave applies a fixed rotation, a smooth absolute value
`sqrt(x² + 0.15²)` in two axes and a spatial scale of 1.55, then evaluates broad
light contours with three sinusoidal terms. Coordinates are Cartesian, avoiding
an azimuth seam. Three octaves are always the limit; increasing intensity does
not recurse further or increase the reflected-ray budget.

Let `d = intensity / 4`. Base octave weights `[0.56, 0.29, 0.15]` are multiplied
by `1 + 0.20 j d` for octave index `j`, then normalized by their sum. The folded
field supplies a patterned attenuation mask for the world's existing emitter:

```text
structure = luminance(folded_RGB) / sky_energy
mask = smoothstep(0.025, 0.26, structure)
gain = 1 - 0.84 sqrt(d) (1 - mask)
output_RGB = original_world_RGB × gain
```

At intensity 4, gain lies between 0.16 and 1; at zero intensity the stage is
bypassed exactly. Multiplying all channels by the same bounded scalar preserves
the original chromaticity and does not increase source radiance or fill dark
pockets with a second bright emitter. The stage changes the illumination's
spatial distribution and integrated power; it is an artistic patterned light
source, not a passive-water transport law or a claim of equal total power.

The 64×64 octahedral environment cache evaluates this structure once per cached
direction, rather than at each water hit. To soften unresolved contours, width
is `0.30 + 0.5 × 0.08 × omega`, where a conservative directional derivative bound
starts at `omega = 10` and grows by 1.55 per octave. The 0.08-radian footprint
regularizes the emitter at the map's scale. This is analytic smoothing with a
finite octave budget, not strict texture band-limiting or full ray differentials.
The existing spectral emitter and SDR gamut handling still constrain the palette.

Surface reflections reveal these repeated directional contours. Diffuse bed illumination
uses their hemispheric average and its spatial spectral filter rather than a transmitted
sky image; refraction reveals the textured/emissive bed, local pigment and caustic
variation. The contours are an artistic light source,
not fractal geometry added to the hydraulic model, and they do not change dispersion,
flow velocities or the camera's timing.

## Glare and ocular echoes

Bloom extracts all channels with one luminance gate, retaining their ratios.
It removes the scattered fraction from the source and redistributes it through normalized
half-resolution Gaussian/glare kernels. Diffraction-shaped horizontal, vertical and diagonal
taps scale by the RGB representative wavelengths, red being wider than blue.
Constant colored fields remain unchanged even at frame edges. Edge clamping and radial shafts
do not establish exact global photon-flux conservation or a full aperture diffraction solver.

The pupil's bounded exposure range is 0.4–6.4; extra dark-scene headroom replaces
the prior 2.5 upper limit. The 2.5 reference used for the pre-existing aperture-size
effect is retained, so exposure headroom does not enlarge that effect by itself.
Exposure remains display adaptation, not a water light source.

The subjective layer uses three independent histories of pupil-exposed HDR light.
The current scene meters the pupil, that exposure multiplies incoming linear
light, and only then do the histories accumulate it. The final grade applies no
second pupil multiplier. Opening the pupil in a dark view therefore cannot amplify
a remembered bright frame. Creative scene exposure remains part of the grade.
Each reservoir follows the unwarped current image with

```text
H_new = H_old + (I - H_old) × alpha
alpha = 1 - exp(-ln(2) × dt / half_life)
```

The half-lives are fixed at 0.10, 0.35 and 0.80 active wall seconds. All three RGB
channels use the same half-life within a reservoir: neutral history does not
change hue as it decays. Dose changes displayed weights, not stored age.
Exponential integration is exact for a constant input and independent of its
frame partition. Spatially transformed output never feeds back into these
reservoirs or temporal antialiasing.

The subjective clock `t` advances with accepted active wall time, independently
of simulation speed. With Trip `D` and subjective setting `S`, its smooth dose is

```text
envelope(t) = 0.72 + 0.17 sin(0.19 t + 0.7) + 0.11 sin(0.071 t - 1.2)
dose = S × clamp(D / 4, 0, 2) × envelope(t)
```

The envelope stays in `[0.44, 1]`; it has no random per-frame flashing. A zero-time
redraw advances neither the effect clock nor the reservoirs. Pause and help
supply zero elapsed wall time. A renderer gap longer than 0.25 seconds discards
retinal history and does not advance the effect phase. Scene changes and terminal
resize also invalidate history without resetting that phase. An internal quality
resize can resample the valid reservoirs while preserving their ages.

Displayed histories have independent small translations and spatial bends in
display-pixel units. They do not rotate or zoom the whole image. Their amplitude
is `(0.5 + 0.45 × rgb_shift) × min(dose, 2)` pixels before each layer's bounded
coefficients; this history warp is distinct from the direct RGB-radius ceiling.
Each layer uses three bilinear channel samples. Frame, row and column
oscillators are cached, so the pixel loop does not evaluate trigonometric waves.

Layer weights start at `[0.10, 0.105, 0.075] × min(dose, 1.7)`. Each is multiplied
by a nonnegative color-affinity factor no greater than one. Their sum is at most
0.476, leaving at least 52.4% current light before subsequent effects. Affinity
reduces incompatible chromaticities and separately rejects bright neutral history
that would bleach visible saturated current colors. Its protection fades near
black, preserving colored afterimages over a dark current view. This is a deliberate
artistic color-retention rule, not additional pigment absorption or a physiological
model. Exponential updates, bilinear samples and history mixing remain convex;
constant fields stay unchanged and this history stage introduces no additive HDR
gain. Spatial warps and clamped history boundaries do not conserve integrated light.

`e`, zero Trip and zero LSD bypass the subjective stages. Disabling persistence
invalidates its reservoirs, so re-enabling cannot reveal stale light. The separate
scene bloom/glare, thin-film response and bed illumination retain their own controls.
These distinctions matter: disabling LSD does not remove the scene's authored
psychedelic lighting or change its hydraulic simulation.

After persistence, a sparse retinal point-spread function operates on the exposed
HDR image. It has a central sample and four symmetric lobes on the horizontal and
vertical axes. Representative RGB wavelengths `[650, 550, 460]` nm set their radii
to `6 × wavelength / 550` display pixels. Their total weight is
`0.13 × min(dose, 1.7)`, with the remainder at the centre. Positive normalized
weights and reflected pixel-cell boundaries preserve constant fields and each
channel's finite-image sum to floating-point tolerance, including corners.
This is a bounded diffraction-inspired approximation, not coherent wave propagation,
an exact Airy pattern or calibrated ocular optics. It naturally spreads bright
glints more visibly because they carry more light; it adds no white source.

At full output resolution, the image is reconstructed from three independently
moving RGB coordinates **before tone/gamut mapping**. Every pixel is eligible,
including dim regions and broad shapes. For channel index `c`,

```text
phase_c = 2 pi c / 3
angle_c = (0.13 + 0.009 c) × t + phase_c
radius_c = rgb_shift × [0.675 + 0.325 sin(0.12 t + phase_c)]
           × smoothstep(0, 1, dose)
output_c(x, y) = bilinear(input_c, x + radius_c cos(angle_c), y + radius_c sin(angle_c))
```

The 2-pixel default is a strict maximum for each channel, not a nominal radius
that can grow to 2.375 pixels. Pairwise channel separation may be larger than one
channel's radius. The setting still clamps to 0–12. Zero radius is an exact bypass
of this displacement while histories, diffraction and chroma adaptation remain
available. Bilinear sampling stays within each source channel's range; clamped
boundaries do not establish integrated-light conservation. The pass uses full
output resolution even in Low, needs no temporal warmup and never feeds back into
reconstruction or HDR reservoirs. This subjective screen-space separation is
independent of the water's physical dispersion.

A separate contrast-driven chroma pattern then rotates only the component
perpendicular to Rec.709 luminance. A bounded chroma-strength adjustment and
same-luminance contraction keep channels nonnegative. Uniform colors and neutral
fields remain unchanged; nonuniform colored regions change smoothly with the
subjective clock. The row/column field is cached and the pixel loop uses a Cayley
rotation without per-pixel trigonometry or allocation. This preserves luminance
at this stage, not each channel's integral or a perceptually uniform color difference.
It is an artistic adaptation pattern rather than a solved retinal response.

The complete perception order is pupil exposure → unwarped HDR reservoirs and
displayed history mixture → normalized retinal PSF → output-grid reconstruction
and vignette → RGB reassembly → subjective chroma adaptation → common tone/gamut
mapping → sRGB. Existing bloom/glare precedes this sequence; its approximate
conservation limits are separate from the normalized retinal PSF. Neither the
complete subjective pipeline nor final tone mapping is claimed to conserve a
scene's physical radiant flux.

`i` controls a separate presentation stage. A bounded worker stores completed,
timestamped display frames and linearly interpolates their decoded sRGB values
at the requested display cadence. It adds roughly two source-frame intervals of
latency (33–150 ms) to provide intermediate frames. This is an image crossfade,
not new hydrodynamic states or optical-flow reconstruction. Its samples never
feed back into the renderer or into the RGB echo histories. Pause, help, resize
and mode changes of the presenter reset its history. Displayed frame rate and
the rate of computing complete scene frames are therefore different measurements.

The terminal contracts in [retina.rs](../src/retina.rs),
[optics.rs](../src/optics.rs) and [color_pipeline.rs](../src/color_pipeline.rs)
cover radius bounds, neutral-history rejection, achromatic half-lives,
frame-partition independence, chroma luminance preservation and PSF energy bounds.
Run them from the repository root:

```sh
cargo check --locked -p acid-rain --all-targets
cargo test --locked --release -p acid-rain retina::tests
cargo test --locked --release -p acid-rain optics::tests
cargo test --locked --release -p acid-rain color_pipeline::tests
```

The [native GPU checks](../native/Tests/README.md) exercise its separate shader
implementation. Historical image comparisons and generated benchmark reports are
not distributed. Neither those older captures nor numerical contracts alone
establish current image quality, displayed frame rate or frame pacing.
