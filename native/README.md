# Acid Rain · Metal

A native macOS window for the same water, dissolved pigments and travelling procedural world.
The Rust simulation is shared directly with the terminal edition. Metal renders continuous
pixels: the terminal framebuffer and character cells are not involved.

## Launch

```sh
./native/run.sh
```

Local application: **`native/build/Acid Rain.app`** (macOS 14+).
The Apple Silicon [release archive](https://github.com/Yamp/acid-rain/releases/latest)
also includes this application. Use `./native/run.sh --fullscreen` to open in fullscreen.

The build script first checks the separate Rust core, builds its static library, compiles
the Swift application and creates a local `.app` in `native/build/`. The app can subsequently
be opened directly in Finder without a terminal. Building requires Rust 1.88+ and Apple's
Swift/macOS SDK tools. No graphics framework needs to be downloaded.

If the optional offline Metal compiler is available, shaders are packaged as a `.metallib`.
Otherwise the application compiles the bundled Metal source using the system Metal runtime
on startup. Both paths render on the GPU.

## Controls

`h` opens the full help overlay and holds the animation. The compact status display appears
at startup and after a control change, then fades out of the image.

| Keys | Action |
|---|---|
| Space | Pause / resume |
| `⌘ F` | Fullscreen |
| `[` / `]`, Left / Right | Speed; `s` restores 2.4× |
| Up / Down | Rain intensity |
| `,` / `.` | Drop diameter; `R` restores rain and diameter |
| `o` / click | One drop at the center / clicked water surface |
| `a` | Show / hide airborne drops; hidden by default, impacts and pigment stay active |
| `x` | Clear pigment and pending drops, preserving water waves |
| `p` | Toggle focused drop observation lighting |
| `r` | Burst of coloured drops |
| `v` / `b`, `B` | Bleaching rate down / up / reset |
| `-` / `+` | Psychedelic intensity |
| `j` / `l`, `J` | RGB displacement down / up / reset to 2 logical pixels |
| `n` / `m`, `N` | Subjective effects down / up / reset to 1; range 0–2 |
| `e` | RGB separation and retinal effects |
| `i` | Temporal frame blending |
| `g` / Tab, `t` | Next scene / automatic Journey |
| `d`, `f` | Drain / river rapids; press again for Classic |
| `c`, `w` | Camera travel / independent current |
| `k`, `0`–`4` | Cycle quality / Auto, High, Balanced, Low, Native |
| `q`, Escape, `⌘ Q` | Quit; Escape first closes help |

Defaults: Classic, Journey, High, speed 2.4×, Trip 4, rain 32 per physical second,
drop diameter 1.5×, bleaching 1×, RGB shift 2 logical pixels, travel/current/blending on,
subjective effects 1, airborne drops hidden. `--lsd 0` disables the subjective layer;
`--lsd 2` increases it. `--show-drops` enables falling drops at launch; `--hide-drops` hides them.
Drain and Rapids retain their local domains; world travel resumes in Classic.

## Rendering and performance

The bounded 192 × 128 surface and depth-averaged flow run on a serial background
queue. Native pigment is transported separately in a three-dimensional Metal volume;
the terminal keeps its existing depth-averaged pigment solver. The display consumes the latest immutable snapshot. Three sets of upload textures
keep the CPU from overwriting resources still in use by the GPU. The world retains a local
material window as the camera moves.

The Metal path includes surface intersections, filtered normals and specular highlights,
Fresnel reflection, bounded secondary reflections, wavelength-dependent refraction,
pigment and water Beer attenuation, coloured bottom illumination, emissive veins and foam.
A GPU luminance reduction drives gradual pupil adaptation. The actual R/G/B channels
drift independently within the configured logical-pixel radius. Three unwarped linear-HDR
reservoirs capture rising light with 0.025, 0.050 and 0.090 second half-lives, then release
it with 0.24, 0.72 and 1.65 second half-lives. A shared RGB response preserves each
reservoir's neutral colours; bounded history weights total at most 54.4%. One fused GPU
pass updates all three from the same exact pixel-area average of the current HDR image,
so small highlights contribute even when history resolution is much lower. Their
contribution, drift and spatial colour rotation vary continuously on an active display
clock, independent of simulation speed. A positive, normalized spectral diffraction
filter redistributes light with a wider red footprint. Colour rotation preserves luminance and leaves neutral
colours neutral; hue agreement limits muddy mixing of unrelated history.
HDR colour is tone-mapped once after recombining the channels, with a common channel
scale to protect saturation. Pause, help and occlusion exclude inactive wall time;
slow active frames still age the reservoirs instead of erasing them.
The final sRGB attachment applies the display transfer exactly once.

Quality controls change internal GPU resolution while the final drawable follows Retina
and window resizing. High has a 1.6-million-pixel budget, Balanced 0.9 million and Low 0.45
million. Native renders every drawable pixel. Auto changes its budget gradually using GPU
timings. RGB displacement remains measured in logical display pixels at every quality.

This is a new GPU implementation of the optics, rather than a pixel-identical port of the
terminal renderer. It uses the same depth-averaged/local-surface physics and material world;
it is not a full 3D Navier–Stokes or volumetric path tracer. Secondary reflections and the
bottom/volume light integrals are deliberately bounded approximations. Retinal history
uses six ping-pong textures, sized from half the drawable resolution and capped at
350,000 pixels each (16.8 MB total). Quality changes preserve their dimensions and age;
a drawable resize or explicit scene discontinuity resets them. Subjective transformations
never feed back into this history. The existing `i` control
toggles the separate one-frame blend. The subjective layer is an artistic perception
effect, not a validated biological model of LSD or a change to the fluid simulation.
Its bottom caustics use a separate bounded GPU forward-flux map; they are driven by
the actual water surface and wavelength-dependent refraction.

Camera projection reserves a stable height envelope and covers every orbit/roll angle.
Wave extrema and the streamed world's cell rebasing therefore do not change its normal
zoom. Exceptional out-of-envelope heights expand the safety fit, which relaxes gradually.
Presentation follows the physics sampling timestamp rather than its later publication;
the camera clock remains monotonic and smoothly corrects phase errors from CPU jitter.

## Verification

Validation commands and image diagnostics live in [Tests](Tests/README.md).
Generated timings and captures are stored locally under `artifacts/` and are not part
of the source archive. Offscreen GPU timings and window presentation cadence are
different measurements; the test guide includes the commands to reproduce them.

The static library and terminal executable are separate packages in the shared Cargo
workspace. They use the root lockfile and target directory.

## Individual drops and three-dimensional clouds

The native renderer can draw falling colored drops before impact (`a` / `--show-drops`).
They are hidden by default. Each impact still creates
a bounded three-dimensional pigment volume transported by the water's mean flow and
localized vortex-ring wakes. Click on the surface or press `o`; `x` clears pigment to
watch another isolated drop. The existing diameter, rain and bleaching controls apply.

A deterministic single-drop view sets drop diameter to 3× (including subsequent `o`
drops) and disables background rain, camera travel and uniform
current, while retaining camera rotation, waves and the chosen optical settings.
It also enables observation lighting: a quieter neutral bottom and broadband light
make the pigment cloud easier to distinguish. Press `p` to return to the normal scene;
`--cloud-study` enables this lighting without disabling rain or travel:

```sh
./native/run.sh --drop-demo --show-drops
./native/run.sh --drop-demo --frames 360 --capture-series native/artifacts/drop-series \
  --capture-every 30 --benchmark native/artifacts/drop-series/timings.json
```

Series files contain actual rendered frames from one continuous simulation; `frames.json`
records each image's physical time, scene time and world origin. Volume
updates follow physical time and remain frozen when paused or when help is open.
The 96×64×32 domain uses two pigment buffers and one velocity texture (9 MiB total),
independent of window dimensions; at most 24 recent impact wakes drive the flow.
Transport is conservative; vortex rings and coherent strain are prescribed velocity
models, not a fully resolved three-dimensional Navier–Stokes simulation. Filaments
smaller than a voxel remain unresolved. Changing water mode clears the cloud domain.
After a severe display stall, transport catches up at most 0.128 physical seconds
per displayed snapshot to keep GPU work bounded; excess stalled time is skipped.

## Contact waves and bottom light

Native drops transfer their metric sphere momentum into a finite-depth gravity-capillary
pressure-impulse solution. Contact uses the actual sub-frame collision time and velocity
relative to the moving surface. The same displacement changes height and column depth,
so the bed stays fixed and the new normals drive reflection, refraction and caustics.
The terminal's existing impulse path is preserved. This is a linear wave closure, not
a resolved nonlinear splash/crown calculation; the empirical contact footprint and the
energy budget are covered by the native impact and wave-energy tests.

Caustics forward-project a 128² incident grid in three spectral channels, then add
normalized finite footprints into a 256² irradiance map. Entry Fresnel, Snell refraction,
actual local bathymetry, source angular width and receiver filtering all participate.
The camera and the light share the same bounded bed intersection. Underwater radiance
uses the refractive-index-squared factor when exiting into air. Ambient illumination
remains separate from the focused directional beam; the incoming pigment path uses
the existing local-column approximation rather than a full volumetric light trace.

The bottom also has authored world-anchored mineral relief, luminous veins, slowly
moving spiral organisms and crawling ribbons. Their bounded reflectance and emission
are separate from physical light transport. Fine details fade with the refracted
pixel footprint. Observation mode (`p`) retains its neutral background.
