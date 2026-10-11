# Native renderer checks

Run these commands from the repository root on macOS with the Swift/macOS SDK
tools and a Metal device. Generated captures, executables and reports under
`native/artifacts/` are local outputs and are not distributed with the source.
For the cross-platform WGPU implementation, see [desktop checks](../../desktop/README.md).

The image checker reads the actual PNG captures produced by the Metal executable. It does not launch the terminal renderer or synthesize replacement images.

First generate a capture, then inspect it:

```sh
./native/run.sh --mode classic --frames 180 --width 1280 --height 720 \
  --capture native/artifacts/classic.png
uv run --script native/Tests/check_images.py native/artifacts/classic.png
```

The checker saves `native/artifacts/image-checks/report.json` and `report.md`, and prints the table. Use `--output PATH` to preserve separate capture runs. After producing two same-size captures, exact expected drawable dimensions and comparisons are optional:

```sh
uv run --script native/Tests/check_images.py 'native/artifacts/classic-*.png' \
  --expect-size 1280x720 --output native/artifacts/classic-checks \
  --compare native/artifacts/classic-001.png native/artifacts/classic-002.png
```

Only explicitly named `--compare FIRST SECOND` pairs are compared. Naming order, scene changes, and capture intervals are not guessed. Comparisons use all pixels at their original dimensions, without resizing; mismatched extents are reported as skipped. A static pair is diagnostic and may be expected for paused captures.

Missing/invalid PNGs, an explicitly requested dimension mismatch, and an entirely black output fail the command. White/gray coverage and low saturation are reported for review without aesthetic pass/fail thresholds. Metrics use decoded 8-bit RGB values and HSV saturation; the reported Rec.709 weighted value is **encoded RGB brightness**, not linear physical luminance. Hue coverage counts only reasonably bright, saturated pixels.

PNG decoding guarantees finite encoded values. It cannot prove that the upstream HDR shader never generated NaN/Inf before conversion; that requires separate GPU validation. Image differences do not measure displayed FPS, frame pacing, or perceived smoothness. GPU command duration, CPU duration, and presentation cadence must be measured and named separately.

Run the checker regression tests with Python 3.14:

```sh
uv run --python 3.14 --with pillow python -m unittest discover \
  -s native/Tests -p 'test_check_images.py' -v
```

These use small known PNG fixtures to verify primary-color coverage, passive gray/white diagnostics, black/invalid-image failures and exact frame differences.

The independent camera property test compiles directly against the production Swift camera and C ABI header, without the Rust library or rendering executable:

```sh
mkdir -p native/artifacts/camera-checks
xcrun swiftc -import-objc-header native/include/AcidRainCore.h \
  native/Sources/Camera.swift native/Tests/CameraTests.swift \
  -o native/artifacts/camera-checks/camera-tests
native/artifacts/camera-checks/camera-tests \
  > native/artifacts/camera-checks/report.md
cat native/artifacts/camera-checks/report.md
```

It tests camera basis orthogonality, finite forward intersections, preservation of drawable aspect and a 6 cm patch-edge guard through an orbit/roll sequence, portrait/wide extents, shifted targets and normal/emergency water-height slabs. It checks the emitted basis geometrically instead of duplicating the production fitting algorithm. Additional contracts cover invariant aperture under waves and patch rebasing, cadence-independent camera smoothing, world-space hold, and gradual release after emergency height fitting. This does not validate pixel shading.

Actual GPU shader tests compile the production `Water.metal` and its bed/caustic
dependencies with Metal's runtime compiler and production fast math, then execute
their functions and render passes on the available GPU:

```sh
mkdir -p native/artifacts/shader-checks
xcrun swiftc -parse-as-library -import-objc-header native/include/AcidRainCore.h \
  native/Tests/ShaderTests.swift -o native/artifacts/shader-checks/shader-tests
native/artifacts/shader-checks/shader-tests
```

These save and print `native/artifacts/shader-checks/report.{json,md}` with the GPU name and exact shader hash. They check dielectric Fresnel against independent equations, pigment extinction, HDR tone mapping and color ratios, lattice alignment, flat-surface intersection, actual postprocessing of constant and spatially varying RGB, sRGB attachment encoding, and unlit dyed water's passive black limit. No offline Metal compiler, Rust build, or visible window is required. The tests append test-only GPU entry points in memory; they do not edit the production shader.

The subjective layer has a separate actual-GPU suite:

```sh
mkdir -p native/artifacts/retina-checks
xcrun swiftc -parse-as-library -import-objc-header native/include/AcidRainCore.h \
  native/Tests/RetinaTests.swift -o native/artifacts/retina-checks/retina-tests
native/artifacts/retina-checks/retina-tests
```

It tests clean bypass, neutral/black preservation, bounded moving RGB offsets in logical
pixels, frozen output, invalid-history handling, time-normalized reservoirs and per-channel
diffraction energy. Reports include the production shader hash. The build also runs
`TemporalPolicyTests.swift` and `CoreClockTests.swift` for publication jitter, display
cadences, command republishing, pause/resume, history half-lives and bounded memory.
GPU correctness and frame presentation smoothness remain separate checks.

The cloud transport, spectral caustics and procedural bed have additional
production-shader suites:

```sh
mkdir -p native/artifacts/cloud-checks native/artifacts/caustics-checks native/artifacts/bed-life-checks
xcrun swiftc -parse-as-library -import-objc-header native/include/AcidRainCore.h \
  native/Tests/CloudTests.swift -o native/artifacts/cloud-checks/cloud-tests
native/artifacts/cloud-checks/cloud-tests native/Sources/Cloud.metal \
  native/artifacts/cloud-checks/report.json
xcrun swiftc -parse-as-library native/Tests/CausticsTests.swift \
  -o native/artifacts/caustics-checks/caustics-tests
native/artifacts/caustics-checks/caustics-tests native/Sources/Caustics.metal \
  native/artifacts/caustics-checks
xcrun swiftc -parse-as-library native/Tests/BedLifeTests.swift \
  -o native/artifacts/bed-life-checks/bed-life-tests
native/artifacts/bed-life-checks/bed-life-tests native/Sources/BedLife.metal \
  native/artifacts/bed-life-checks
```

Cloud checks cover conservative transport, injection, bleaching and world rebasing.
Caustics checks cover spectral flux, focusing, temporal continuity and intersections
with actual bathymetry. Bed checks cover material bounds, animation, world anchoring
and footprint filtering. Each suite creates its own report; a source checkout does
not include the results of an earlier run.

To collect separate offscreen timings and real-window presentation diagnostics:

```sh
./native/run.sh --frames 240 --warmup 30 --width 1280 --height 720 \
  --benchmark native/artifacts/offscreen-timings.json
./native/run.sh --smoke 10 --exercise-controls \
  --benchmark native/artifacts/window-smoke.json
```

Offscreen CPU/GPU durations describe frame preparation without display presentation.
The window run requires an interactive desktop session and records presentation
cadence separately. Preserve the source revision, device, dimensions, settings and
raw reports when comparing runs; neither command establishes performance on another
machine or backend.
