# Smooth optics validation — 0.2.1

The original macOS application and terminal are the primary acceptance targets.
The portable desktop renderer uses the same revised optical and retinal models.
Preferred startup controls and simulation resolution are unchanged.

## Numerical accuracy

Tests execute the production Metal/WGSL functions and compare them with independent
polynomials, dense double-precision integration or analytic temporal responses.
On the Apple M1 Pro, the Metal reference fixtures measured:

| Contract | Result |
| --- | ---: |
| Moving GGX highlight relative RMS error | 8.765% before; 0.2308% after |
| Integrated highlight energy / reference | 0.998613 |
| Worst phase-dependent energy error | 0.1922% |
| Worst tested varying-normal, oblique or grazing GGX error | 0.6318% |
| Mixed quadratic height and gradient maximum error | 1.24 × 10⁻⁶ |
| Height error at sampled wave nodes | 0 |
| Tested cubic ray intersection residual | 9.13 × 10⁻⁸ m |
| Degenerate-Newton crest residual | 1.465 mm before; 0.0000894 mm after |

These are errors of the stated fixtures, not universal error bounds. The water is
a finite height field: bounded intersection searches can miss crests between
marching probes or select a later crossing, and local pixel quadrature is not
complete path tracing. Environment
magnification is smooth but is not a full spherical radiance convolution.
See [the optical model and its limits](OPTICS.md#gpu-surface-reconstruction-and-highlight-integration).

Additional contracts cover cubic crests missed by bilinear intersections, upward
reflections into steep crests, the conservative height envelope, consistent drop
contact and picking, rotationally invariant terminal normal filtering, and bloom
flux at odd-sized image edges. Wave batching retains all 384 spectral modes and
769 radial nodes; independent f64 summation checks its rounding error.

Three unwarped HDR histories have attack half-lives of 0.025/0.050/0.090 seconds
and release half-lives of 0.24/0.72/1.65 seconds. Actual GPU pulse tests at
30/60/120 Hz check the analytic tail. Area restriction preserves impulse flux
across fractional reduction ratios, and quality changes retain history textures.
The subjective layer is authored perception, not a validated biological model.

## Performance on Apple M1 Pro

Final sequential baseline/candidate pairs, milliseconds per measured frame:

| Native output / quality | CPU simulation before → after | GPU mean before → after | GPU p95 before → after |
| --- | ---: | ---: | ---: |
| 1280×720 / High | 20.090 → 12.293 | 8.937 → 8.094 | 19.613 → 14.008 |
| 1920×1080 / High | 20.128 → 12.309 | 12.713 → 12.928 | 24.357 → 22.057 |
| 1920×1080 / Native | 20.088 → 12.336 | 13.716 → 15.075 | 22.198 → 15.025 |

The simulation uses about 39% less CPU time. GPU mean improves at 720p, is nearly
unchanged at 1080p High, and costs about 10% more at full-resolution 1080p Native.
Previous diagnostic pairs showed appreciable GPU timing variation; these are
single final pairs, not confidence intervals or a universal speedup claim.

Terminal High simulation, shading and ANSI encoding into a sink:

| Terminal cells | Mean before → after, ms | p95 before → after, ms |
| --- | ---: | ---: |
| 120×40 | 30.376 → 30.021 | 31.016 → 30.796 |
| 200×60 | 30.529 → 30.245 | 31.271 → 30.850 |
| 320×100 | 76.900 → 76.790 | 78.004 → 78.423 |

Terminal frame cost is essentially unchanged; emulator rendering and transport
are excluded. The isolated radial-wave builder is 2.5–5.9× faster for batches of
4–128 sources, with one-source cost unchanged. That mechanism speedup is not a
whole-frame speedup.

Separate 15-second native windows with fixed preferred controls, no exercised
shortcuts, and the same actual 1920×1460 drawable (1450×1103 internal High)
presented 56.678 → 57.246 frames/s. Both had a 16.667 ms median interval and
25.000 ms p95; maximum intervals were 87.50 → 100.00 ms. This does **not** establish
a material displayed-cadence improvement. Window layout imposed a taller drawable
than the requested 960×540 logical-point size; measured extents are reported here.

## Reproduction and evidence

Baseline: `1a542c21be747e4817c4c3746838a5218b0b0c8d`.
Measurements use frozen executable/shader copies, matching seed 7, identical
controls and sequential baseline/candidate runs. Offline native fixtures use
360 fixed-60-Hz frames with 120 warmup, with High or Native quality at 1280×720
and 1920×1080. High at 1920×1080 has a 1686×948 internal scene extent in both
versions. Terminal fixtures use 30 warmup and 180 measured frames; their fixed
impact workload is eight drops per physical second, independent of the runtime
rain control. Simulation, GPU command time and actual window presentation are
separate measurements.

Local raw timings, exact commands, binary and source hashes, full-resolution A/B
captures, numerical reports and smoke logs are retained in
`artifacts/smooth-optics/` in the implementation worktree. The release's
`validation.zip` contains CI logs and software-GPU captures for its exact commit.
Windows WARP and Linux software Vulkan establish shader execution and numerical
contracts; their timings are not hardware GPU performance claims.

Validation includes the Rust workspace, strict Clippy, formatting, native camera
and optical suites, 150 terminal PTY checks and 50 native window controls.
Static images show smoother portions of the highlight contours, with fine
irregularities remaining; they do not prove that all temporal shimmer is gone.

Native optical commands are documented in [the test guide](../native/Tests/README.md).
Portable numerical tests run with `ACID_RAIN_GPU_TESTS=1` and the selected WGPU
backend. Begin Rust verification with `cargo check --locked --workspace --all-targets`.
