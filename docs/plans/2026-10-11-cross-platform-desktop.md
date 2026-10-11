# Cross-platform desktop implementation plan

**Goal:** Ship an efficient Windows/Linux GPU application preserving the current Metal water, volumetric pigment, caustics, living bed and subjective effects; publish source and downloadable releases to Yamp/acid-rain.

**Architecture:** Rust winit/wgpu host, shared safe Rust simulation API, latest-frame worker, WGSL port of the existing algorithms. Keep the terminal program and native Swift/Metal host. Baseline FP32 manual sampling works without optional float filtering; enable equivalent hardware filtering where supported. Explicit aligned per-dispatch constants prevent cloud-step aliasing. Hardware limits are checked before rendering.

**Tech Stack:** Rust 2024, wgpu 27, winit 0.30, egui 0.33, shared native/core rlib, Naga validation, GitHub Actions on Windows/Linux/macOS.

Generated research artifacts are excluded from release archives. Numerical checks and reproducible capture commands accompany the source.

## Tasks and ownership

1. Root: workspace, dependency lock, shared shader sampling, module composition, integration, camera/control review, macOS GPU validation, publication.
2. Safe core facade: native/core Rust API and immutable owned snapshots, no C ABI changes; tests for layout, snapshot ownership and commands.
3. Shader port: Water WGSL including optical rendering, drops, exposure and retinal passes; preserve formulas and pre-tone ordering.
4. Cloud WGSL: FP32 normalization, prepare, velocity and conservative advection, aligned dispatch/source indexing; mass/positivity/bleach GPU fixtures.
5. Bed/caustics WGSL: world-anchored filtered material, bed hit, mapping and additive rasterization; planar energy and geometry fixtures.
6. GPU host: textures, buffers, bounded upload/dispatch, history/volume ping-pong, asynchronous simulation, capability checks, offline captures and benchmarks.
7. Application/policies: camera and temporal policies, controls, help/HUD, resize/occlusion, adaptive quality, CLI/offline options.
8. CI/release: check before build/test; source/tests on three OSes, software GPU correctness smoke where available, portable optimized archives plus checksums, macOS native bundle.

## Verification and completion

- Check before every Rust execution/build; meaningful shader/layout/clock regression tests precede ports.
- Validate WGSL with Naga; run actual GPU numerical fixtures and captures on this Mac.
- Test Windows and Linux through GitHub Actions, distinguishing software-driver correctness from real hardware performance.
- Preserve true-color terminal controls and all existing physics tests.
- Save and print benchmark tables; do not claim universal performance from this Mac or CI software adapters.
- Audit public staging for credentials, absolute personal paths, build outputs and irrelevant artifacts.
- Push the completed project, run CI to green, publish a versioned GitHub release, verify downloadable assets/checksums.
