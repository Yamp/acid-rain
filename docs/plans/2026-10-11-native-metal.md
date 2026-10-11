# Native macOS Metal edition

**Goal:** Build a separate, runnable macOS application beside the terminal edition, preserving the current water, pigment and streaming world while rendering continuous pixels with Metal.

**Architecture:** Keep the existing Rust simulation as the source of truth, compiled into a separate static library. A native Swift/AppKit host uploads bounded simulation/material textures; Metal traces and shades the water at window resolution and applies HDR optical post-processing. The existing terminal files and binary remain intact.

**Tech stack:** Rust 2024, Swift, AppKit, MetalKit, Metal Shading Language, local Xcode toolchain; no new downloaded graphics framework.

## Design decisions

- Use `native/` in the current checkout: the terminal work is already uncommitted and is the intended shared baseline.
- A CPU-framebuffer-only viewer would preserve the image but retain the terminal renderer's resolution/cost limits. A complete simulation rewrite would duplicate tested physics. Share simulation and move per-pixel optics to Metal.
- Keep Classic/Journey, speed 2.4, intensity 4, rain 32, drop diameter 1.5, bleaching 1, RGB displacement 2 logical pixels and travelling world as defaults.
- Bound CPU simulation, texture residency and frames in flight; quality controls adjust GPU render resolution, with adaptive quality available.
- Use linear HDR working textures and one final sRGB transfer. Preserve saturated highlight chroma and use actual displaced channel sampling.

## Implementation

1. **Shared core bridge** (`native/core/`, `native/include/AcidRainCore.h`): test ABI layout, valid finite texture exports, pause/control behavior and simulation continuity; compile existing modules by path without changing them. Export height/normal, pigment/depth, cached bed and environmental radiance.
2. **Metal renderer** (`native/Sources/Water.metal`): bounded height-field intersection, orbiting/translating camera, water Fresnel reflection, dispersive refraction, Beer attenuation through water/pigment, bed and foam shading. Add chroma-preserving tone mapping, restrained bloom/diffraction, channel displacement and bounded temporal persistence.
3. **Native host** (`native/Sources/`, `native/build.sh`, `native/run.sh`, `native/Info.plist`): MTKView, resizable Retina window/fullscreen, controls and help overlay, safe GPU resource ownership, packaged `.app`. Add deterministic offscreen GPU capture and measured benchmark output.
4. **Integration verification**: `cargo check` before core tests/build, existing terminal regression gates, compile shaders and Swift, run actual Metal capture and window smoke. Inspect images, exercise native controls/resize/pause, record measured CPU/GPU frame timings with hardware and resolution.
5. **Delivery** (`native/README.md`, root README): document launch, controls, quality, implementation boundaries and test evidence; leave runnable local app bundle and captures.

No commit, push, signing identity or installation outside the project is required.
