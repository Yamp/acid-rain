# Soft spectral optics and help screen

**Goal:** Complete runtime help on h, richer pastel spectral color, bounded multiple reflections, wavelength-shaped glare, optional RGB spatial/temporal splitting, and a coherent display-color pipeline.

**Architecture:** Preserve the accepted physical height model and wave time controls. Add a separate static help view; keep optical operations and trails in linear Rec.709/D65 HDR. Use a luminance tone curve, neutral-axis gamut compression and accurate cached sRGB transfer. Reflection paths have fixed bounce/throughput budgets; screen-space effects reuse buffers.

**Decisions:** Source illumination and a thin-film lookup produce soft colored layers, avoiding a global rainbow hue rotation. A three-channel retinal echo supplies the requested drifting split; e toggles it. Help suspends updates, preserves pause state, supports scrolling/resizing, and closes on h/?/Esc. q/Ctrl-C always quit. This is a custom terminal display transform, not an ACES certification or monitor calibration.

1. Archive current source/binary and preview in benchmarks/optics-before.
2. Implement help view and h/? action, complete key table, current settings and small-terminal scrolling. Add PTY checks for overlay idle, navigation, resize, closing/resuming and paused-state preservation.
3. Replace per-channel tone/gamma approximation with tested linear-light luminance grading, smoothly bounded gamut compression and accurate sRGB LUT. Blend scene transitions in linear light.
4. Introduce spatially varied pastel spectral emitter pigments and a cached wavelength-dependent thin-film response. Retain water IOR/geometry and bounded radiance.
5. Replace the single secondary bounce with a bounded iterative reflection path. Propagate Fresnel throughput; terminate with cached environment lighting and energy-aware transmission/scatter.
6. Add normalized wavelength-scaled diffraction/glare and a reusable optional RGB retinal history. Preserve constant fields and avoid unbounded brightness, frame-rate dependence or history reset jumps.
7. Run fmt/check/Clippy/tests/native build, PTY, visual scene/animation QA and saved matched before/after CPU measurements. Document artistic approximations and costs.

Alternatives: full spectral path tracing is too costly for the CPU terminal target; additional image-wide color filters would not supply layered optics. Bounded optical paths plus cached spectral responses and linear-light screen effects fit the existing architecture.
