# Clarity and frame cadence

Goal: prevent washed-out transitions, restore moving surface detail in Drain,
and reduce its frame cost and jitter without reducing default speed or resolution.

The reference is the immutable `benchmarks/whirlpool-after` executable and source
archive. Preserve the current workspace and all existing controls. No commit or
separate branch is needed for these requested corrections.

## Investigation

1. Sweep a complete Journey cycle, recording HDR luminance/chroma before and
   after postprocessing and retinal persistence, plus the actual display output.
2. Compare Drain and Classic rain response and surface slopes. Drain currently
   bypasses the dispersive wave geometry once its mode transition completes.
3. Profile real elapsed-time feedback, hydraulic retries, render phases and PTY
   frame delivery. Fixed-60-Hz CPU benchmarks alone cannot establish smoothness.

## Changes and checks

1. Add a regression for exposure-driven brightening of a fading retinal echo;
   correct metering in the same radiance domain as the visible image. Check
   world transitions and highlight/chroma distributions, not just one still.
2. Reuse hydraulic concentration quotients within each RK stage. Compare the
   entire derivative and mass ledger against the original implementation.
3. Apply measured hydraulic scheduling/step optimizations only after profiling;
   retain CFL/positivity validation and advance the complete physical interval.
4. Add bounded, current-coupled small-wave detail for Drain with documented
   approximation limits. Hydraulic depth, discharge and dye remain authoritative.
   Check finite amplitudes, zero-mean detail, rain response and frame partition.
5. Run cargo check before Rust execution, focused regression tests, complete
   release tests, strict Clippy and formatter checks. Freeze the final sources.
6. Compare matched native before/after timings and PTY cadence; inspect captured
   frames and motion. Save measurements and a comparison table under benchmarks.

The additional request for smoother perceived motion is implemented by a separate
timestamped presentation worker. It blends completed frames in linear display
light, keeps a bounded history and adds no recursively accumulated blur. Blending
starts enabled and `i` toggles it. Pause, help and viewport changes reset the
history. Measure display cadence separately from complete scene computation.

Keep soft rose/lilac/mint colors, reflection/refraction/interference/diffraction,
the default Drain and speed 3. Do not trade the regression for a brighter global
light, extra fake foam, half-resolution Auto output, or dropped simulation time.
