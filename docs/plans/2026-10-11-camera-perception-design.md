# Stable native camera and subjective perception

The user observes small jerky zoom changes and requests a smoother camera plus
subjective psychedelic postprocessing: drifting RGB channels, patterned color
adaptation, variable-strength layered trails and retinal diffraction.

## Confirmed camera causes

The current frustum is recomputed from instantaneous wave extrema and whichever
viewport corner limits the resident patch. A one-cell world rebase moves the local
target and changes the fitted aperture despite continuous physical motion. The
diagnostic in `native/artifacts/camera-motion-audit/before.json` measures a 1.768%
aperture change (26.2 Retina pixels for a stationary landmark) at a 2 micrometre
world displacement. Changing height extrema also pumps the lens.

The host extrapolates from the *publication* timestamp, after variable CPU work.
That timestamp does not identify the time represented by the simulation sample;
publishing unchanged state after a command restarts prediction as well. Recorded
counterexamples make camera time move backward on a later display frame.

## Design

- Replace corner-dependent fitting with a conservative analytic cone covering all
  headings and roll angles, a fixed water-height envelope and a small target reserve.
  Keep edge coverage, aspect ratio and picking correct. The modest constant framing
  tradeoff replaces reactive zoom. Validate native mode extrema and extreme poses.
- Track the actual simulation sample timestamp independently of publication time.
  Use a continuous bounded presentation clock and smooth camera parameters in world
  coordinates; handle rebasing, pause, speed changes and stalls explicitly. Frame
  blending can be disabled without reintroducing camera clock jumps.
- Keep physical scene rendering intact. Maintain three bounded half-resolution HDR
  temporal reservoirs with exponential half-lives of 0.10, 0.35 and 0.80 active wall
  seconds. Accumulate raw, unwarped scene radiance; never feed distorted post output
  into its own history. Freeze on hold, initialize/reset before sampling.
- Compose restrained, normalized old-frame overlays with independent slow drift.
  Shift actual R/G/B samples independently within the configured logical-pixel
  radius. Add coherent spatial chroma adaptation and normalized wavelength-dependent
  diffraction, then tone-map the recombined linear image once. Keep neutral/black
  inputs stable and avoid additive white veils.
- Subjective intensity defaults to 1, adjustable with `n/m`, reset with `N`, CLI
  `--lsd 0...2`. Existing `e`, Trip and RGB-radius controls remain meaningful. Falling
  spheres stay hidden by default. No terminal control/default changes are intended.

Pure smoothing of the old fit would lag behind shrinking safe bounds and could
expose world edges. A large ring of full-size saved frames would add avoidable
memory and bandwidth. Stable geometry and bounded time-based reservoirs address
the observed causes and the requested effects at predictable cost.

## Verification

Reproduce camera jumps before changes, then verify invariant framing, continuous
world-space motion, timestamp behavior, viewport edge safety and picking. Test
history semigroup/cadence invariance and hold/reset behavior independently. Execute
production Metal tests for RGB bounds/drift, constant-field and black preservation,
color bounds, diffraction normalization and invalid-history bypass. Build and sign
the native app, inspect actual captures/UI, run controls/resizing and isolated
matched performance/fullscreen checks. Save and print timing tables; distinguish
display cadence from CPU physics rate. Preserve the terminal release binary.

The existing dirty workspace is retained; no unrelated rollback or integration
commit is part of this request.
