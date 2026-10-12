# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-10-12

### Changed
- Continuous reconstructed GPU water geometry with matching optical normals and caustics.
- Footprint-aware highlight integration and rotationally consistent terminal normal filtering.
- Longer colored afterimages with fast capture, separate decay rates and conserved source-area sampling.
- Shared batched spectral wave evaluation and exact-time physical-impact profile reuse, retaining the complete wave basis.

### Fixed
- Highlight reconstruction, bloom motion and edge-energy artifacts.
- Terminal interpolation stalls when source rendering and delivery exceed the old delay cap.
- Retinal history being cleared by quality changes or slow active frames.

Numerical accuracy, measurement conditions and remaining model limits are recorded
in [the validation report](docs/SMOOTH_OPTICS_VALIDATION.md).

## [0.2.0] - 2026-10-11

### Added
- A windowed Rust/wgpu edition for Windows, Linux and macOS, with portable release archives.
- Shared water physics, conservative 3D pigment clouds, spectral caustics and procedural beds.
- Smooth camera presentation, floating RGB channels, retinal histories and diffraction.
- Manual/automatic GPU quality, full keyboard help, drop observation and headless PNG/JSON diagnostics.
- Original Swift/Metal application and enhanced terminal renderer in the same repository.
- Cross-platform CI and real GPU conservation/optics contracts, including software Vulkan.

### Changed
- Startup profile: Classic water, Journey, High quality, speed 2.4x, Trip 4, RGB shift 2px.
- Shared Cargo workspace with checked-in dependencies; published builds retain portable CPU targets.

## [0.1.2] - 2026-02-27

### Added
- Windows binary in GitHub Release artifacts.

## [0.1.1] - 2026-02-27

### Added
- GitHub Release automation for Linux and macOS binaries.

### Changed
- Resolved merge conflict in `src/term.rs`.

## [0.1.0] - 2023-04-XX

### Added
- Initial release of Acid Rain on crates.io.
