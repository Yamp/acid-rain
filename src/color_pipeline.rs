//! Linear Rec.709 / D65 light to an SDR sRGB terminal.
//! A custom display transform with a chromaticity-preserving intensity shoulder.
//! Saturated HDR light keeps its channel ratios instead of approaching white;
//! its display luminance is consequently lower than a neutral of equal HDR Y.
//! This is not an ACES output transform or an ICC monitor profile.
use once_cell::sync::Lazy;

use crate::scene::SceneFrame;

pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
const TRANSFER_STEPS: usize = 8192;

#[inline(always)]
pub fn luminance(rgb: [f32; 3]) -> f32 { rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2] }

static SRGB: Lazy<[f32; TRANSFER_STEPS + 1]> = Lazy::new(|| {
    std::array::from_fn(|i| {
        let value = i as f64 / TRANSFER_STEPS as f64;
        if value <= 0.0031308 {
            (12.92 * value) as f32
        } else {
            (1.055 * value.powf(1.0 / 2.4) - 0.055) as f32
        }
    })
});

static LINEAR_BYTES: Lazy<[f32; 256]> = Lazy::new(|| {
    std::array::from_fn(|i| {
        let value = i as f64 / 255.0;
        if value <= 0.04045 {
            (value / 12.92) as f32
        } else {
            ((value + 0.055) / 1.055).powf(2.4) as f32
        }
    })
});

#[inline(always)]
pub fn encode_srgb(value: f32) -> f32 {
    let value = if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let coordinate = value * TRANSFER_STEPS as f32;
    let lower = (coordinate as usize).min(TRANSFER_STEPS - 1);
    let t = coordinate - lower as f32;
    SRGB[lower] + (SRGB[lower + 1] - SRGB[lower]) * t
}

#[inline(always)]
pub fn decode_byte(value: u8) -> f32 { LINEAR_BYTES[value as usize] }

/// Bring hue/saturation excursions into the nonnegative RGB cone. Only negative
/// channels require chroma contraction toward the same-luminance neutral; HDR
/// magnitude is left alone for the subsequent common-channel shoulder.
#[inline(always)]
fn nonnegative_chroma(
    rgb: [f32; 3],
    y: f32,
) -> [f32; 3] {
    let chroma = rgb.map(|c| c - y);
    let minimum = rgb.into_iter().fold(f32::INFINITY, f32::min);
    let scale = if minimum < 0.0 {
        y / (y - minimum)
    } else {
        1.0
    };
    if scale == 1.0 {
        rgb
    } else {
        chroma.map(|c| (y + c * scale).max(0.0)) // Numerical roundoff guard.
    }
}

pub struct DisplayTransform {
    exposure: f32,
    contrast: f32,
    saturation: f32,
    gain: [f32; 3],
    hue: [[f32; 3]; 3],
}

impl DisplayTransform {
    pub fn new(
        scene: &SceneFrame,
        exposure: f32,
    ) -> Self {
        let (s, c) = scene.hue_angle.sin_cos();
        let gain_y = luminance(scene.gain).max(1e-6);
        Self {
            exposure: (scene.exposure * exposure).clamp(0.01, 32.0),
            contrast: scene.contrast.clamp(0.5, 1.5),
            saturation: scene.saturation.clamp(0.0, 2.0),
            gain: scene.gain.map(|v| v / gain_y),
            hue: [
                [
                    0.213 + 0.787 * c - 0.213 * s,
                    0.715 - 0.715 * c - 0.715 * s,
                    0.072 - 0.072 * c + 0.928 * s,
                ],
                [
                    0.213 - 0.213 * c + 0.143 * s,
                    0.715 + 0.285 * c + 0.140 * s,
                    0.072 - 0.072 * c - 0.283 * s,
                ],
                [
                    0.213 - 0.213 * c - 0.787 * s,
                    0.715 - 0.715 * c + 0.715 * s,
                    0.072 + 0.928 * c + 0.072 * s,
                ],
            ],
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn apply(
        &self,
        hdr: [f32; 3],
    ) -> [f32; 3] {
        self.apply_linear(hdr).map(encode_srgb)
    }

    /// Gamut-bounded display light before the sRGB transfer. Perceptual
    /// channel displacement belongs here, after the HDR intensity shoulder.
    #[inline(always)]
    pub fn apply_linear(
        &self,
        hdr: [f32; 3],
    ) -> [f32; 3] {
        let balanced: [f32; 3] = std::array::from_fn(|i| {
            let value = if hdr[i].is_finite() {
                hdr[i].clamp(0.0, 65504.0)
            } else {
                0.0
            };
            value * self.gain[i] * self.exposure
        });
        let y = luminance(balanced);
        if y <= 1e-10 {
            return [0.0; 3];
        }
        let rotated = self
            .hue
            .map(|row| row[0] * balanced[0] + row[1] * balanced[1] + row[2] * balanced[2]);
        let rotated_y = luminance(rotated);
        let graded = rotated.map(|value| y + (value - rotated_y) * self.saturation);
        let positive = nonnegative_chroma(graded, y);
        let peak = positive.into_iter().fold(0.0_f32, f32::max);
        // The same scalar curve still maps neutrals exactly as before. For
        // chromatic light, the peak sets SDR headroom and one multiplier keeps
        // channel ratios intact. Preserving a near-white output Y instead would
        // force saturated colors toward white: SDR cannot retain both at once.
        let tone = (peak * (peak * 2.51 + 0.03) / (peak * (peak * 2.43 + 0.59) + 0.14)).min(0.9995);
        let mapped_peak =
            (tone + (self.contrast - 1.0) * tone * (1.0 - tone) * (tone - 0.18)).clamp(0.0, 0.9995);
        positive.map(|value| (value / peak) * mapped_peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_transfer_matches_srgb_including_the_dark_toe() {
        for i in 0..=20_000 {
            let linear = i as f32 / 20_000.0;
            let reference = if linear <= 0.0031308 {
                12.92 * linear
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            assert!(
                (encode_srgb(linear) - reference).abs() < 1e-5,
                "linear={linear}"
            );
        }
    }
}

#[cfg(test)]
mod grading_tests {
    use super::*;
    use crate::scene::SceneKind;

    #[test]
    fn linear_display_output_matches_the_encoded_wrapper() {
        let scene = SceneFrame::new(SceneKind::Acid, 7.0, 2.0);
        let grade = DisplayTransform::new(&scene, 1.3);
        for input in [[0.0; 3], [0.02, 0.04, 0.06], [0.8, 0.4, 0.1], [16.0; 3], [
            16.0, 0.0, 0.0,
        ]] {
            let encoded = grade.apply(input);
            let linear = grade.apply_linear(input);
            assert!(
                linear
                    .iter()
                    .all(|c| c.is_finite() && (0.0..=1.0).contains(c))
            );
            assert_eq!(linear.map(encode_srgb), encoded, "input {input:?}");
        }
    }

    fn identity_grade() -> DisplayTransform {
        DisplayTransform {
            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            gain: [1.0; 3],
            hue: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    #[test]
    fn saturated_hdr_primaries_and_secondaries_do_not_turn_white() {
        let grade = identity_grade();
        for energy in [4.0, 16.0] {
            for color in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ] {
                let output = grade.apply_linear(color.map(|c| c * energy));
                for (channel, input) in output.into_iter().zip(color) {
                    if input == 0.0 {
                        assert!(channel < 1e-6, "{color:?} at {energy}: {output:?}");
                    } else {
                        assert!(channel > 0.9 && channel <= 1.0, "{output:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn exposure_changes_intensity_without_diluting_color_ratios() {
        let color = [1.0, 0.23, 0.04];
        for exposure in [0.01, 0.1, 1.0, 4.0, 16.0] {
            let mut grade = identity_grade();
            grade.exposure = exposure;
            let output = grade.apply_linear(color);
            for (channel, expected_ratio) in output.into_iter().zip(color) {
                assert!(
                    (channel / output[0] - expected_ratio).abs() < 2e-6,
                    "exposure={exposure}: {output:?}"
                );
            }
        }
    }

    #[test]
    fn hue_and_saturation_grading_remains_bounded_across_bright_exposures() {
        for kind in [
            SceneKind::Acid,
            SceneKind::Aurora,
            SceneKind::Storm,
            SceneKind::Abyss,
            SceneKind::Lava,
            SceneKind::Prism,
        ] {
            let scene = SceneFrame::new(kind, 91.0, 4.0);
            for source in [[1.0, 0.0, 0.0], [0.02, 1.0, 0.08], [0.1, 0.02, 1.0], [
                1.0, 0.3, 0.01,
            ]] {
                let reference = DisplayTransform::new(&scene, 0.25).apply_linear(source);
                let reference_peak = reference.into_iter().fold(0.0_f32, f32::max);
                for exposure in [1.0, 4.0, 16.0, 32.0] {
                    let output = DisplayTransform::new(&scene, exposure).apply_linear(source);
                    let peak = output.into_iter().fold(0.0_f32, f32::max);
                    for (actual, expected) in output.into_iter().zip(reference) {
                        assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
                        assert!(
                            (actual / peak - expected / reference_peak).abs() < 2e-6,
                            "{kind:?}, exposure={exposure}, source={source:?}: {output:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn neutral_shoulder_matches_the_previous_scalar_curve() {
        let mut grade = identity_grade();
        for contrast in [0.5, 1.0, 1.5] {
            grade.contrast = contrast;
            for value in [0.0_f32, 0.0001, 0.01, 0.1, 0.5, 1.0, 4.0, 16.0, 65504.0] {
                let tone = (value * (value * 2.51 + 0.03) / (value * (value * 2.43 + 0.59) + 0.14))
                    .min(0.9995);
                let expected = (tone + (contrast - 1.0) * tone * (1.0 - tone) * (tone - 0.18))
                    .clamp(0.0, 0.9995);
                let actual = grade.apply_linear([value; 3]);
                assert!(actual.into_iter().all(|c| (c - expected).abs() < 1e-6));
            }
        }
    }

    #[test]
    fn neutral_ramp_remains_neutral_monotonic_and_finite() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 0.0, 1.0);
        scene.gain = [1.0; 3];
        let grade = DisplayTransform::new(&scene, 1.0);
        let mut previous = 0.0;
        for value in [0.0, 0.0001, 0.01, 0.1, 0.5, 1.0, 10.0, 1000.0, 1e20] {
            let color = grade.apply([value; 3]);
            assert!(
                color
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
            assert!((color[0] - color[1]).abs() < 1e-6 && (color[1] - color[2]).abs() < 1e-6);
            assert!(color[0] >= previous);
            previous = color[0];
        }
        assert!(
            grade
                .apply([f32::NAN, -1.0, f32::INFINITY])
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn negative_gamut_correction_preserves_luminance_and_chroma_direction() {
        for rgb in [[1.7, -0.1, 0.9], [-0.7, 0.9, 2.2], [0.2, 0.4, 0.6]] {
            let y = luminance(rgb).clamp(0.01, 0.99);
            let output = nonnegative_chroma(rgb, y);
            assert!(output.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!((luminance(output) - y).abs() < 1e-6);
            let a = rgb.map(|v| v - y);
            let b = output.map(|v| v - y);
            assert!((a[0] * b[1] - a[1] * b[0]).abs() < 1e-6);
        }
        let hdr = [16.0, 0.4, 2.0];
        assert_eq!(nonnegative_chroma(hdr, luminance(hdr)), hdr);
    }

    #[test]
    fn byte_transfer_roundtrips_every_terminal_value() {
        for value in 0..=255_u8 {
            assert_eq!(
                (encode_srgb(decode_byte(value)) * 255.0).round() as u8,
                value
            );
        }
    }
}
