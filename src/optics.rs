//! Cached visible-band coating response and a gentle pastel emitter spectrum.
use once_cell::sync::Lazy;

use crate::color_pipeline::luminance;

const FILM_STEPS: usize = 2048;
const MAX_PATH_NM: f32 = 2400.0;

// Air / thin dielectric film / water. Normal-incidence Airy interference is
// averaged into broad RGB sensitivity bands; angle enters the optical path.
static FILM: Lazy<[[f32; 3]; FILM_STEPS + 1]> = Lazy::new(|| {
    let r01 = (1.0_f64 - 1.46) / (1.0 + 1.46);
    let r12 = (1.46_f64 - 1.333) / (1.46 + 1.333);
    std::array::from_fn(|i| {
        let path = i as f64 / FILM_STEPS as f64 * MAX_PATH_NM as f64;
        std::array::from_fn(|c| {
            let center = [640.0_f64, 545.0, 450.0][c];
            let mut reflection = 0.0;
            let mut normalization = 0.0;
            for band in 0..9 {
                let wavelength = center + (band as f64 - 4.0) * 12.0;
                let weight = (-0.5 * ((wavelength - center) / 32.0).powi(2)).exp();
                let interference =
                    2.0 * r01 * r12 * (std::f64::consts::TAU * path / wavelength).cos();
                let airy = (r01 * r01 + r12 * r12 + interference)
                    / (1.0 + r01 * r01 * r12 * r12 + interference);
                reflection += airy * weight;
                normalization += weight;
            }
            (reflection / normalization) as f32
        })
    })
});

pub fn film_reflection(
    base: [f32; 3],
    cos_view: f32,
    height: f32,
    strength: f32,
    phase: f32,
) -> [f32; 3] {
    if strength <= 0.0 {
        return base;
    }
    let thickness = 475.0 + 220.0 * (phase * 0.35 + height * 18.0).sin();
    let cosine = (1.0 - (1.0 - cos_view.clamp(0.0, 1.0).powi(2)) / (1.46 * 1.46)).sqrt();
    let path = 2.0 * 1.46 * thickness * cosine;
    let coordinate = (path / MAX_PATH_NM * FILM_STEPS as f32).clamp(0.0, FILM_STEPS as f32);
    let lo = (coordinate as usize).min(FILM_STEPS - 1);
    let t = coordinate - lo as f32;
    let amount = strength.clamp(0.0, 2.0) * 0.65;
    std::array::from_fn(|c| {
        let coating = FILM[lo][c] + (FILM[lo + 1][c] - FILM[lo][c]) * t;
        (base[c] + (1.0 - base[c]) * (coating - 0.0204) * amount).clamp(0.0, 1.0)
    })
}

/// Soft pigments, with luminance normalization supplied by the emitter mixer.
pub fn pastel_spectrum(phase: f32) -> [f32; 3] {
    const PIGMENTS: [[f32; 3]; 7] = [
        [0.70, 0.29, 0.50],
        [0.43, 0.29, 0.70],
        [0.25, 0.38, 0.70],
        [0.24, 0.60, 0.58],
        [0.32, 0.65, 0.36],
        [0.72, 0.55, 0.44],
        [0.68, 0.40, 0.42],
    ];
    let p = phase.rem_euclid(7.0);
    // A tiny negative f32 can wrap to exactly 7.0 after rounding. Wrap the
    // integer index too, so the seam remains continuous and in bounds.
    let i = p.floor() as usize % PIGMENTS.len();
    let t = p.fract();
    let t = t * t * (3.0 - 2.0 * t);
    std::array::from_fn(|c| PIGMENTS[i][c] + (PIGMENTS[(i + 1) % 7][c] - PIGMENTS[i][c]) * t)
}

pub fn spectral_emitter(
    rgb: [f32; 3],
    direction: [f32; 3],
    phase: f32,
    amount: f32,
) -> [f32; 3] {
    let y = luminance(rgb);
    if y <= 1e-8 || amount <= 0.0 {
        return rgb;
    }
    let [x, y_dir, z] = direction;
    let chroma_phase = 3.8 + x * 11.0 - y_dir * 9.0
        + z * 4.0
        + (x * y_dir * 6.0 + phase * 0.7).sin() * 1.7
        + phase * 0.16;
    let pigment = pastel_spectrum(chroma_phase);
    let pigment_y = luminance(pigment).max(1e-6);
    let fraction = amount.clamp(0.0, 0.90);
    // Blend equal-luminance emitters rather than multiplying pigments: a rose
    // source under a peach spectrum must not compound into a harsh red.
    let tinted: [f32; 3] =
        std::array::from_fn(|c| rgb[c] * (1.0 - fraction) + fraction * y * pigment[c] / pigment_y);
    let scale = y / luminance(tinted).max(1e-8);
    tinted.map(|c| c * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pastel_palette_wraps_negative_roundoff_without_indexing_past_the_end() {
        let seam = pastel_spectrum(0.0);
        // rem_euclid can round a tiny negative phase up to exactly 7.0.
        // This comes from emitter phases crossing zero during normal animation.
        for phase in [
            -f32::from_bits(1),
            -f32::MIN_POSITIVE,
            -1e-7,
            -2e-7,
            0.0,
            7.0,
            -7.0,
        ] {
            let color = pastel_spectrum(phase);
            for channel in 0..3 {
                assert!(color[channel].is_finite());
                assert!(
                    (color[channel] - seam[channel]).abs() < 1e-6,
                    "phase={phase}"
                );
            }
        }
    }

    #[test]
    fn film_is_passive_bounded_and_has_spectral_variation() {
        let base = [0.0204; 3];
        assert_eq!(film_reflection(base, 0.7, 0.02, 0.0, 0.0), base);
        assert_eq!(film_reflection([1.0; 3], 0.0, 0.0, 2.0, 1.0), [1.0; 3]);
        let mut variation = 0.0_f32;
        for i in 0..100 {
            let p = film_reflection(base, 0.7, 0.03, 1.0, i as f32 * 0.1);
            assert!(p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            variation = variation.max((p[0] - p[2]).abs());
        }
        assert!(variation > 0.003);
    }

    #[test]
    fn spectral_emitters_preserve_luminance_and_have_no_palette_seam() {
        let original = [0.53, 0.31, 0.43];
        for direction in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let color = spectral_emitter(original, direction, 1.0, 0.75);
            assert!((luminance(color) - luminance(original)).abs() < 1e-6);
            assert!(color.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
        let a = pastel_spectrum(7.0 - 1e-5);
        let b = pastel_spectrum(0.0);
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-6);
        }
    }
}
