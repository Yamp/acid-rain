//! Passive spectral extinction of dissolved pigments. The tracer carries
//! column-integrated mass; this module neither transports mass nor emits light.
//!
//! A homogeneous distributed source escapes with E(tau)=(1-exp(-tau))/tau.
//! Relative to the existing clear-water volume integral, dissolved pigment
//! therefore multiplies it by E(tau_water+tau_dye)/E(tau_water). Background
//! radiance traverses the whole column and still uses full Beer attenuation.
//! See https://pbr-book.org/4ed/Volume_Scattering/Transmittance .

/// Broad artistic extinction bands per unit column mass, in RGB order.
/// Their scale is tuned for the tracer's dimensionless mass, not a measured ink.
const CYAN_EXTINCTION: [f32; 3] = [2.0, 0.65, 0.30];
const MAGENTA_EXTINCTION: [f32; 3] = [0.30, 1.80, 0.45];
const AMBER_EXTINCTION: [f32; 3] = [0.25, 0.65, 1.80];
const EXTINCTION_SCALE: f32 = 0.85;

#[derive(Debug, Clone, Copy)]
pub struct PigmentTransport {
    /// Full-path Beer attenuation for radiance behind the dyed column.
    pub transmission: [f32; 3],
    /// Relative escape of an existing source distributed through that column.
    pub volume: [f32; 3],
}

#[inline]
fn optical_depth(
    areal_mass: [f32; 3],
    path_over_depth: f32,
) -> [f32; 3] {
    let path = path_over_depth.max(0.0);
    if path == 0.0 {
        return [0.0; 3];
    }
    let [cyan, magenta, amber] = areal_mass.map(|value| value.max(0.0));
    std::array::from_fn(|c| {
        let extinction = cyan * CYAN_EXTINCTION[c]
            + magenta * MAGENTA_EXTINCTION[c]
            + amber * AMBER_EXTINCTION[c];
        // Keep an empty column transparent even for an unbounded test path.
        if extinction == 0.0 {
            0.0
        } else {
            extinction * EXTINCTION_SCALE * path
        }
    })
}

/// Full-path pigment Beer transmittance, also suitable for thin foam films.
#[inline]
pub fn transmittance(
    areal_mass: [f32; 3],
    path_over_depth: f32,
) -> [f32; 3] {
    optical_depth(areal_mass, path_over_depth).map(|tau| (-tau).exp())
}

/// Exact homogeneous-medium factors relative to the existing water shading.
/// `water_transmittance` is the caller's already computed exp(-water_tau),
/// reused so pigment transport needs only three new exponentials per sample.
#[inline]
pub fn transport(
    areal_mass: [f32; 3],
    path_over_depth: f32,
    water_tau: [f32; 3],
    water_transmittance: [f32; 3],
) -> PigmentTransport {
    let dye_tau = optical_depth(areal_mass, path_over_depth);
    let transmission = dye_tau.map(|tau| (-tau).exp());
    PigmentTransport {
        transmission,
        volume: std::array::from_fn(|c| {
            escape_ratio(
                water_tau[c],
                dye_tau[c],
                water_transmittance[c],
                transmission[c],
            )
        }),
    }
}

#[inline(always)]
fn thin_escape(tau: f32) -> f32 {
    // Integral series through tau^4; avoids cancellation in 1-exp(-tau).
    1.0 + tau * (-0.5 + tau * (1.0 / 6.0 + tau * (-1.0 / 24.0 + tau / 120.0)))
}

#[inline]
fn escape_ratio(
    water_tau: f32,
    dye_tau: f32,
    water_beer: f32,
    dye_beer: f32,
) -> f32 {
    if dye_tau == 0.0 {
        return 1.0;
    }
    if dye_tau.is_infinite() {
        return 0.0;
    }
    let water = water_tau.max(0.0);
    if water.is_infinite() {
        return 1.0;
    }
    let total = water + dye_tau;
    const THIN: f32 = 0.05;
    let result = if total < THIN {
        thin_escape(total) / thin_escape(water)
    } else if water < THIN {
        ((1.0 - water_beer * dye_beer) / total) / thin_escape(water)
    } else {
        // Algebraically identical to E(total)/E(water). Normalize optical
        // depths first so two very thick finite media cannot overflow their sum.
        let largest = water.max(dye_tau);
        let fraction = (water / largest) / (water / largest + dye_tau / largest);
        fraction * (1.0 - water_beer * dye_beer) / (1.0 - water_beer)
    };
    // The exact ratio lies between full-column Beer and one. This only removes
    // floating-point roundoff, never a nonzero brightness floor.
    result.clamp(dye_beer, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(
        mass: [f32; 3],
        ratio: f32,
        water_tau: [f32; 3],
    ) -> PigmentTransport {
        transport(mass, ratio, water_tau, water_tau.map(|t| (-t).exp()))
    }

    #[test]
    fn clear_pigment_and_zero_path_are_exact_identity() {
        for water in [0.0_f32, 1e-8, 0.02, 1.0, 100.0] {
            for (mass, path) in [([0.0; 3], 1.4), ([0.2, 0.8, 0.3], 0.0)] {
                let result = evaluate(mass, path, [water; 3]);
                assert_eq!(result.transmission, [1.0; 3]);
                assert_eq!(result.volume, [1.0; 3]);
            }
        }
    }

    #[test]
    fn spectral_coefficients_soften_extinction_by_fifteen_percent() {
        let actual = transmittance([0.2, 0.3, 0.4], 1.2);
        let old_tau = [
            0.2 * 2.0 + 0.3 * 0.30 + 0.4 * 0.25,
            0.2 * 0.65 + 0.3 * 1.80 + 0.4 * 0.65,
            0.2 * 0.30 + 0.3 * 0.45 + 0.4 * 1.80,
        ];
        for channel in 0..3 {
            let expected = (-old_tau[channel] * 0.85_f32 * 1.2).exp();
            assert!((actual[channel] - expected).abs() < 1e-7);
            assert!(actual[channel] > (-old_tau[channel] * 1.2).exp());
        }
    }

    #[test]
    fn distributed_escape_matches_homogeneous_volume_integration() {
        for water in [0.0_f32, 1e-7, 0.02, 0.4, 8.0, 200.0] {
            let mass = [0.2, 0.3, 0.4];
            let result = evaluate(mass, 1.2, [water; 3]);
            for channel in 0..3 {
                let dye = -result.transmission[channel].ln() as f64;
                let integral = |tau: f64| {
                    if tau == 0.0 {
                        1.0
                    } else {
                        -(-tau).exp_m1() / tau
                    }
                };
                let expected = integral(water as f64 + dye) / integral(water as f64);
                assert!((result.volume[channel] as f64 - expected).abs() < 2e-6);
                assert!(result.volume[channel] > result.transmission[channel]);
            }
        }
    }

    #[test]
    fn both_factors_are_passive_and_monotonic_without_a_brightness_floor() {
        let mut previous = PigmentTransport {
            transmission: [1.0; 3],
            volume: [1.0; 3],
        };
        for amount in [0.0, 1e-8, 0.001, 0.1, 1.0, 10.0, 1000.0, 1e8] {
            let result = evaluate([amount; 3], 1.0, [0.03, 0.04, 0.03]);
            for c in 0..3 {
                assert!(
                    result.transmission[c].is_finite()
                        && (0.0..=1.0).contains(&result.transmission[c])
                );
                assert!(result.volume[c].is_finite() && (0.0..=1.0).contains(&result.volume[c]));
                assert!(result.transmission[c] <= previous.transmission[c]);
                assert!(result.volume[c] <= previous.volume[c]);
                assert!(result.volume[c] >= result.transmission[c]);
                assert_eq!(
                    0.0 * result.volume[c],
                    0.0,
                    "attenuation cannot create a light source"
                );
            }
            previous = result;
        }
        assert!(previous.transmission.iter().all(|v| *v == 0.0));
        assert!(previous.volume.iter().all(|v| *v < 1e-7));
    }

    #[test]
    fn optically_thin_and_thick_media_keep_their_correct_limits() {
        let tiny = evaluate([1e-8; 3], 1.0, [1e-8; 3]);
        assert!(tiny.volume.iter().all(|v| *v > 0.99999));
        let thick = evaluate([0.2, 0.3, 0.4], 1.0, [1e10; 3]);
        assert!(thick.volume.iter().all(|v| *v > 0.99999));
        let opaque = evaluate([1e20; 3], 1.0, [1e10; 3]);
        assert!(opaque.volume.iter().all(|v| *v < 1e-8));
    }

    #[test]
    fn thin_series_switch_and_extreme_finite_depths_remain_stable() {
        for water in [0.0_f32, 0.049_99, 0.05, 0.050_01, 1e20, 1e38] {
            for dye in [1e-8_f32, 0.01, 0.049_99, 0.05, 0.050_01, 1e20, 1e38] {
                let actual = escape_ratio(water, dye, (-water).exp(), (-dye).exp());
                let escape = |tau: f64| {
                    if tau == 0.0 {
                        1.0
                    } else {
                        -(-tau).exp_m1() / tau
                    }
                };
                let expected = escape(water as f64 + dye as f64) / escape(water as f64);
                assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
                assert!(
                    (actual as f64 - expected).abs() < 1e-6,
                    "water={water}, dye={dye}, actual={actual}, expected={expected}"
                );
            }
        }
    }
}
