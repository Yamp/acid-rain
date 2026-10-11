use std::f32::consts::{
    SQRT_2,
    TAU,
};

use once_cell::sync::Lazy;

use crate::physics::{
    PATCH_LENGTH,
    angular_frequency,
    damping_rate,
    impulse_response,
};

const RADIAL_SAMPLES: usize = 768;
const SPECTRAL_BINS: usize = 384;
const MAX_K: f32 = 192.0;
const DK: f32 = MAX_K / SPECTRAL_BINS as f32;

/// Instantaneous Gaussian pressure impulse on a locally planar water surface.
/// The integral of pressure over area and time is the sphere's normal momentum.
/// This is a linear free-surface closure, not a resolved crown/splash simulation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PressureImpact {
    /// J / rho, with units m^4/s. The sphere and water have equal density.
    momentum_over_density: f32,
    sigma: f32,
    depth: f32,
}

impl PressureImpact {
    pub fn new(
        radius: f32,
        normal_speed: f32,
        depth: f32,
    ) -> Option<Self> {
        if ![radius, normal_speed, depth]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            return None;
        }
        let momentum_over_density = (2.0 * TAU / 3.0) * radius.powi(3) * normal_speed;
        let sigma = 1.5 * radius;
        if !momentum_over_density.is_finite() || !sigma.is_finite() {
            return None;
        }
        Some(Self {
            momentum_over_density,
            // A finite contact footprint prevents the point-impulse ultraviolet
            // energy divergence. This empirical width transfers 8.75% of a
            // sphere's incoming kinetic energy to ideal deep-water waves.
            sigma,
            depth,
        })
    }

    fn coefficients(
        &self,
        age: f32,
        lifetime: f32,
        width: usize,
        height: usize,
    ) -> Option<[f32; SPECTRAL_BINS]> {
        if !age.is_finite()
            || !lifetime.is_finite()
            || age <= 0.0
            || age >= lifetime
            || lifetime <= 0.0
            || width.min(height) < 2
        {
            return None;
        }
        let fade = ((age / lifetime - 0.8) / 0.2).clamp(0.0, 1.0);
        let retirement = 1.0 - fade * fade * (3.0 - 2.0 * fade);
        let resolution = width.min(height) as f32;
        Some(std::array::from_fn(|mode| {
            let k = (mode as f32 + 0.5) * DK;
            let omega = angular_frequency(k, self.depth);
            let gamma = damping_rate(k);
            let frequency = (omega * omega - gamma * gamma).max(0.0).sqrt();
            let response = if frequency > f32::EPSILON {
                (-gamma * age).exp() * (frequency * age).sin() / frequency
            } else {
                0.0
            };
            // eta_hat = -(J/rho) k tanh(kh) exp(-sigma² k²/2) G(k,t).
            // Inverse 2D Fourier/Hankel measure contributes k dk / (2 pi).
            -self.momentum_over_density
                * k
                * k
                * (k * self.depth).tanh()
                * (-0.5 * self.sigma * self.sigma * k * k).exp()
                * response
                * (DK / TAU)
                * spectral_weight(k * PATCH_LENGTH / (TAU * resolution))
                * retirement
        }))
    }
}

/// Leave resolved wavelengths intact, then smoothly remove carriers with fewer
/// than about three samples per cycle. The margin below Nyquist also protects
/// surface derivatives and the sidebands introduced by each wave's envelope.
pub(crate) fn spectral_weight(cycles_per_sample: f32) -> f32 {
    if !cycles_per_sample.is_finite() {
        return 0.0;
    }
    let fade = ((cycles_per_sample.abs() - 0.15) / 0.20).clamp(0.0, 1.0);
    1.0 - fade * fade * (3.0 - 2.0 * fade)
}


/// J0: its convergent power series near the origin and DLMF 10.17
/// asymptotics beyond 16. Calculated once for a shared radial basis, in f64.
fn bessel_j0(x: f64) -> f64 {
    let x = x.abs();
    if x < 16.0 {
        let mut term = 1.0;
        let mut sum = 1.0;
        for n in 1..80 {
            term *= -x * x / (4.0 * (n * n) as f64);
            sum += term;
            if term.abs() < 1e-16 {
                break;
            }
        }
        sum
    } else {
        let mut coefficient = 1.0;
        let mut p = 1.0;
        let mut q = 0.0;
        for n in 1..=16 {
            coefficient *= -((2 * n - 1) * (2 * n - 1)) as f64 / (8.0 * n as f64 * x);
            if n % 2 == 0 {
                p += if (n / 2) % 2 == 0 {
                    coefficient
                } else {
                    -coefficient
                };
            } else {
                q += if (n / 2) % 2 == 0 {
                    coefficient
                } else {
                    -coefficient
                };
            }
        }
        let (s, c) = (x - std::f64::consts::FRAC_PI_4).sin_cos();
        (2.0 / (std::f64::consts::PI * x)).sqrt() * (c * p - s * q)
    }
}

/// Radial eigenfunctions solve the horizontal Laplacian exactly. The shared
/// matrix is independent of drops, grid resolution and time.
static RADIAL_BASIS: Lazy<Vec<[f32; SPECTRAL_BINS]>> = Lazy::new(|| {
    (0..=RADIAL_SAMPLES)
        .map(|index| {
            let r = index as f64 / RADIAL_SAMPLES as f64 * SQRT_2 as f64 * PATCH_LENGTH as f64;
            std::array::from_fn(|mode| bessel_j0((mode as f64 + 0.5) * DK as f64 * r) as f32)
        })
        .collect()
});

fn coefficients(
    strength: f32,
    size: f32,
    age: f32,
    lifetime: f32,
    width: usize,
    height: usize,
) -> Option<[f32; SPECTRAL_BINS]> {
    if !strength.is_finite()
        || !size.is_finite()
        || !age.is_finite()
        || !lifetime.is_finite()
        || strength <= 0.0
        || size <= 0.0
        || age <= 0.0
        || age >= lifetime
        || lifetime <= 0.0
        || width.min(height) < 2
    {
        return None;
    }
    // A compensated Gaussian momentum impulse: depression + displaced annulus,
    // zero k=0 component (no injection of water volume). Size changes the actual
    // footprint independently of strength; the unit size preserves prior waves.
    let sigma = (0.035 + 0.045 * strength.min(1.0).sqrt()) * size.clamp(0.25, 3.0);
    let sigma2 = sigma * sigma;
    // Retirement is a numerical window, separate from physical viscous decay.
    let fade = ((age / lifetime - 0.8) / 0.2).clamp(0.0, 1.0);
    let retirement = 1.0 - fade * fade * (3.0 - 2.0 * fade);
    let resolution = width.min(height).max(1) as f32;
    let output = std::array::from_fn(|mode| {
        let k = (mode as f32 + 0.5) * DK;
        let filter = spectral_weight(k * PATCH_LENGTH / (TAU * resolution));
        let momentum_spectrum = -0.45
            * strength.min(1.0)
            * 0.5
            * sigma2
            * sigma2
            * k
            * k
            * k
            * (-0.5 * sigma2 * k * k).exp()
            * DK;
        momentum_spectrum * impulse_response(k, age).0 * filter * retirement
    });
    output
        .iter()
        .any(|value: &f32| value.abs() > 1e-12)
        .then_some(output)
}

/// Spectral Green response to an impact, in meters. Sampling coordinates remain
/// normalized to the square patch. Phase and group propagation both follow the
/// dispersion relation; there is no independent carrier/envelope speed.
pub(crate) struct RadialWave {
    pub radius: f32,
    samples: [f32; RADIAL_SAMPLES + 1],
}

impl RadialWave {
    pub fn for_grid(
        strength: f32,
        age: f32,
        lifetime: f32,
        width: usize,
        height: usize,
    ) -> Option<Self> {
        Self::for_grid_sized(strength, 1.0, age, lifetime, width, height)
    }

    /// `size` scales the impact diameter, independently of momentum strength.
    /// Finite positive sizes are bounded to 0.25..=3; invalid sizes are rejected.
    pub fn for_grid_sized(
        strength: f32,
        size: f32,
        age: f32,
        lifetime: f32,
        width: usize,
        height: usize,
    ) -> Option<Self> {
        let weights = coefficients(strength, size, age, lifetime, width, height)?;
        Self::from_coefficients(weights)
    }

    pub fn for_pressure_impact(
        impact: PressureImpact,
        age: f32,
        lifetime: f32,
        width: usize,
        height: usize,
    ) -> Option<Self> {
        Self::from_coefficients(impact.coefficients(age, lifetime, width, height)?)
    }

    fn from_coefficients(weights: [f32; SPECTRAL_BINS]) -> Option<Self> {
        // Band-limited coefficients above this mode are exactly zero. Four
        // independent accumulators let native SIMD evaluate the retained sum;
        // no mode, physical term or sampling resolution is approximated away.
        let active = weights.iter().rposition(|value| *value != 0.0)? + 1;
        let groups = active / 4;
        let mut samples = [0.0; RADIAL_SAMPLES + 1];
        for (sample, basis) in samples.iter_mut().zip(RADIAL_BASIS.iter()) {
            let mut partial = [0.0; 4];
            for group in 0..groups {
                let offset = group * 4;
                for lane in 0..4 {
                    partial[lane] += weights[offset + lane] * basis[offset + lane];
                }
            }
            let mut total = partial.iter().sum::<f32>();
            for mode in groups * 4..active {
                total += weights[mode] * basis[mode];
            }
            *sample = total;
        }
        Some(Self {
            radius: SQRT_2,
            samples,
        })
    }

    #[inline]
    pub fn sample(
        &self,
        radius: f32,
    ) -> f32 {
        if !radius.is_finite() || radius < 0.0 || radius > self.radius {
            return 0.0;
        }
        let index = radius * (RADIAL_SAMPLES as f32 / SQRT_2);
        let lo = (index as usize).min(RADIAL_SAMPLES - 1);
        let blend = index - lo as f32;
        self.samples[lo] + (self.samples[lo + 1] - self.samples[lo]) * blend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_pressure_impulse_has_the_sphere_momentum_and_bounded_energy() {
        let rho = f64::from(crate::physics::DENSITY);
        for radius in [0.001375_f32, 0.00825, 0.0165] {
            let speed = 2.9_f32;
            let source = PressureImpact::new(radius, speed, 0.7).unwrap();
            let mass = 4.0 * std::f64::consts::PI / 3.0 * rho * f64::from(radius).powi(3);
            let momentum = f64::from(source.momentum_over_density) * rho;
            assert!((momentum / (mass * f64::from(speed)) - 1.0).abs() < 2e-7);
            let sigma = f64::from(source.sigma);
            // Independent kinetic-energy integral of phi_hat=-j_hat/rho:
            // E=rho/2 int(phi |D| phi) dA = J²/(4 pi rho) int(k² e^-sigma²k²) dk.
            let dk = 12.0 / sigma / 20_000.0;
            let integral = (0..20_000)
                .map(|i| {
                    let k = (i as f64 + 0.5) * dk;
                    k * k * (-sigma * sigma * k * k).exp() * dk
                })
                .sum::<f64>();
            let numeric = momentum * momentum / (4.0 * std::f64::consts::PI * rho) * integral;
            let exact =
                momentum * momentum / (16.0 * std::f64::consts::PI.sqrt() * rho * sigma.powi(3));
            assert!((numeric / exact - 1.0).abs() < 1e-10);
            let fraction = numeric / (0.5 * mass * f64::from(speed).powi(2));
            assert!(
                (fraction - std::f64::consts::PI.sqrt() / (6.0 * 1.5_f64.powi(3))).abs() < 4e-8
            );
            assert!(fraction < 0.09);
        }
    }

    #[test]
    fn physical_pressure_impulse_scales_with_volume_and_impact_velocity() {
        let a = PressureImpact::new(0.00825, 2.9, 0.7).unwrap();
        let speed = PressureImpact::new(0.00825, 5.8, 0.7).unwrap();
        let large = PressureImpact::new(0.0165, 2.9, 0.7).unwrap();
        assert_eq!(large.momentum_over_density, a.momentum_over_density * 8.0);
        assert_eq!(large.sigma, a.sigma * 2.0);
        let wa = RadialWave::for_pressure_impact(a, 0.04, 10.0, 192, 128).unwrap();
        let wb = RadialWave::for_pressure_impact(speed, 0.04, 10.0, 192, 128).unwrap();
        assert!(wa.sample(0.0) < -0.001);
        for r in [0.0, 0.008, 0.02, 0.05, 0.12, 0.4] {
            assert_eq!(wb.sample(r), 2.0 * wa.sample(r));
        }
        assert!(RadialWave::for_pressure_impact(a, 0.0, 10.0, 192, 128).is_none());
        assert!(
            RadialWave::for_pressure_impact(a, 0.0001, 10.0, 192, 128)
                .unwrap()
                .sample(0.0)
                < 0.0
        );
    }

    #[test]
    fn physical_pressure_hankel_response_matches_fine_independent_quadrature() {
        let radius = 0.00825_f64;
        let speed = 2.9_f64;
        let sigma = 1.5 * radius;
        let momentum_over_density = 4.0 * std::f64::consts::PI / 3.0 * radius.powi(3) * speed;
        for depth in [0.02_f32, 0.7] {
            let source = PressureImpact::new(radius as f32, speed as f32, depth).unwrap();
            for age in [0.003, 0.04, 0.15, 0.5] {
                let wave = RadialWave::for_pressure_impact(source, age, 10.0, 192, 128).unwrap();
                let dk = f64::from(MAX_K) / 4096.0;
                for radius_m in [0.0_f64, 0.027, 0.11, 0.4] {
                    let exact = (0..4096)
                        .map(|i| {
                            let k = (i as f64 + 0.5) * dk;
                            let gamma =
                                2.0 * f64::from(crate::physics::KINEMATIC_VISCOSITY) * k * k;
                            let omega2 = (f64::from(crate::physics::GRAVITY) * k
                                + f64::from(
                                    crate::physics::SURFACE_TENSION / crate::physics::DENSITY,
                                ) * k.powi(3))
                                * (k * f64::from(depth)).tanh();
                            let frequency = (omega2 - gamma * gamma).sqrt();
                            let response = (-gamma * f64::from(age)).exp()
                                * (frequency * f64::from(age)).sin()
                                / frequency;
                            -momentum_over_density
                                * k
                                * k
                                * (k * f64::from(depth)).tanh()
                                * (-0.5 * sigma * sigma * k * k).exp()
                                * response
                                * f64::from(spectral_weight(
                                    (k * 2.0 / (std::f64::consts::TAU * 128.0)) as f32,
                                ))
                                * bessel_j0(k * radius_m)
                                * dk
                                / std::f64::consts::TAU
                        })
                        .sum::<f64>();
                    assert!(
                        (f64::from(wave.sample(radius_m as f32 / PATCH_LENGTH)) - exact).abs()
                            < 0.00004,
                        "depth={depth} age={age} r={radius_m}"
                    );
                }
            }
        }
    }

    #[test]
    fn physical_pressure_redistributes_surface_volume_with_an_outward_ring() {
        let source = PressureImpact::new(0.00825, 2.9, 0.7).unwrap();
        for age in [0.01, 0.04, 0.10] {
            let wave = RadialWave::for_pressure_impact(source, age, 10.0, 192, 128).unwrap();
            let dr = f64::from(SQRT_2 * PATCH_LENGTH) / 20_000.0;
            let (volume, absolute) = (0..20_000).fold((0.0_f64, 0.0_f64), |(sum, abs), i| {
                let r = (i as f64 + 0.5) * dr;
                let v = f64::from(wave.sample(r as f32 / PATCH_LENGTH))
                    * std::f64::consts::TAU
                    * r
                    * dr;
                (sum + v, abs + v.abs())
            });
            // The unbounded transform has an exact zero mode. A finite radial
            // table with linear reconstruction has O(dr²) integration error;
            // constrain its volume residual below 0.5% of displaced volume.
            assert!(
                volume.abs() < absolute * 0.005,
                "age={age}: {volume}/{absolute}"
            );
            assert!((0..100).any(|i| wave.sample(i as f32 * 0.001) > 0.00001));
        }
    }

    #[test]
    #[ignore]
    fn physical_impact_builder_benchmark() {
        use std::{
            hint::black_box,
            time::Instant,
        };
        let source = PressureImpact::new(0.00825, 2.9, 0.7).unwrap();
        black_box(RadialWave::for_grid_sized(0.2, 1.5, 0.1, 10.0, 192, 128));
        let mut rows = Vec::new();
        println!("model | sources | mean builder batch ms");
        for physical in [false, true] {
            let start = Instant::now();
            for _ in 0..50 {
                for i in 0..128 {
                    let age = 0.01 + i as f32 * 0.025;
                    black_box(if physical {
                        RadialWave::for_pressure_impact(source, age, 10.0, 192, 128)
                    } else {
                        RadialWave::for_grid_sized(0.2, 1.5, age, 10.0, 192, 128)
                    });
                }
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 50.0;
            let model = if physical { "physical" } else { "legacy" };
            println!("{model} | 128 | {ms:.3}");
            rows.push(format!(
                "{{\"model\":\"{model}\",\"sources\":128,\"mean_ms\":{ms}}}"
            ));
        }
        let path = std::env::var("AR_IMPACT_BENCHMARK").expect("provide artifact output path");
        std::fs::write(path, format!("[{}]\n", rows.join(","))).unwrap();
    }

    #[test]
    fn larger_drops_widen_the_impulse_at_fixed_strength() {
        let first_zero = |size| {
            let wave = RadialWave::for_grid_sized(0.5, size, 0.001, 10.0, 480, 480).unwrap();
            assert!(wave.sample(0.0) < 0.0);
            let step = 0.0001;
            (1..10_000)
                .map(|index| index as f32 * step)
                .find(|radius| wave.sample(*radius) >= 0.0)
                .expect("impulse has a displaced annulus")
        };
        let small = first_zero(0.5);
        let medium = first_zero(1.0);
        let large = first_zero(2.0);
        assert!((medium / small - 2.0).abs() < 0.04, "{small} -> {medium}");
        assert!((large / medium - 2.0).abs() < 0.04, "{medium} -> {large}");
    }

    #[test]
    fn unit_drop_size_preserves_the_original_response_exactly() {
        for (strength, age, width, height) in [
            (0.1, 0.001, 240, 120),
            (0.5, 0.5, 200, 60),
            (1.0, 3.0, 80, 24),
        ] {
            let original = RadialWave::for_grid(strength, age, 10.0, width, height).unwrap();
            let sized =
                RadialWave::for_grid_sized(strength, 1.0, age, 10.0, width, height).unwrap();
            assert_eq!(sized.samples, original.samples);
        }
    }

    #[test]
    fn invalid_drop_sizes_are_rejected_and_finite_sizes_are_bounded() {
        for size in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
            assert!(RadialWave::for_grid_sized(0.5, size, 0.1, 10.0, 200, 60).is_none());
        }
        for (size, bound) in [(0.001, 0.25), (100.0, 3.0)] {
            let wave = RadialWave::for_grid_sized(0.5, size, 0.1, 10.0, 200, 60).unwrap();
            let bounded = RadialWave::for_grid_sized(0.5, bound, 0.1, 10.0, 200, 60).unwrap();
            assert_eq!(wave.samples, bounded.samples);
        }
    }

    #[test]
    fn a_pressure_impulse_initially_depresses_its_center() {
        let wave = RadialWave::for_grid(0.3, 0.001, 10.0, 240, 120).unwrap();
        assert!(wave.sample(0.0) < -0.00001);
    }

    #[test]
    fn bessel_basis_matches_independent_angular_quadrature() {
        for x in [
            0.0,
            0.1,
            2.4048255577,
            8.0,
            15.999,
            16.001,
            30.0,
            130.0,
            540.0,
        ] {
            let expected = (0..4096)
                .map(|i| {
                    let theta = (i as f64 + 0.5) * std::f64::consts::TAU / 4096.0;
                    (x * theta.cos()).cos()
                })
                .sum::<f64>()
                / 4096.0;
            assert!((bessel_j0(x) - expected).abs() < 2e-9, "x={x}");
        }
    }

    #[test]
    fn lookup_matches_direct_hankel_sum_between_texels() {
        for age in [0.001, 0.1, 0.6, 3.0, 9.9] {
            let weights = coefficients(0.3, 1.0, age, 10.0, 200, 60).unwrap();
            let wave = RadialWave::for_grid(0.3, age, 10.0, 200, 60).unwrap();
            for i in 0..600 {
                let r = (i as f32 + 0.37) / 600.0 * SQRT_2;
                let expected: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(mode, weight)| {
                        weight
                            * bessel_j0(
                                (mode as f64 + 0.5) * DK as f64 * r as f64 * PATCH_LENGTH as f64,
                            ) as f32
                    })
                    .sum();
                assert!(
                    (wave.sample(r) - expected).abs() < 2e-5,
                    "age={age}, radius={r}"
                );
            }
        }
    }

    #[test]
    fn spectral_quadrature_converges_against_eight_times_finer_integration() {
        let strength: f32 = 0.3;
        let sigma = 0.035 + 0.045 * strength.sqrt();
        let sigma2 = sigma * sigma;
        let step = MAX_K / (SPECTRAL_BINS * 8) as f32;
        for age in [0.01, 0.6, 3.0, 7.5] {
            let wave = RadialWave::for_grid(strength, age, 10.0, 240, 120).unwrap();
            for r in [0.0, 0.03, 0.13, 0.8, 1.3] {
                let mut expected = 0.0_f64;
                for mode in 0..SPECTRAL_BINS * 8 {
                    let k = (mode as f32 + 0.5) * step;
                    let density = -0.45
                        * strength
                        * 0.5
                        * sigma2
                        * sigma2
                        * k.powi(3)
                        * (-0.5 * sigma2 * k * k).exp();
                    let filtered = spectral_weight(k * PATCH_LENGTH / (TAU * 120.0));
                    expected += (density * impulse_response(k, age).0 * filtered * step) as f64
                        * bessel_j0((k * r * PATCH_LENGTH) as f64);
                }
                assert!(
                    (wave.sample(r) as f64 - expected).abs() < 2e-5,
                    "age={age}, r={r}"
                );
            }
        }
    }

    #[test]
    fn resolution_filter_removes_undersampled_carriers_smoothly() {
        assert_eq!(spectral_weight(3.0 / 40.0), 1.0);
        for i in 0..1000 {
            let f = i as f32 / 1000.0;
            let w = spectral_weight(f);
            assert!((0.0..=1.0).contains(&w));
            assert!(spectral_weight(f + 0.001) <= w);
            assert!((w - spectral_weight(f + 0.001)).abs() < 0.008);
            if f >= 0.5 {
                assert_eq!(w, 0.0);
            }
        }
    }

    #[test]
    fn response_is_continuous_at_birth_and_retirement() {
        assert!(RadialWave::for_grid(0.3, 0.0, 10.0, 200, 60).is_none());
        let born = RadialWave::for_grid(0.3, 0.0001, 10.0, 200, 60).unwrap();
        assert!(born.samples.iter().all(|height| height.abs() < 0.0001));
        let retired = RadialWave::for_grid(0.3, 9.999, 10.0, 200, 60).unwrap();
        assert!(retired.samples.iter().all(|height| height.abs() < 1e-7));
        assert!(RadialWave::for_grid(0.3, 10.0, 10.0, 200, 60).is_none());
        for age in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(RadialWave::for_grid(0.3, age, 10.0, 200, 60).is_none());
        }
    }

    #[test]
    fn unresolved_modes_are_absent_on_a_single_cell() {
        assert!(RadialWave::for_grid(0.3, 1.0, 10.0, 1, 1).is_none());
    }
}
