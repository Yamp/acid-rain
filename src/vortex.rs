//! Finite-core vortices deform the actual surface, not its color. The steady
//! radial pressure balance dh/dr = v_theta^2/(g*r) produces a smooth bowl.
//! A finite-residence backtrace and diffused two-arm perturbation prevent the
//! indefinitely tightening windings of an unbounded material-history model.

use crate::{
    physics::{
        GRAVITY,
        PATCH_LENGTH,
    },
    waves::spectral_weight,
};

const RADIAL_STEPS: usize = 256;
const RESIDENCE: f32 = 1.8;
const DIFFUSIVITY: f32 = 0.00008;

#[derive(Clone, Copy, Debug)]
pub(crate) struct VortexSample {
    pub source: [f32; 2],
    pub elevation: f32,
    pub stretch: f32,
}

struct Vortex {
    center: [f32; 2],
    core2: f32,
    omega: f32,
    depth: f32,
    // Backtrace cos/sin, two diffused spiral coefficients, local shear bound.
    radial: [[f32; 5]; RADIAL_STEPS + 1],
}

impl Vortex {
    fn new(
        center: [f32; 2],
        core: f32,
        omega: f32,
        phase: f32,
        samples: f32,
    ) -> Self {
        let core2 = core * core;
        let mut vortex = Self {
            center,
            core2,
            omega,
            depth: omega * omega * core2 * PATCH_LENGTH * PATCH_LENGTH / (2.0 * GRAVITY),
            radial: [[0.0; 5]; RADIAL_STEPS + 1],
        };
        for (index, row) in vortex.radial.iter_mut().enumerate() {
            let q = index as f32 / RADIAL_STEPS as f32;
            let angle = -omega * q * RESIDENCE;
            let (sine, cosine) = angle.sin_cos();
            let shear = 2.0 * omega.abs() * RESIDENCE * q * (1.0 - q);
            // A regularized angular seed vanishes quadratically at the core.
            // Bound its angular + radial phase gradients before grid sampling.
            let wavenumber = 2.0 / core
                * (q.sqrt() + 2.0 * RESIDENCE * omega.abs() * q * (q * (1.0 - q)).sqrt());
            let diffusion = (-DIFFUSIVITY * RESIDENCE * wavenumber * wavenumber).exp();
            let resolved = spectral_weight(wavenumber / std::f32::consts::TAU / samples);
            let amplitude = 0.085 * q * (1.0 - q) * diffusion * resolved;
            let (spiral_sine, spiral_cosine) = (2.0 * angle - phase).sin_cos();
            *row = [
                cosine,
                sine,
                amplitude * spiral_cosine,
                -amplitude * spiral_sine,
                shear,
            ];
        }
        vortex
    }

    #[inline]
    fn radial_sample(
        &self,
        q: f32,
    ) -> [f32; 5] {
        let index = q.clamp(0.0, 1.0) * RADIAL_STEPS as f32;
        let lo = (index as usize).min(RADIAL_STEPS - 1);
        let blend = index - lo as f32;
        let mut result = [0.0; 5];
        for (channel, value) in result.iter_mut().enumerate() {
            *value = self.radial[lo][channel]
                + (self.radial[lo + 1][channel] - self.radial[lo][channel]) * blend;
        }
        result
    }

    #[inline]
    fn sample(
        &self,
        point: [f32; 2],
    ) -> VortexSample {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let inv_radius2 = (dx * dx + dy * dy + self.core2).recip();
        let q = self.core2 * inv_radius2;
        let radial = self.radial_sample(q);
        // cos(2 phi), sin(2 phi) with a finite core: no atan2 seam or 1/r pole.
        let seed_cosine = (dx * dx - dy * dy) * inv_radius2;
        let seed_sine = 2.0 * dx * dy * inv_radius2;
        VortexSample {
            source: [
                self.center[0] + dx * radial[0] - dy * radial[1],
                self.center[1] + dx * radial[1] + dy * radial[0],
            ],
            elevation: -self.depth * q + seed_cosine * radial[2] + seed_sine * radial[3],
            stretch: 1.0 + radial[4],
        }
    }

    fn velocity(
        &self,
        point: [f32; 2],
    ) -> [f32; 2] {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let omega = self.omega * self.core2 / (dx * dx + dy * dy + self.core2);
        [-dy * omega, dx * omega]
    }
}

pub(crate) struct VortexFrame {
    vortices: [Vortex; 2],
}

impl VortexFrame {
    /// Periodic pigment-current streamfunction in m²/s; normalized position.
    pub fn periodic_streamfunction(
        &self,
        point: [f32; 2],
    ) -> f32 {
        if !point.iter().all(|p| p.is_finite()) {
            return 0.0;
        }
        let pi = std::f32::consts::PI;
        self.vortices
            .iter()
            .map(|vortex| {
                // Chord distances are periodic and recover Euclidean distance near
                // each core. This changes the distant current, not the wave geometry.
                let sx = ((point[0] - vortex.center[0]) * pi).sin() / pi;
                let sy = ((point[1] - vortex.center[1]) * pi).sin() / pi;
                0.5 * vortex.omega
                    * vortex.core2
                    * PATCH_LENGTH
                    * PATCH_LENGTH
                    * (vortex.core2 + sx * sx + sy * sy).ln()
            })
            .sum()
    }

    /// Analytic velocity of the periodic current in m/s.
    pub fn periodic_velocity(
        &self,
        point: [f32; 2],
    ) -> [f32; 2] {
        if !point.iter().all(|p| p.is_finite()) {
            return [0.0; 2];
        }
        let pi = std::f32::consts::PI;
        let mut velocity = [0.0; 2];
        for vortex in &self.vortices {
            let x = (point[0] - vortex.center[0]) * pi;
            let y = (point[1] - vortex.center[1]) * pi;
            let sx = x.sin() / pi;
            let sy = y.sin() / pi;
            let factor = vortex.omega * vortex.core2 * PATCH_LENGTH
                / (2.0 * pi * (vortex.core2 + sx * sx + sy * sy));
            velocity[0] -= factor * (2.0 * y).sin();
            velocity[1] += factor * (2.0 * x).sin();
        }
        velocity
    }

    /// Prescribed incompressible horizontal velocity, in patch coordinates/s.
    pub fn velocity(
        &self,
        point: [f32; 2],
    ) -> [f32; 2] {
        let a = self.vortices[0].velocity(point);
        let b = self.vortices[1].velocity(point);
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Bounded RK4 trajectory in locally frozen flow. Fresh impacts start at
    /// their actual location, rather than being relocated by a texture warp.
    pub fn transport(
        &self,
        mut point: [f32; 2],
        age: f32,
    ) -> [f32; 2] {
        let steps = (age / 0.08).ceil().clamp(1.0, 128.0) as usize;
        let dt = age / steps as f32;
        for _ in 0..steps {
            let a = self.velocity(point);
            let b = self.velocity([point[0] + a[0] * dt * 0.5, point[1] + a[1] * dt * 0.5]);
            let c = self.velocity([point[0] + b[0] * dt * 0.5, point[1] + b[1] * dt * 0.5]);
            let d = self.velocity([point[0] + c[0] * dt, point[1] + c[1] * dt]);
            for i in 0..2 {
                point[i] += dt / 6.0 * (a[i] + 2.0 * b[i] + 2.0 * c[i] + d[i]);
            }
        }
        point
    }

    pub fn new(
        elapsed: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let elapsed = if elapsed.is_finite() { elapsed } else { 0.0 };
        let sine = |rate: f64, offset: f64| {
            (elapsed * rate + offset)
                .rem_euclid(std::f64::consts::TAU)
                .sin() as f32
        };
        let samples = width.min(height).max(1) as f32;
        let phase = |omega: f64| (elapsed * omega * 0.52).rem_euclid(std::f64::consts::TAU) as f32;
        Self {
            vortices: [
                Vortex::new(
                    [
                        0.465 + 0.008 * sine(0.09, 0.0),
                        0.49 + 0.008 * sine(0.07, 0.0),
                    ],
                    0.115,
                    2.82,
                    phase(0.82),
                    samples,
                ),
                Vortex::new(
                    [
                        0.62 + 0.01 * sine(0.06, 1.0),
                        0.54 + 0.01 * sine(0.05, 1.57),
                    ],
                    0.14,
                    -1.93,
                    phase(-0.66),
                    samples,
                ),
            ],
        }
    }

    #[inline]
    pub fn sample(
        &self,
        point: [f32; 2],
    ) -> VortexSample {
        let first = self.vortices[0].sample(point);
        let second = self.vortices[1].sample(point);
        // Compose radial flow maps instead of adding offsets, preserving each
        // vortex's finite, smooth rotational map where the two fields overlap.
        let advected = self.vortices[1].sample(first.source);
        VortexSample {
            source: advected.source,
            elevation: first.elevation + second.elevation,
            stretch: first.stretch * advected.stretch,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_current_matches_its_streamfunction_and_both_patch_seams() {
        let field = VortexFrame::new(2.0, 96, 64);
        assert!(field.periodic_velocity([0.4, 0.55])[0].abs() > 0.01);
        for point in [[0.0, 0.25], [0.3, 0.0], [0.44, 0.52]] {
            let psi = field.periodic_streamfunction(point);
            let velocity = field.periodic_velocity(point);
            for axis in 0..2 {
                let mut shifted = point;
                shifted[axis] += 1.0;
                assert!((field.periodic_streamfunction(shifted) - psi).abs() < 1e-7);
                for (a, b) in field.periodic_velocity(shifted).iter().zip(velocity) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
            let e = 0.0001;
            let dx = (field.periodic_streamfunction([point[0] + e, point[1]])
                - field.periodic_streamfunction([point[0] - e, point[1]]))
                / (2.0 * e * PATCH_LENGTH);
            let dy = (field.periodic_streamfunction([point[0], point[1] + e])
                - field.periodic_streamfunction([point[0], point[1] - e]))
                / (2.0 * e * PATCH_LENGTH);
            assert!((velocity[0] + dy).abs() < 0.0005);
            assert!((velocity[1] - dx).abs() < 0.0005);
        }
    }

    #[test]
    fn periodic_current_retains_the_existing_finite_core_rotation() {
        let field = VortexFrame {
            vortices: [
                Vortex::new([0.5, 0.5], 0.12, 2.0, 0.0, 96.0),
                Vortex::new([0.5, 0.5], 0.12, 0.0, 0.0, 96.0),
            ],
        };
        let point = [0.5001, 0.5001];
        let expected = field.velocity(point).map(|v| v * PATCH_LENGTH);
        let actual = field.periodic_velocity(point);
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn flow_is_divergence_free_and_transport_starts_at_the_drop() {
        let field = VortexFrame::new(2.0, 120, 80);
        for point in [[0.465, 0.49], [0.4, 0.6], [0.55, 0.4]] {
            assert_eq!(field.transport(point, 0.0), point);
            let e = 0.0001;
            let xp = field.velocity([point[0] + e, point[1]]);
            let xm = field.velocity([point[0] - e, point[1]]);
            let yp = field.velocity([point[0], point[1] + e]);
            let ym = field.velocity([point[0], point[1] - e]);
            let divergence = (xp[0] - xm[0] + yp[1] - ym[1]) / (2.0 * e);
            assert!(divergence.abs() < 0.003, "divergence={divergence}");
            let short = field.transport(point, 0.001);
            let velocity = field.velocity(point);
            for i in 0..2 {
                assert!((short[i] - point[i] - velocity[i] * 0.001).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn opposite_circulations_have_opposite_flow_and_finite_stationary_cores() {
        for omega in [-0.82, 0.82] {
            let vortex = Vortex::new([0.5, 0.5], 0.16, omega, 0.0, 160.0);
            assert_eq!(vortex.velocity([0.5, 0.5]), [0.0, 0.0]);
            let velocity = vortex.velocity([0.66, 0.5]);
            assert_eq!(velocity[0], 0.0);
            assert_eq!(velocity[1].signum(), omega.signum());
            let source = vortex.sample([0.66, 0.5]).source;
            assert_eq!((source[1] - 0.5).signum(), -omega.signum());
            let radius = ((source[0] - 0.5).powi(2) + (source[1] - 0.5).powi(2)).sqrt();
            assert!((radius - 0.16).abs() < 0.00001);
        }
    }

    #[test]
    fn mean_pressure_bowl_balances_tangential_acceleration() {
        let vortex = Vortex::new([0.0, 0.0], 0.16, 0.82, 0.0, 160.0);
        let mean_height = |r: f32| {
            // The two-arm perturbation changes sign after a quarter turn.
            (vortex.sample([r, 0.0]).elevation + vortex.sample([0.0, r]).elevation) * 0.5
        };
        for radius in [0.001, 0.01, 0.07, 0.16, 0.5] {
            let epsilon = 0.0001;
            let slope =
                (mean_height(radius + epsilon) - mean_height(radius - epsilon)) / (2.0 * epsilon);
            let velocity = vortex.velocity([radius, 0.0]);
            let expected =
                velocity[1] * velocity[1] * PATCH_LENGTH * PATCH_LENGTH / (GRAVITY * radius);
            assert!(
                (slope - expected).abs() < 0.0001,
                "{radius}: {slope} vs {expected}"
            );
        }
    }

    #[test]
    fn vortices_have_finite_depressed_cores() {
        let field = VortexFrame::new(0.0, 160, 120);
        let core = field.sample([0.465, 0.49]);
        assert!(core.elevation < -0.005);
        assert!(core.elevation > -0.08);
        assert!(core.source.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn distortion_is_continuous_and_bounded_for_long_sessions() {
        for elapsed in [0.0, 3.0, 100.0, 1_000_000.0] {
            let field = VortexFrame::new(elapsed, 120, 40);
            for x in 0..21 {
                for y in 0..21 {
                    let point = [x as f32 / 20.0, y as f32 / 20.0];
                    let here = field.sample(point);
                    let near = field.sample([point[0] + 0.00001, point[1]]);
                    assert!(here.source.iter().all(|value| value.is_finite()));
                    assert!(here.elevation.is_finite() && here.elevation.abs() < 0.08);
                    assert!((here.elevation - near.elevation).abs() < 0.0001);
                    assert!((here.source[0] - near.source[0]).abs() < 0.0001);
                    assert!((here.source[1] - near.source[1]).abs() < 0.0001);
                }
            }
        }
    }

    #[test]
    fn fully_resolved_vortex_geometry_is_resize_independent() {
        let small = VortexFrame::new(7.0, 160, 160);
        let large = VortexFrame::new(7.0, 320, 320);
        for x in 0..20 {
            for y in 0..20 {
                let point = [x as f32 / 20.0, y as f32 / 20.0];
                let a = small.sample(point);
                let b = large.sample(point);
                assert!((a.elevation - b.elevation).abs() < 1e-6);
                assert_eq!(a.source, b.source);
            }
        }
    }
}
