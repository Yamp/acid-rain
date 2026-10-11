//! Bounded world-space current and wave sources. Only feature cells overlapping
//! the 2 m simulation window are materialized; the hash fixes their identity.
//! Compact vortices use a divergence-free streamfunction and their own steady
//! radial pressure balance. Superposition, finite-residence surface distortion
//! and localized standing-wave forcing are small-amplitude closures, not a
//! solved three-dimensional turbulent free surface.
use crate::{
    physics::{
        DEPTH,
        GRAVITY,
        PATCH_LENGTH,
        angular_frequency,
    },
    vortex::VortexSample,
    waves::spectral_weight,
};

pub(crate) const CURRENT: [f32; 2] = [0.09, -0.055];
const CELL: f64 = 0.9;
const MAX_RADIUS: f64 = 0.64;
const RESIDENCE: f32 = 1.25;
const RADIAL: usize = 64;

#[derive(Debug)]
struct Feature {
    center: [f32; 2],
    radius: f32,
    inverse_radius2: f32,
    omega: f32,
    rotation: [[f32; 2]; RADIAL + 1],
    packet: [f32; 2],
    phase: f32,
    initial_phase: f32,
    frequency: f32,
    amplitude: f32,
}

#[derive(Debug)]
pub(crate) struct WorldFlow {
    features: Vec<Feature>,
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn unit(bits: u64) -> f32 { (bits >> 40) as f32 / 16_777_216.0 }

impl WorldFlow {
    pub fn new(
        origin: [f64; 2],
        seed: u64,
        elapsed: f64,
        size: [usize; 2],
    ) -> Self {
        let mut features = Vec::with_capacity(25);
        let minimum = origin.map(|p| ((p - MAX_RADIUS) / CELL).floor() as i64);
        let maximum =
            origin.map(|p| ((p + PATCH_LENGTH as f64 + MAX_RADIUS) / CELL).floor() as i64);
        let time = if elapsed.is_finite() { elapsed } else { 0.0 };
        for iy in minimum[1]..=maximum[1] {
            for ix in minimum[0]..=maximum[0] {
                let hash = mix(seed
                    ^ mix(ix as u64)
                    ^ mix((iy as u64).wrapping_add(0x9e37_79b9_7f4a_7c15)));
                let center = [
                    (ix as f64 * CELL + (0.12 + 0.76 * unit(hash)) as f64 * CELL - origin[0])
                        as f32,
                    (iy as f64 * CELL + (0.12 + 0.76 * unit(mix(hash))) as f64 * CELL - origin[1])
                        as f32,
                ];
                let radius = 0.40 + 0.24 * unit(mix(hash.wrapping_add(1)));
                if center
                    .iter()
                    .any(|p| *p + radius < 0.0 || *p - radius > PATCH_LENGTH)
                {
                    continue;
                }
                let phase = unit(mix(hash.wrapping_add(2))) * std::f32::consts::TAU;
                let omega = (1.3 + 1.0 * unit(mix(hash.wrapping_add(3))))
                    * if hash & 1 == 0 { -1.0 } else { 1.0 };
                let omega = if hash.is_multiple_of(5) { 0.0 } else { omega };
                let rotation = std::array::from_fn(|i| {
                    let q = i as f32 / RADIAL as f32;
                    let (sine, cosine) = (-omega * q * q * RESIDENCE).sin_cos();
                    [cosine, sine]
                });
                let wavelength = 0.18 + 0.28 * unit(mix(hash.wrapping_add(4)));
                let k = std::f32::consts::TAU / wavelength;
                let (sine, cosine) = phase.sin_cos();
                let packet = [k * cosine, k * sine];
                let frequency = angular_frequency(k, DEPTH);
                let resolved = spectral_weight(
                    (packet[0].abs() * PATCH_LENGTH / size[0].max(1) as f32)
                        .max(packet[1].abs() * PATCH_LENGTH / size[1].max(1) as f32)
                        / std::f32::consts::TAU,
                );
                features.push(Feature {
                    center,
                    radius,
                    inverse_radius2: 1.0 / (radius * radius),
                    omega,
                    rotation,
                    packet,
                    phase: (phase as f64 - time * frequency as f64)
                        .rem_euclid(std::f64::consts::TAU) as f32,
                    initial_phase: phase,
                    frequency,
                    amplitude: (if omega == 0.0 { 0.008 } else { 0.003 }) * resolved,
                });
            }
        }
        Self { features }
    }

    /// Geometry, world hashes and the angular reconstruction table are reused;
    /// only one carrier phase per feature changes with physical time.
    pub fn set_time(
        &mut self,
        elapsed: f64,
    ) {
        if !elapsed.is_finite() {
            return;
        }
        for feature in &mut self.features {
            feature.phase = (feature.initial_phase as f64 - elapsed * feature.frequency as f64)
                .rem_euclid(std::f64::consts::TAU) as f32;
        }
    }

    /// Local psi [m²/s], without uniform through-flow. Its face curl is shared
    /// by pigment; compact support avoids periodic copies at window boundaries.
    pub fn streamfunction(
        &self,
        point: [f32; 2],
    ) -> f32 {
        let point = point.map(|p| p * PATCH_LENGTH);
        self.features
            .iter()
            .map(|feature| {
                let d = [point[0] - feature.center[0], point[1] - feature.center[1]];
                let q = (1.0 - (d[0] * d[0] + d[1] * d[1]) * feature.inverse_radius2).max(0.0);
                -feature.omega * feature.radius * feature.radius / 6.0 * q * q * q
            })
            .sum()
    }

    pub fn velocity(
        &self,
        point: [f32; 2],
    ) -> [f32; 2] {
        let point = point.map(|p| p * PATCH_LENGTH);
        let mut velocity = [0.0; 2];
        for feature in &self.features {
            let dx = point[0] - feature.center[0];
            let dy = point[1] - feature.center[1];
            let q = (1.0 - (dx * dx + dy * dy) * feature.inverse_radius2).max(0.0);
            let omega = feature.omega * q * q;
            velocity[0] -= omega * dy;
            velocity[1] += omega * dx;
        }
        velocity
    }

    pub fn sample(
        &self,
        point: [f32; 2],
    ) -> VortexSample {
        let physical = point.map(|p| p * PATCH_LENGTH);
        let mut result = VortexSample {
            source: point,
            elevation: 0.0,
            stretch: 1.0,
        };
        for feature in &self.features {
            let dx = physical[0] - feature.center[0];
            let dy = physical[1] - feature.center[1];
            let r2 = (dx * dx + dy * dy) * feature.inverse_radius2;
            if r2 >= 1.0 {
                continue;
            }
            let q = 1.0 - r2;
            let index = q * RADIAL as f32;
            let lo = (index as usize).min(RADIAL - 1);
            let t = index - lo as f32;
            let rotation: [f32; 2] = std::array::from_fn(|axis| {
                feature.rotation[lo][axis]
                    + (feature.rotation[lo + 1][axis] - feature.rotation[lo][axis]) * t
            });
            result.source[0] += (dx * (rotation[0] - 1.0) - dy * rotation[1]) / PATCH_LENGTH;
            result.source[1] += (dy * (rotation[0] - 1.0) + dx * rotation[1]) / PATCH_LENGTH;
            // dh/dr = (omega*r*q²)²/(g*r); h(R)=0. The pressure bowl
            // is tied to this feature's actual finite-core tangential current.
            result.elevation -= feature.omega * feature.omega * feature.radius * feature.radius
                / (10.0 * GRAVITY)
                * q.powi(5);
            result.elevation += feature.amplitude
                * q
                * q
                * q
                * (feature.packet[0] * dx + feature.packet[1] * dy + feature.phase).sin();
            result.stretch += 4.0 * feature.omega.abs() * RESIDENCE * r2 * q;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PATCH_LENGTH;

    #[test]
    fn world_features_are_bounded_and_do_not_repeat_per_patch() {
        for origin in [[-1.0; 2], [1500.0, -2700.0], [-0.1, 0.7]] {
            let field = WorldFlow::new(origin, 7, 0.0, [96, 64]);
            assert!(field.features.len() <= 25);
            for x in 0..10 {
                for y in 0..10 {
                    let sample = field.sample([x as f32 / 9.0, y as f32 / 9.0]);
                    assert!(sample.elevation.is_finite() && sample.elevation.abs() < 0.1);
                    assert!(sample.source.iter().all(|s| s.is_finite()));
                }
            }
        }
        let a = WorldFlow::new([-1.0; 2], 7, 0.0, [96, 64]);
        let b = WorldFlow::new([1.0, -1.0], 7, 0.0, [96, 64]);
        assert_ne!(a.velocity([0.4, 0.6]), b.velocity([0.4, 0.6]));
    }

    #[test]
    fn world_flow_is_spatially_varied_and_seeded() {
        let a = WorldFlow::new([-1.0; 2], 7, 3.0, [96, 64]);
        let b = WorldFlow::new([-1.0; 2], 11, 3.0, [96, 64]);
        let speed: f32 = (0..20)
            .map(|i| {
                let point = [i as f32 / 20.0, 0.53];
                let v = a.velocity(point);
                v[0].hypot(v[1])
            })
            .sum();
        assert!(speed > 0.1, "local currents must actually circulate");
        assert_ne!(
            a.sample([0.45, 0.57]).elevation,
            b.sample([0.45, 0.57]).elevation
        );
    }

    #[test]
    fn shifted_windows_agree_at_the_same_world_point() {
        let a = WorldFlow::new([-1.0; 2], 7, 3.0, [96, 64]);
        let b = WorldFlow::new([-0.5, -1.25], 7, 3.0, [96, 64]);
        for y in 2..15 {
            for x in 5..17 {
                let p = [x as f32 / 20.0, y as f32 / 20.0];
                let q = [p[0] - 0.25, p[1] + 0.125];
                let av = a.velocity(p);
                let bv = b.velocity(q);
                let ashape = a.sample(p);
                let bshape = b.sample(q);
                for axis in 0..2 {
                    assert!((av[axis] - bv[axis]).abs() < 2e-6);
                    let displacement =
                        (ashape.source[axis] - p[axis]) - (bshape.source[axis] - q[axis]);
                    assert!(displacement.abs() < 2e-6);
                }
                assert!((ashape.elevation - bshape.elevation).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn local_velocity_matches_streamfunction_curl() {
        let flow = WorldFlow::new([-1.0; 2], 7, 0.0, [96, 64]);
        for p in [[0.2, 0.3], [0.6, 0.7], [0.5, 0.5]] {
            let e = 0.0001;
            let dx = (flow.streamfunction([p[0] + e, p[1]])
                - flow.streamfunction([p[0] - e, p[1]]))
                / (2.0 * e * PATCH_LENGTH);
            let dy = (flow.streamfunction([p[0], p[1] + e])
                - flow.streamfunction([p[0], p[1] - e]))
                / (2.0 * e * PATCH_LENGTH);
            let v = flow.velocity(p);
            assert!((v[0] + dy).abs() < 0.0002);
            assert!((v[1] - dx).abs() < 0.0002);
        }
    }
}
