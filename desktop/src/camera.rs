//! Native camera geometry and world-space presentation, without platform APIs.

use acid_rain_native_core::{
    ARUniforms,
    Float4,
};

pub const NOMINAL_HEIGHT_ENVELOPE: [f32; 2] = [-0.16, 0.12];
pub const TARGET_RESERVE: f32 = 0.04;

/// Fit a complete ray cone, independent of the current roll and orbit corner.
/// A fixed wave envelope and target reserve prevent rain and rebasing from
/// changing the lens. Extraordinary heights still invoke the safe fallback.
pub fn fill(
    uniforms: &mut ARUniforms,
    aspect: f32,
    height_envelope: Option<[f32; 2]>,
) {
    let target = [uniforms.camera.0[0], uniforms.camera.0[1], 0.0];
    let heading = uniforms.time.0[0] / 180.0 * (2.0 * std::f32::consts::PI);
    let requested_envelope = height_envelope.unwrap_or(NOMINAL_HEIGHT_ENVELOPE);
    let heights = [
        requested_envelope[0].min(uniforms.world.0[1] - 0.002),
        requested_envelope[1].max(uniforms.world.0[2] + 0.002),
    ];
    let offset = (target[0] - 1.0).hypot(target[1] - 1.0);
    let radius = (0.94 - TARGET_RESERVE.max(offset)).max(0.005);
    let required_elevation = (heights[0].abs().max(heights[1].abs()) / radius).atan() + 0.01;
    let elevation = uniforms.camera.0[3].max(required_elevation);
    let (s, c) = elevation.sin_cos();
    let distance = ((heights[1] + 0.08) / s).max(1.0);
    let eye = [
        target[0] - distance * c * heading.cos(),
        target[1] - distance * c * heading.sin(),
        distance * s,
    ];
    let forward = normalized(std::array::from_fn(|i| target[i] - eye[i]));
    let basis_right = normalized([forward[1], -forward[0], 0.0]);
    let basis_up = cross(basis_right, forward);
    let (sr, cr) = uniforms.camera.0[2].sin_cos();
    let right: [f32; 3] = std::array::from_fn(|i| basis_right[i] * cr + basis_up[i] * sr);
    let up: [f32; 3] = std::array::from_fn(|i| basis_up[i] * cr - basis_right[i] * sr);
    let half_width = (uniforms.optics.0[0] * 0.5).tan();
    let half_height = half_width / aspect.max(0.1);
    // A horizontal cut through the cone is an ellipse. Its major-axis
    // endpoints bound the distance from the target for every heading and roll.
    let mut limit = s / c.max(0.00001) * 0.99;
    for h in heights {
        for sign in [-1.0, 1.0] {
            let numerator = radius * s + sign * h * c;
            let denominator = distance - h * s + sign * radius * c;
            if denominator > 0.0 {
                limit = limit.min((numerator / denominator).max(0.00001));
            }
        }
    }
    let requested = half_width.hypot(half_height);
    let rounding = 0.02;
    let blend = (1.0 - (requested - limit).abs() / rounding).max(0.0);
    let cone = requested.min(limit) - 0.25 * rounding * blend * blend;
    let scale = cone.max(0.00001) / requested.max(0.00001);
    uniforms.eye = Float4([eye[0], eye[1], eye[2], 1.0]);
    uniforms.forward = Float4([forward[0], forward[1], forward[2], 0.0]);
    uniforms.right = Float4([right[0], right[1], right[2], half_width * scale]);
    uniforms.up = Float4([up[0], up[1], up[2], half_height * scale]);
}

fn normalized(v: [f32; 3]) -> [f32; 3] {
    let length = v.into_iter().map(|x| x * x).sum::<f32>().sqrt();
    v.map(|x| x / length)
}
fn cross(
    a: [f32; 3],
    b: [f32; 3],
) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Top-left drawable UV to local metres, using the same physical water lattice as the shader.
pub fn surface_point(
    uv: [f32; 2],
    uniforms: &ARUniforms,
    water: &[f32],
    width: usize,
    height: usize,
) -> Option<[f32; 2]> {
    if !uv
        .into_iter()
        .all(|x| x.is_finite() && (0.0..=1.0).contains(&x))
        || width <= 1
        || height <= 1
        || width.checked_mul(height)?.checked_mul(4)? != water.len()
    {
        return None;
    }
    let eye: [f32; 3] = std::array::from_fn(|i| uniforms.eye.0[i]);
    let ray: [f32; 3] = std::array::from_fn(|i| {
        uniforms.forward.0[i]
            + uniforms.right.0[i] * ((2.0 * uv[0] - 1.0) * uniforms.right.0[3])
            + uniforms.up.0[i] * ((1.0 - 2.0 * uv[1]) * uniforms.up.0[3])
    });
    if ray[2] >= -0.0001 {
        return None;
    }
    let cubic = |h: [f32; 4], t: f32| {
        // Identical to waterCubicCoefficients in the shared surface shader.
        let a = 0.5 * (-h[0] + 3.0 * h[1] - 3.0 * h[2] + h[3]);
        let b = h[0] - 2.5 * h[1] + 2.0 * h[2] - 0.5 * h[3];
        let c = 0.5 * (h[2] - h[0]);
        ((a * t + b) * t + c) * t + h[1]
    };
    let residual = |distance: f32| {
        let point: [f32; 3] = std::array::from_fn(|i| eye[i] + ray[i] * distance);
        let x = point[0] * 0.5 * width as f32;
        let y = point[1] * 0.5 * height as f32;
        // Floor the unbounded lattice coordinate; clamp ghost samples only.
        let (ix, iy) = (x.floor() as isize, y.floor() as isize);
        let (fx, fy) = (x - ix as f32, y - iy as f32);
        let rows = std::array::from_fn(|row| {
            let sy = (iy + row as isize - 1).clamp(0, height as isize - 1) as usize;
            let samples = std::array::from_fn(|column| {
                let sx = (ix + column as isize - 1).clamp(0, width as isize - 1) as usize;
                water[(sy * width + sx) * 4]
            });
            cubic(samples, fx)
        });
        point[2] - cubic(rows, fy)
    };
    let mut near = ((uniforms.world.0[2] + 0.00001 - eye[2]) / ray[2]).max(0.0);
    let mut far = (uniforms.world.0[1] - 0.00001 - eye[2]) / ray[2];
    if far <= near || residual(near) < 0.0 || residual(far) > 0.0 {
        return None;
    }
    let (mut a, mut b) = (residual(near), residual(far));
    // Match the primary shader's safeguarded surface intersection.
    for _ in 0..20 {
        let weight = (a / (a - b).max(1e-9)).clamp(0.05, 0.95);
        let middle = near + (far - near) * weight;
        let value = residual(middle);
        if value > 0.0 {
            near = middle;
            a = value;
        } else {
            far = middle;
            b = value;
        }
    }
    let distance = (near + far) * 0.5;
    let point = [eye[0] + ray[0] * distance, eye[1] + ray[1] * distance];
    point
        .into_iter()
        .all(|x| (0.0..=2.0).contains(&x))
        .then_some(point)
}

#[derive(Clone, Debug)]
pub struct CameraPresentation {
    pose: Option<[f64; 4]>,
    velocity: [f64; 4],
    roll_residual: f64,
    roll_velocity: f64,
    last_scene_time: f32,
    envelope: [f64; 2],
}
impl Default for CameraPresentation {
    fn default() -> Self {
        Self {
            pose: None,
            velocity: [0.0; 4],
            roll_residual: 0.0,
            roll_velocity: 0.0,
            last_scene_time: 0.0,
            envelope: NOMINAL_HEIGHT_ENVELOPE.map(f64::from),
        }
    }
}
impl CameraPresentation {
    pub fn reset(&mut self) { *self = Self::default(); }

    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &mut self,
        uniforms: &mut ARUniforms,
        origin: [f64; 2],
        snapshot_scene_time: f32,
        presentation_scene_time: f32,
        delta: f64,
        held: bool,
        aspect: f32,
    ) {
        let target = [
            origin[0] + f64::from(uniforms.camera.0[0]),
            origin[1] + f64::from(uniforms.camera.0[1]),
            f64::from(uniforms.camera.0[3]),
            f64::from(uniforms.optics.0[0]),
        ];
        let roll_rate = 4.0 * std::f64::consts::PI / 180.0;
        let desired_residual =
            f64::from(uniforms.camera.0[2]) - f64::from(snapshot_scene_time) * roll_rate;
        let dt = if delta.is_finite() {
            delta.clamp(0.0, 0.1)
        } else {
            0.0
        };
        match &mut self.pose {
            None => {
                self.pose = Some(target);
                self.roll_residual = desired_residual;
                self.last_scene_time = presentation_scene_time;
            },
            Some(pose) if !held => {
                let omega = 1.0 / 0.06;
                let decay = (-omega * dt).exp();
                for ((position, velocity), desired) in
                    pose.iter_mut().zip(&mut self.velocity).zip(target)
                {
                    let displacement = *position - desired;
                    let transient = *velocity + omega * displacement;
                    *position = desired + (displacement + transient * dt) * decay;
                    *velocity = (*velocity - omega * transient * dt) * decay;
                }
                let displacement = self.roll_residual - desired_residual;
                let transient = self.roll_velocity + omega * displacement;
                self.roll_residual = desired_residual + (displacement + transient * dt) * decay;
                self.roll_velocity = (self.roll_velocity - omega * transient * dt) * decay;
                self.last_scene_time = presentation_scene_time;
            },
            Some(_) => {},
        }
        // Expand exceptional wave bounds immediately, then release over two
        // seconds. Ordinary rain leaves the envelope exactly constant.
        let desired_envelope = [
            f64::from(NOMINAL_HEIGHT_ENVELOPE[0]).min(f64::from(uniforms.world.0[1]) - 0.002),
            f64::from(NOMINAL_HEIGHT_ENVELOPE[1]).max(f64::from(uniforms.world.0[2]) + 0.002),
        ];
        let release = if held { 1.0 } else { (-dt / 2.0).exp() };
        self.envelope[0] = desired_envelope[0]
            .min(desired_envelope[0] + (self.envelope[0] - desired_envelope[0]) * release);
        self.envelope[1] = desired_envelope[1]
            .max(desired_envelope[1] + (self.envelope[1] - desired_envelope[1]) * release);
        let pose = self.pose.unwrap();
        uniforms.camera = Float4([
            (pose[0] - origin[0]) as f32,
            (pose[1] - origin[1]) as f32,
            (f64::from(self.last_scene_time) * roll_rate + self.roll_residual) as f32,
            pose[2] as f32,
        ]);
        uniforms.optics.0[0] = pose[3] as f32;
        uniforms.time.0[0] = self.last_scene_time;
        fill(uniforms, aspect, Some(self.envelope.map(|x| x as f32)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(
        time: f32,
        target: [f32; 2],
    ) -> ARUniforms {
        let mut u = ARUniforms::default();
        u.time.0[0] = time;
        u.camera = Float4([
            target[0],
            target[1],
            time * 4.0_f32.to_radians(),
            70.0_f32.to_radians(),
        ]);
        u.optics.0[0] = 61.0_f32.to_radians();
        u.world = Float4([0.0, -0.03, 0.03, 0.0]);
        u
    }
    fn norm(v: [f32; 3]) -> f32 { v.into_iter().map(|x| x * x).sum::<f32>().sqrt() }
    fn ray(
        u: &ARUniforms,
        uv: [f32; 2],
    ) -> [f32; 3] {
        std::array::from_fn(|i| {
            u.forward.0[i]
                + u.right.0[i] * ((2.0 * uv[0] - 1.0) * u.right.0[3])
                + u.up.0[i] * ((1.0 - 2.0 * uv[1]) * u.up.0[3])
        })
    }

    #[test]
    fn stable_lens_matches_native_rain_roll_orbit_and_rebase_fixture() {
        let mut baseline = pose(7.25, [1.0; 2]);
        fill(&mut baseline, 1.6, None);
        for i in 0..=720 {
            let time = i as f32 * 0.25;
            let mut u = pose(time, [1.0 + 1.0 / 96.0, 1.0 - 1.0 / 64.0]);
            u.world.0[1] = -0.08;
            u.world.0[2] = 0.08;
            u.camera.0[2] = time * 0.13;
            fill(&mut u, 1.6, None);
            assert!((u.right.0[3] - baseline.right.0[3]).abs() < 1e-7);
            assert!((u.up.0[3] - baseline.up.0[3]).abs() < 1e-7);
        }
    }

    #[test]
    fn conservative_frustum_matches_native_223200_intersection_fixture() {
        let mut intersections = 0;
        for seconds in (0..=180).step_by(6) {
            for aspect in [0.5, 1.0, 1.6, 3.0] {
                for elevation in [0.45, 0.8, 1.1] {
                    for heights in [[-0.16, 0.12], [-0.35, 0.25], [-0.55, 0.60]] {
                        for target in [
                            [1.0, 1.0],
                            [1.0 + 1.0 / 96.0, 1.0 - 1.0 / 64.0],
                            [0.96, 1.0],
                            [1.06, 1.02],
                        ] {
                            let mut u = pose(seconds as f32, target);
                            u.camera.0[2] = seconds as f32 * 0.13;
                            u.camera.0[3] = elevation;
                            u.world.0[1] = heights[0];
                            u.world.0[2] = heights[1];
                            fill(&mut u, aspect, None);
                            assert!((u.right.0[3] / u.up.0[3] - aspect).abs() < 1e-5);
                            for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
                                for y in [0.0, 0.25, 0.5, 0.75, 1.0] {
                                    let ray = ray(&u, [x, y]);
                                    assert!(ray[2] < 0.0);
                                    for h in heights {
                                        let distance = (h - u.eye.0[2]) / ray[2];
                                        assert!(distance > 0.0);
                                        for (axis, direction) in ray.iter().enumerate().take(2) {
                                            let hit = u.eye.0[axis] + direction * distance;
                                            assert!((0.05998..=1.94002).contains(&hit));
                                        }
                                        intersections += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(intersections, 223_200);
    }

    #[test]
    fn picking_matches_cubic_crests_and_clamped_ghost_samples_on_both_axes() {
        // The tilted axis reveals hit height; the profile axis stays fixed.
        let fixtures = [
            ([0.0, 0.08, 0.08, 0.0], 0.75, 0.09),
            ([0.08, 0.0, 0.0, 0.0], 0.125, 0.06375),
            ([0.0, 0.0, 0.0, 0.08], 1.625, 0.085625),
        ];
        for axis in 0..2 {
            for (profile, coordinate, height) in fixtures {
                let mut u = ARUniforms {
                    eye: Float4([0.25, 0.25, 1.0, 1.0]),
                    forward: Float4([0.0, 0.0, -1.0, 0.0]),
                    world: Float4([0.0, -0.12, 0.12, 0.0]),
                    ..ARUniforms::default()
                };
                u.eye.0[axis] = coordinate;
                u.forward.0[1 - axis] = 0.5;
                let mut water = vec![0.0; 4 * 4 * 4];
                for y in 0..4 {
                    for x in 0..4 {
                        water[(y * 4 + x) * 4] = profile[if axis == 0 { x } else { y }];
                    }
                }
                let point = surface_point([0.5; 2], &u, &water, 4, 4).unwrap();
                let expected = 0.25 + 0.5 * (1.0 - height);
                assert!((point[axis] - coordinate).abs() < 0.00001);
                assert!(
                    (point[1 - axis] - expected).abs() < 0.00001,
                    "axis {axis}, coordinate {coordinate}: got {}, expected {expected}",
                    point[1 - axis]
                );
            }
        }
    }

    #[test]
    fn picking_matches_native_displaced_lattice_and_rejects_invalid_input() {
        let mut u = pose(0.0, [1.0; 2]);
        u.camera.0[2] = 0.23;
        u.camera.0[3] = 0.8;
        u.optics.0[0] = 0.9;
        u.world.0[1] = -0.08;
        u.world.0[2] = 0.08;
        fill(&mut u, 1.6, None);
        let (width, height) = (64, 48);
        let mut water = vec![0.0; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                water[(y * width + x) * 4] = 0.03 * (2.0 * x as f32 / width as f32 - 1.0);
            }
        }
        for uv in [[0.5, 0.5], [0.1, 0.2], [0.9, 0.8]] {
            let point = surface_point(uv, &u, &water, width, height).unwrap();
            let ray = ray(&u, uv);
            let distance = (0.03 * (u.eye.0[0] - 1.0) - u.eye.0[2]) / (ray[2] - 0.03 * ray[0]);
            let expected = [
                u.eye.0[0] + ray[0] * distance,
                u.eye.0[1] + ray[1] * distance,
            ];
            assert!((point[0] - expected[0]).hypot(point[1] - expected[1]) < 0.0001);
        }
        assert!(surface_point([-0.1, 0.5], &u, &water, width, height).is_none());
        assert!(surface_point([f32::NAN, 0.5], &u, &water, width, height).is_none());
        assert!(surface_point([0.5, 0.5], &u, &[], width, height).is_none());
    }

    #[test]
    fn damping_matches_native_cadence_hold_world_rebase_and_emergency_release() {
        let origin = [-1.0; 2];
        let mut whole = CameraPresentation::default();
        let mut split = CameraPresentation::default();
        let mut a = pose(0.0, [1.0; 2]);
        let mut b = a;
        whole.fill(&mut a, origin, 0.0, 0.0, 0.0, false, 1.6);
        split.fill(&mut b, origin, 0.0, 0.0, 0.0, false, 1.6);
        let mut source = pose(0.0, [1.01, 1.005]);
        source.camera.0[3] += 0.05;
        source.optics.0[0] -= 0.08;
        source.camera.0[2] += 0.03;
        a = source;
        whole.fill(&mut a, origin, 0.0, 0.1, 0.1, false, 1.6);
        for i in 1..=6 {
            b = source;
            split.fill(&mut b, origin, 0.0, i as f32 / 60.0, 1.0 / 60.0, false, 1.6);
        }
        for (x, y) in a.camera.0.into_iter().zip(b.camera.0) {
            assert!((x - y).abs() < 1e-6);
        }
        assert!((a.optics.0[0] - b.optics.0[0]).abs() < 1e-6);
        let world_eye = [
            f64::from(a.eye.0[0]) + origin[0],
            f64::from(a.eye.0[1]) + origin[1],
            f64::from(a.eye.0[2]),
        ];
        let rebased_origin = [origin[0] + 2.0 / 96.0, origin[1] - 2.0 / 64.0];
        let mut held = pose(100.0, [1.5, 0.5]);
        whole.fill(&mut held, rebased_origin, 100.0, 100.0, 0.1, true, 1.6);
        let after_eye = [
            f64::from(held.eye.0[0]) + rebased_origin[0],
            f64::from(held.eye.0[1]) + rebased_origin[1],
            f64::from(held.eye.0[2]),
        ];
        for (x, y) in world_eye.into_iter().zip(after_eye) {
            assert!((x - y).abs() < 1e-6);
        }
        assert!(norm(std::array::from_fn(|i| a.right.0[i] - held.right.0[i])) < 1e-6);
        let mut emergency = CameraPresentation::default();
        let mut normal = pose(0.0, [1.0; 2]);
        emergency.fill(&mut normal, origin, 0.0, 0.0, 0.0, false, 1.6);
        let mut extreme = pose(0.0, [1.0; 2]);
        extreme.world.0[1] = -0.4;
        emergency.fill(&mut extreme, origin, 0.0, 0.0, 1.0 / 60.0, false, 1.6);
        let mut release = pose(0.0, [1.0; 2]);
        emergency.fill(&mut release, origin, 0.0, 0.0, 1.0 / 60.0, false, 1.6);
        assert!(extreme.right.0[3] < release.right.0[3] && release.right.0[3] < normal.right.0[3]);
        assert!(
            release.right.0[3] - extreme.right.0[3]
                < (normal.right.0[3] - extreme.right.0[3]) * 0.02
        );
    }

    #[test]
    fn sampled_world_target_matches_native_velocity_and_target_reserve_fixture() {
        let mut camera = CameraPresentation::default();
        let mut previous_raw = 0.0;
        let mut previous_smooth = 0.0;
        let mut previous_raw_velocity = 0.0;
        let mut previous_smooth_velocity = 0.0;
        let mut max_raw_jump = 0.0_f64;
        let mut max_smooth_jump = 0.0_f64;
        let mut max_offset = 0.0_f32;
        let mut aperture: Option<f32> = None;
        for frame in 0..=360 {
            let time = f64::from(frame) / 60.0;
            let sample_time = (time * 30.0 + 0.000001).floor() / 30.0;
            let target = [0.13 * sample_time, 0.02 * sample_time];
            let origin = [
                -1.0 + (target[0] / (2.0 / 96.0)).round() * (2.0 / 96.0),
                -1.0 + (target[1] / (2.0 / 64.0)).round() * (2.0 / 64.0),
            ];
            let mut u = pose((sample_time * 2.4) as f32, [
                (target[0] - origin[0]) as f32,
                (target[1] - origin[1]) as f32,
            ]);
            camera.fill(
                &mut u,
                origin,
                (sample_time * 2.4) as f32,
                (time * 2.4) as f32,
                1.0 / 60.0,
                false,
                1.6,
            );
            let smooth = origin[0] + f64::from(u.camera.0[0]);
            let raw_velocity = (target[0] - previous_raw) * 60.0;
            let smooth_velocity = (smooth - previous_smooth) * 60.0;
            if frame > 2 {
                max_raw_jump = max_raw_jump.max((raw_velocity - previous_raw_velocity).abs());
                max_smooth_jump =
                    max_smooth_jump.max((smooth_velocity - previous_smooth_velocity).abs());
            }
            previous_raw = target[0];
            previous_smooth = smooth;
            previous_raw_velocity = raw_velocity;
            previous_smooth_velocity = smooth_velocity;
            max_offset = max_offset.max((u.camera.0[0] - 1.0).hypot(u.camera.0[1] - 1.0));
            if let Some(aperture) = aperture {
                assert!((u.right.0[3] - aperture).abs() < 1e-7);
            } else {
                aperture = Some(u.right.0[3]);
            }
        }
        assert!(max_offset < TARGET_RESERVE);
        assert!(max_smooth_jump < max_raw_jump * 0.15);
    }
}
