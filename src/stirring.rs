//! Coherent, bounded near-surface wakes seeded by impacts in an existing current.
//!
//! Each source is an equal/opposite pair of compact streamfunction blobs. Their
//! centres follow the mean current and each other's induced flow, using a common
//! midpoint state. A perfectly vertical impact in still water injects no swirl.
//! This is an empirical sheared-impact wake closure, not a 3-D splash solver or
//! a solved vorticity equation. Core spreading and drag dissipate wake activity;
//! they never remove pigment. Compact support is C2 in psi, C1 in velocity.
use std::time::Duration;

use crate::dye::Boundary;

const LENGTH: f32 = 2.0;
const TICK: Duration = Duration::from_millis(8);
const MAX_PAIRS: usize = 16;
const MAX_SPEED: f32 = 0.36;
const SUPPORT_SPREAD: f32 = 0.00010;
const DECAY_TIME: f32 = 1.4;
const MAX_AGE: f32 = 8.0;
const OPEN_TAPER: f32 = 0.18;
const BINS: usize = 16;
// max_{0<=r<=R} 6r/R * (1-r²/R²)² = 96/(25 sqrt(5)).
const PEAK_FACTOR: f32 = 1.7173002;

#[derive(Clone, Copy, Debug)]
struct Pair {
    centers: [[f32; 2]; 2],
    initial_radius: f32,
    radius: f32,
    initial_amplitude: f32,
    amplitude: f32,
    age: f32,
}
impl Pair {
    fn update_envelope(&mut self) {
        self.radius =
            (self.initial_radius * self.initial_radius + 4.0 * SUPPORT_SPREAD * self.age).sqrt();
        let birth = smoothstep(self.age / 0.16);
        let retirement = 1.0 - smoothstep((self.age - 6.0) / 2.0);
        self.amplitude =
            self.initial_amplitude * birth * (-self.age / DECAY_TIME).exp() * retirement;
    }

    fn remaining_strength(&self) -> f32 {
        self.initial_amplitude.abs() * (-self.age / DECAY_TIME).exp() / self.radius
    }

    fn overlaps_plane_window(&self) -> bool {
        self.centers.iter().any(|center| {
            let dx = center[0] - center[0].clamp(0.0, LENGTH);
            let dy = center[1] - center[1].clamp(0.0, LENGTH);
            dx * dx + dy * dy < self.radius * self.radius
        })
    }
}

#[derive(Debug)]
pub struct Stirring {
    pairs: Vec<Pair>,
    midpoint: Vec<Pair>,
    pending: Duration,
    scale: f32,
}
impl Default for Stirring {
    fn default() -> Self { Self::new() }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn taper(
    x: f32,
    boundary: Boundary,
) -> (f32, f32) {
    if boundary != Boundary::OpenChannel {
        return (1.0, 0.0);
    }
    if x <= 0.0 || x >= LENGTH {
        return (0.0, 0.0);
    }
    if x < OPEN_TAPER {
        let s = x / OPEN_TAPER;
        (
            s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
            30.0 * s * s * (1.0 - s) * (1.0 - s) / OPEN_TAPER,
        )
    } else if x > LENGTH - OPEN_TAPER {
        let s = (LENGTH - x) / OPEN_TAPER;
        (
            s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
            -30.0 * s * s * (1.0 - s) * (1.0 - s) / OPEN_TAPER,
        )
    } else {
        (1.0, 0.0)
    }
}

fn displacement(
    point: [f32; 2],
    center: [f32; 2],
    boundary: Boundary,
) -> [f32; 2] {
    let wrap = |x: f32| (x + LENGTH * 0.5).rem_euclid(LENGTH) - LENGTH * 0.5;
    [
        if boundary == Boundary::Periodic {
            wrap(point[0] - center[0])
        } else {
            point[0] - center[0]
        },
        if boundary == Boundary::OpenPlane {
            point[1] - center[1]
        } else {
            wrap(point[1] - center[1])
        },
    ]
}

fn canonicalize(
    point: &mut [f32; 2],
    boundary: Boundary,
) {
    if boundary == Boundary::Periodic {
        point[0] = point[0].rem_euclid(LENGTH);
    }
    if boundary != Boundary::OpenPlane {
        point[1] = point[1].rem_euclid(LENGTH);
    }
}

fn sample_pairs(
    pairs: &[Pair],
    scale: f32,
    point: [f32; 2],
    boundary: Boundary,
) -> [f32; 2] {
    let (window, derivative) = taper(point[0], boundary);
    if window == 0.0 {
        return [0.0; 2];
    }
    let mut psi = 0.0;
    let mut velocity = [0.0; 2];
    for pair in pairs {
        let inverse = 1.0 / (pair.radius * pair.radius);
        for (center, sign) in pair.centers.iter().zip([-1.0, 1.0]) {
            let d = displacement(point, *center, boundary);
            let weight = 1.0 - (d[0] * d[0] + d[1] * d[1]) * inverse;
            if weight <= 0.0 {
                continue;
            }
            let amplitude = pair.amplitude * sign;
            psi += amplitude * weight * weight * weight;
            let gradient = 6.0 * amplitude * inverse * weight * weight;
            velocity[0] += gradient * d[1];
            velocity[1] -= gradient * d[0];
        }
    }
    [
        velocity[0] * window * scale,
        (velocity[1] * window + derivative * psi) * scale,
    ]
}

/// Conservative spatial upper bound: every point belongs to a bin and every
/// blob whose support meets that bin contributes its maximum possible speed.
/// One multiplier scales the entire psi field, retaining zero divergence.
fn normalization(
    pairs: &[Pair],
    boundary: Boundary,
) -> f32 {
    let mut bounds = [0.0_f32; BINS * BINS];
    let spacing = LENGTH / BINS as f32;
    for pair in pairs {
        let bound = pair.amplitude.abs()
            * (PEAK_FACTOR / pair.radius
                + if boundary == Boundary::OpenChannel {
                    1.875 / OPEN_TAPER
                } else {
                    0.0
                });
        if bound == 0.0 {
            continue;
        }
        for center in pair.centers {
            let x0 = ((center[0] - pair.radius) / spacing).floor() as isize;
            let x1 = ((center[0] + pair.radius) / spacing).floor() as isize;
            let y0 = ((center[1] - pair.radius) / spacing).floor() as isize;
            let y1 = ((center[1] + pair.radius) / spacing).floor() as isize;
            for y in y0..=y1 {
                let y = if boundary != Boundary::OpenPlane {
                    y.rem_euclid(BINS as isize) as usize
                } else if (0..BINS as isize).contains(&y) {
                    y as usize
                } else {
                    continue;
                };
                for x in x0..=x1 {
                    let x = if boundary == Boundary::Periodic {
                        x.rem_euclid(BINS as isize) as usize
                    } else if (0..BINS as isize).contains(&x) {
                        x as usize
                    } else {
                        continue;
                    };
                    bounds[y * BINS + x] += bound;
                }
            }
        }
    }
    (MAX_SPEED / bounds.into_iter().fold(0.0_f32, f32::max).max(MAX_SPEED)).min(1.0)
}

impl Stirring {
    pub fn new() -> Self {
        Self {
            pairs: Vec::with_capacity(MAX_PAIRS),
            midpoint: Vec::with_capacity(MAX_PAIRS),
            pending: Duration::ZERO,
            scale: 1.0,
        }
    }

    pub fn is_active(&self) -> bool { !self.pairs.is_empty() }

    /// Recenter an open-plane window without advancing time or moving any
    /// surviving wake in world space. Centres may remain outside the rectangle
    /// while their compact support still overlaps it. Keep the current common
    /// normalization through the shift; recalculating its bin bound here could
    /// otherwise change a surviving wake solely because the window moved.
    pub(crate) fn shift_origin(
        &mut self,
        delta_m: [f32; 2],
    ) {
        if !delta_m.iter().all(|value| value.is_finite()) {
            return;
        }
        for pairs in [&mut self.pairs, &mut self.midpoint] {
            for pair in pairs.iter_mut() {
                for center in &mut pair.centers {
                    for axis in 0..2 {
                        center[axis] -= delta_m[axis];
                    }
                }
            }
            pairs.retain(Pair::overlaps_plane_window);
        }
    }

    /// Seed a weak rotational wake aligned with the pre-impact current. The
    /// radius is the impact's momentum footprint in metres, not a pixel radius.
    /// Source strength is empirical and proportional to local current speed;
    /// there is no unexplained horizontal angular momentum from a still impact.
    pub fn deposit(
        &mut self,
        point: [f32; 2],
        radius_m: f32,
        strength: f32,
        base_velocity: [f32; 2],
        boundary: Boundary,
    ) {
        if !point
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            || !base_velocity.iter().all(|v| v.is_finite())
            || !radius_m.is_finite()
            || radius_m <= 0.0
            || !strength.is_finite()
            || strength <= 0.0
        {
            return;
        }
        let speed = base_velocity[0].hypot(base_velocity[1]);
        if !speed.is_finite() || speed < 1e-6 {
            return;
        }
        let radius = (radius_m * 1.6).clamp(0.075, 0.30);
        let peak = (0.48 * speed * (strength.min(1.0) / 0.2).sqrt()).min(0.14);
        let amplitude = peak * radius / PEAK_FACTOR;
        if self.pairs.len() == MAX_PAIRS {
            let (oldest, weak) = self
                .pairs
                .iter()
                .enumerate()
                .map(|(i, pair)| (i, pair.remaining_strength()))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            // Preserve established wakes during a burst. Replace only a source
            // that has already lost >=80% of the new source's potential speed;
            // otherwise omit this unresolved source. The bounded weak-source
            // truncation is separate from the physical pigment conservation.
            if weak > 0.20 * amplitude / radius {
                return;
            }
            self.pairs.remove(oldest);
        }
        let offset = [
            -base_velocity[1] / speed * radius * 0.30,
            base_velocity[0] / speed * radius * 0.30,
        ];
        let mut centers = [
            [point[0] * LENGTH + offset[0], point[1] * LENGTH + offset[1]],
            [point[0] * LENGTH - offset[0], point[1] * LENGTH - offset[1]],
        ];
        for center in &mut centers {
            canonicalize(center, boundary);
        }
        self.pairs.push(Pair {
            centers,
            initial_radius: radius,
            radius,
            initial_amplitude: amplitude,
            amplitude: 0.0,
            age: 0.0,
        });
        self.scale = normalization(&self.pairs, boundary);
    }

    pub fn advance(
        &mut self,
        duration: Duration,
        base_velocity: impl Fn([f32; 2]) -> [f32; 2],
        boundary: Boundary,
    ) {
        self.pending = self.pending.saturating_add(duration);
        while self.pending >= TICK {
            self.pending -= TICK;
            if self.pairs.is_empty() {
                continue;
            }
            let dt = TICK.as_secs_f32();
            self.scale = normalization(&self.pairs, boundary);
            self.midpoint.clear();
            self.midpoint.extend_from_slice(&self.pairs);
            for (pair, middle) in self.pairs.iter().zip(&mut self.midpoint) {
                for (position, predicted) in pair.centers.iter().zip(&mut middle.centers) {
                    let point = if boundary == Boundary::OpenPlane {
                        position.map(|v| v / LENGTH)
                    } else {
                        [(position[0] / LENGTH).clamp(0.0, 1.0), position[1] / LENGTH]
                    };
                    let base = base_velocity(point).map(|v| if v.is_finite() { v } else { 0.0 });
                    let wake = sample_pairs(&self.pairs, self.scale, *position, boundary);
                    for axis in 0..2 {
                        predicted[axis] = position[axis] + 0.5 * dt * (base[axis] + wake[axis]);
                    }
                    canonicalize(predicted, boundary);
                }
                middle.age += 0.5 * dt;
                middle.update_envelope();
            }
            let middle_scale = normalization(&self.midpoint, boundary);
            for (pair, middle) in self.pairs.iter_mut().zip(&self.midpoint) {
                for (position, predicted) in pair.centers.iter_mut().zip(&middle.centers) {
                    let point = if boundary == Boundary::OpenPlane {
                        predicted.map(|v| v / LENGTH)
                    } else {
                        [
                            (predicted[0] / LENGTH).clamp(0.0, 1.0),
                            predicted[1] / LENGTH,
                        ]
                    };
                    let base = base_velocity(point).map(|v| if v.is_finite() { v } else { 0.0 });
                    let wake = sample_pairs(&self.midpoint, middle_scale, *predicted, boundary);
                    for axis in 0..2 {
                        position[axis] += dt * (base[axis] + wake[axis]);
                    }
                    canonicalize(position, boundary);
                }
                pair.age += dt;
                pair.update_envelope();
            }
            self.pairs.retain(|pair| {
                pair.age < MAX_AGE
                    && match boundary {
                        Boundary::Periodic => true,
                        Boundary::OpenPlane => pair.overlaps_plane_window(),
                        Boundary::OpenChannel => pair
                            .centers
                            .iter()
                            .any(|p| p[0] + pair.radius > 0.0 && p[0] - pair.radius < LENGTH),
                    }
            });
            self.scale = normalization(&self.pairs, boundary);
        }
    }

    /// Wake-only velocity in m/s at normalized coordinates.
    #[cfg(test)]
    pub fn velocity(
        &self,
        point: [f32; 2],
        boundary: Boundary,
    ) -> [f32; 2] {
        if !point.iter().all(|p| p.is_finite()) {
            return [0.0; 2];
        }
        sample_pairs(&self.pairs, self.scale, point.map(|p| p * LENGTH), boundary)
    }

    /// Add psi [m²/s] to an existing row-major vertex field. Only compact source
    /// bounding boxes are visited; exponentials are cached per source per tick.
    /// Periodic edge vertices receive the same additions as their twins; open
    /// plane edges are independent and sources outside either axis are clipped.
    pub fn fill_streamfunction(
        &self,
        vertices: &mut [f32],
        nx: usize,
        ny: usize,
        boundary: Boundary,
    ) {
        if nx == 0 || ny == 0 || vertices.len() != (nx + 1) * (ny + 1) {
            return;
        }
        let dx = LENGTH / nx as f32;
        let dy = LENGTH / ny as f32;
        for pair in &self.pairs {
            let inverse = 1.0 / (pair.radius * pair.radius);
            for (center, sign) in pair.centers.iter().zip([-1.0, 1.0]) {
                let x0 = ((center[0] - pair.radius) / dx).ceil() as isize;
                let x1 = ((center[0] + pair.radius) / dx).floor() as isize;
                let y0 = ((center[1] - pair.radius) / dy).ceil() as isize;
                let y1 = ((center[1] + pair.radius) / dy).floor() as isize;
                let amplitude = pair.amplitude * sign * self.scale;
                for iy in y0..=y1 {
                    let ry = iy as f32 * dy - center[1];
                    let y = if boundary != Boundary::OpenPlane {
                        iy.rem_euclid(ny as isize) as usize
                    } else if (0..=ny as isize).contains(&iy) {
                        iy as usize
                    } else {
                        continue;
                    };
                    for ix in x0..=x1 {
                        let x = if boundary == Boundary::Periodic {
                            ix.rem_euclid(nx as isize) as usize
                        } else if (0..=nx as isize).contains(&ix) {
                            ix as usize
                        } else {
                            continue;
                        };
                        let rx = ix as f32 * dx - center[0];
                        let weight = (1.0 - (rx * rx + ry * ry) * inverse).max(0.0);
                        let value =
                            amplitude * weight * weight * weight * taper(x as f32 * dx, boundary).0;
                        vertices[y * (nx + 1) + x] += value;
                        if y == 0 && boundary != Boundary::OpenPlane {
                            vertices[ny * (nx + 1) + x] += value;
                        }
                        if x == 0 && boundary == Boundary::Periodic {
                            vertices[y * (nx + 1) + nx] += value;
                            if y == 0 {
                                vertices[ny * (nx + 1) + nx] += value;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifting_open_plane_keeps_the_same_world_wake_and_pending_tick() {
        let mut wakes = seeded(Boundary::OpenPlane);
        wakes.advance(
            Duration::from_millis(3),
            |_| [0.22, 0.04],
            Boundary::OpenPlane,
        );
        let pairs = wakes.pairs.clone();
        let midpoint = wakes.midpoint.clone();
        let pending = wakes.pending;
        let point = [0.50, 0.55];
        let velocity = wakes.velocity(point, Boundary::OpenPlane);
        assert!(velocity[0].hypot(velocity[1]) > 0.005);
        let delta = [0.125, -0.25];
        wakes.shift_origin(delta);
        assert_eq!(wakes.pending, pending);
        for (old, new) in pairs
            .iter()
            .zip(&wakes.pairs)
            .chain(midpoint.iter().zip(&wakes.midpoint))
        {
            assert_eq!(old.age, new.age);
            assert_eq!(old.radius, new.radius);
            assert_eq!(old.amplitude, new.amplitude);
            for (old, new) in old.centers.iter().zip(&new.centers) {
                assert_eq!(*new, [old[0] - delta[0], old[1] - delta[1]]);
            }
        }
        let shifted = wakes.velocity(
            [point[0] - delta[0] / LENGTH, point[1] - delta[1] / LENGTH],
            Boundary::OpenPlane,
        );
        for axis in 0..2 {
            assert!((velocity[axis] - shifted[axis]).abs() < 1e-6);
        }
    }

    #[test]
    fn shifting_open_plane_retires_only_wakes_without_support_in_the_window() {
        let mut wakes = seeded(Boundary::OpenPlane);
        let mut edge = wakes.pairs[0];
        edge.centers = [[-0.025, 1.0], [-0.05, 1.0]];
        edge.radius = 0.16;
        let mut corner = edge;
        corner.centers = [[-0.13, -0.13], [-0.14, -0.14]];
        wakes.pairs = vec![edge, corner];
        wakes.midpoint = wakes.pairs.clone();
        wakes.shift_origin([0.0; 2]);
        assert_eq!(wakes.pairs.len(), 1);
        assert_eq!(wakes.midpoint.len(), 1);
        assert_eq!(wakes.pairs[0].centers, edge.centers);
        wakes.shift_origin([0.0, -4.0]);
        assert!(wakes.pairs.is_empty());
        assert!(wakes.midpoint.is_empty());
    }

    #[test]
    fn open_plane_wakes_advect_out_through_y_without_becoming_stale_sources() {
        let mut wakes = Stirring::new();
        wakes.deposit([0.5, 0.9], 0.06, 0.2, [0.0, 0.3], Boundary::OpenPlane);
        wakes.advance(Duration::from_secs(3), |_| [0.0, 0.3], Boundary::OpenPlane);
        assert!(wakes.pairs.is_empty());
    }

    #[test]
    fn open_plane_coordinates_and_distances_do_not_wrap() {
        let mut position = [-0.15, 2.08];
        canonicalize(&mut position, Boundary::OpenPlane);
        assert_eq!(position, [-0.15, 2.08]);
        assert_eq!(displacement([0.2, 0.1], [0.2, 1.9], Boundary::OpenPlane), [
            0.0,
            0.1_f32 - 1.9
        ]);
    }

    #[test]
    fn open_plane_wake_does_not_reappear_at_opposite_y_edge() {
        let mut wakes = Stirring::new();
        wakes.deposit([0.5, 0.995], 0.1, 0.4, [0.2, 0.0], Boundary::OpenPlane);
        wakes.advance(
            Duration::from_millis(160),
            |_| [0.2, 0.0],
            Boundary::OpenPlane,
        );
        assert_eq!(wakes.velocity([0.51, 0.005], Boundary::OpenPlane), [0.0; 2]);
        let (nx, ny) = (80, 64);
        let mut psi = vec![0.0; (nx + 1) * (ny + 1)];
        wakes.fill_streamfunction(&mut psi, nx, ny, Boundary::OpenPlane);
        assert!(psi[..=nx].iter().all(|value| *value == 0.0));
        assert!(psi[ny * (nx + 1)..].iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn open_plane_streamfunction_agrees_with_untapered_velocity_at_edges() {
        let mut wakes = Stirring::new();
        wakes.deposit([0.01, 0.5], 0.1, 0.4, [0.2, 0.0], Boundary::OpenPlane);
        wakes.advance(
            Duration::from_millis(160),
            |_| [0.2, 0.0],
            Boundary::OpenPlane,
        );
        let (nx, ny) = (256, 256);
        let mut psi = vec![0.0; (nx + 1) * (ny + 1)];
        wakes.fill_streamfunction(&mut psi, nx, ny, Boundary::OpenPlane);
        assert!(psi.iter().all(|value| value.is_finite()));
        let (x, y) = (0, 132);
        let point = [x as f32 / nx as f32, y as f32 / ny as f32];
        let velocity = wakes.velocity(point, Boundary::OpenPlane);
        assert!(velocity[0].hypot(velocity[1]) > 0.005);
        // u=-d(psi)/dy, v=d(psi)/dx; use a second-order one-sided
        // derivative at the physical edge, which has no periodic twin.
        let h = LENGTH / nx as f32;
        let here = y * (nx + 1) + x;
        let from_psi = [
            -(psi[here + nx + 1] - psi[here - nx - 1]) / (2.0 * h),
            (-3.0 * psi[here] + 4.0 * psi[here + 1] - psi[here + 2]) / (2.0 * h),
        ];
        for axis in 0..2 {
            assert!(
                (from_psi[axis] - velocity[axis]).abs() < 0.002,
                "axis {axis}: finite difference {} vs analytic {}",
                from_psi[axis],
                velocity[axis]
            );
        }
    }

    #[test]
    fn open_taper_has_continuous_second_derivatives_at_both_join_types() {
        let epsilon = 0.0001;
        for (edge, direction) in [
            (0.0, 1.0),
            (OPEN_TAPER, -1.0),
            (LENGTH - OPEN_TAPER, 1.0),
            (LENGTH, -1.0),
        ] {
            let (_, slope) = taper(edge, Boundary::OpenChannel);
            let (_, near_slope) = taper(edge + direction * epsilon, Boundary::OpenChannel);
            assert!(slope.abs() < 1e-5);
            assert!(
                (near_slope - slope).abs() / epsilon < 1.0,
                "second derivative must approach zero at taper join {edge}"
            );
        }
    }
    fn seeded(boundary: Boundary) -> Stirring {
        let mut wakes = Stirring::new();
        wakes.deposit([0.48, 0.52], 0.06, 0.2, [0.22, 0.04], boundary);
        wakes.advance(Duration::from_millis(160), |_| [0.22, 0.04], boundary);
        wakes
    }
    #[test]
    fn vertical_impact_in_quiescent_water_does_not_invent_horizontal_vorticity() {
        let mut wakes = Stirring::new();
        wakes.deposit([0.5, 0.5], 0.06, 0.5, [0.0; 2], Boundary::Periodic);
        wakes.advance(Duration::from_secs(1), |_| [0.0; 2], Boundary::Periodic);
        assert!(wakes.pairs.is_empty());
        assert_eq!(wakes.velocity([0.5, 0.5], Boundary::Periodic), [0.0; 2]);
    }
    #[test]
    fn current_aligned_impact_creates_paired_coherent_stretching() {
        let wakes = seeded(Boundary::Periodic);
        assert_eq!(wakes.pairs.len(), 1);
        let pair = wakes.pairs[0];
        let mut middle = [
            (pair.centers[0][0] + pair.centers[1][0]) * 0.25,
            (pair.centers[0][1] + pair.centers[1][1]) * 0.25,
        ];
        middle[0] += pair.radius * 0.10;
        let epsilon = 0.0001;
        let xp = wakes.velocity([middle[0] + epsilon, middle[1]], Boundary::Periodic);
        let xm = wakes.velocity([middle[0] - epsilon, middle[1]], Boundary::Periodic);
        let yp = wakes.velocity([middle[0], middle[1] + epsilon], Boundary::Periodic);
        let ym = wakes.velocity([middle[0], middle[1] - epsilon], Boundary::Periodic);
        let strain = (xp[0] - xm[0]).abs()
            + (yp[1] - ym[1]).abs()
            + ((yp[0] - ym[0]) + (xp[1] - xm[1])).abs();
        assert!(
            strain / epsilon > 0.01,
            "wake must deform a material patch, not merely translate it"
        );
        assert!(pair.centers[0][0] > 0.90 && pair.centers[1][0] > 0.90);
    }
    #[test]
    fn fixed_ticks_and_coupled_midpoints_are_frame_partition_independent() {
        let mut a = Stirring::new();
        let mut b = Stirring::new();
        for flow in [&mut a, &mut b] {
            flow.deposit([0.48, 0.52], 0.06, 0.2, [0.22, 0.04], Boundary::Periodic);
            flow.deposit([0.55, 0.47], 0.08, 0.3, [0.22, 0.04], Boundary::Periodic);
        }
        a.advance(
            Duration::from_millis(800),
            |_| [0.22, 0.04],
            Boundary::Periodic,
        );
        for _ in 0..80 {
            b.advance(
                Duration::from_millis(10),
                |_| [0.22, 0.04],
                Boundary::Periodic,
            );
        }
        assert_eq!(a.pairs.len(), b.pairs.len());
        for (a, b) in a.pairs.iter().zip(&b.pairs) {
            assert_eq!(a.centers, b.centers);
            assert_eq!(a.age, b.age);
        }
    }
    #[test]
    fn periodic_streamfunction_has_matching_seams_and_open_edges_are_tapered() {
        for boundary in [Boundary::Periodic, Boundary::OpenChannel] {
            let mut wakes = Stirring::new();
            wakes.deposit([0.02, 0.98], 0.10, 0.4, [0.15, 0.03], boundary);
            wakes.advance(Duration::from_millis(160), |_| [0.15, 0.03], boundary);
            let (nx, ny) = (40, 32);
            let mut psi = vec![0.0; (nx + 1) * (ny + 1)];
            wakes.fill_streamfunction(&mut psi, nx, ny, boundary);
            assert!(psi.iter().any(|p| p.abs() > 1e-5));
            for x in 0..=nx {
                assert_eq!(psi[x], psi[ny * (nx + 1) + x]);
            }
            for y in 0..=ny {
                if boundary == Boundary::Periodic {
                    assert_eq!(psi[y * (nx + 1)], psi[y * (nx + 1) + nx]);
                } else {
                    assert_eq!(psi[y * (nx + 1)], 0.0);
                    assert_eq!(psi[y * (nx + 1) + nx], 0.0);
                }
            }
        }
    }
    #[test]
    fn burst_budget_is_finite_and_wakes_relax_without_moving_screen_noise() {
        let mut wakes = Stirring::new();
        for i in 0..1000 {
            wakes.deposit(
                [
                    ((i * 37) % 101) as f32 / 101.0,
                    ((i * 17) % 97) as f32 / 97.0,
                ],
                0.1,
                1.0,
                [4.0, -2.0],
                Boundary::Periodic,
            );
        }
        assert!(wakes.pairs.len() <= 16);
        wakes.advance(
            Duration::from_millis(200),
            |_| [0.2, 0.0],
            Boundary::Periodic,
        );
        let mut maximum = 0.0_f32;
        for y in 0..81 {
            for x in 0..81 {
                let v = wakes.velocity([x as f32 / 80.0, y as f32 / 80.0], Boundary::Periodic);
                maximum = maximum.max(v[0].hypot(v[1]));
            }
        }
        assert!(maximum > 0.01 && maximum <= 0.361, "speed={maximum}");
        let initial_radius = wakes.pairs[0].radius;
        wakes.advance(Duration::from_secs(12), |_| [0.2, 0.0], Boundary::Periodic);
        assert!(wakes.pairs.iter().all(|p| p.radius >= initial_radius));
        assert!(wakes.pairs.is_empty());
    }
}
