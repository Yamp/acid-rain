//! Rain-driven narrow-band optical waves over a resolved hydraulic current.
//!
//! This small-amplitude closure does not own hydraulic depth, momentum or dye.
//! The selected 0.18 m band is gravity dominated; surface tension contributes to
//! its finite-depth phase speed, but this is not a fully dispersive wave solver.
use std::time::Duration;

use crate::{
    physics::{
        DENSITY,
        GRAVITY,
        KINEMATIC_VISCOSITY,
        PATCH_LENGTH,
        SURFACE_TENSION,
    },
    waves::spectral_weight,
};

const NX: usize = 96;
const NY: usize = 64;
const CELLS: usize = NX * NY;
const DX: f32 = PATCH_LENGTH / NX as f32;
const DY: f32 = PATCH_LENGTH / NY as f32;
const WAVELENGTH: f32 = 0.18;
const K: f32 = std::f32::consts::TAU / WAVELENGTH;
const TICK: Duration = Duration::from_millis(8);

#[derive(Clone, Debug)]
pub(crate) struct Ripples {
    state: Vec<[f32; 2]>,
    predicted: Vec<[f32; 2]>,
    advected: Vec<[f32; 2]>,
    medium: Vec<[f32; 4]>,
    pending: Duration,
    active: bool,
}

#[derive(Clone, Copy)]
struct Footprint {
    indices: [usize; 4],
    blend: [f32; 2],
}

impl Footprint {
    fn new(
        x: f32,
        y: f32,
    ) -> Self {
        let x = wrapped(x, NX);
        let y = wrapped(y, NY);
        let ix = x.floor() as usize;
        let iy = y.floor() as usize;
        let jx = (ix + 1) % NX;
        let jy = (iy + 1) % NY;
        Self {
            indices: [iy * NX + ix, iy * NX + jx, jy * NX + ix, jy * NX + jx],
            blend: [x - ix as f32, y - iy as f32],
        }
    }

    fn sample(
        self,
        field: &[[f32; 2]],
    ) -> [f32; 2] {
        let [a, b, c, d] = self.indices.map(|i| field[i]);
        let [x, y] = self.blend;
        std::array::from_fn(|channel| {
            let upper = a[channel] + (b[channel] - a[channel]) * x;
            let lower = c[channel] + (d[channel] - c[channel]) * x;
            upper + (lower - upper) * y
        })
    }
}

#[inline]
fn cubic(
    values: [f32; 4],
    t: f32,
) -> f32 {
    let [a, b, c, d] = values;
    b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)))
}

#[inline]
fn wrapped(
    value: f32,
    size: usize,
) -> f32 {
    if value >= 0.0 && value < size as f32 {
        value
    } else {
        let remainder = value.rem_euclid(size as f32);
        // Floating-point Euclidean remainder can round a tiny negative input
        // up to the divisor. Both samplers require a strictly half-open range.
        if remainder >= size as f32 {
            0.0
        } else {
            remainder
        }
    }
}

fn transported(
    field: &[[f32; 2]],
    x: f32,
    y: f32,
) -> [f32; 2] {
    let x = wrapped(x, NX);
    let y = wrapped(y, NY);
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let tx = x - ix as f32;
    let ty = y - iy as f32;
    let columns = [(ix + NX - 1) % NX, ix, (ix + 1) % NX, (ix + 2) % NX];
    let rows = [(iy + NY - 1) % NY, iy, (iy + 1) % NY, (iy + 2) % NY];
    std::array::from_fn(|channel| {
        if tx == 0.0 {
            cubic(rows.map(|row| field[row * NX + ix][channel]), ty)
        } else {
            let values =
                rows.map(|row| cubic(columns.map(|column| field[row * NX + column][channel]), tx));
            cubic(values, ty)
        }
    })
}

fn force(
    field: &[[f32; 2]],
    medium: &[[f32; 4]],
    x: usize,
    y: usize,
) -> f32 {
    let i = y * NX + x;
    let left = y * NX + (x + NX - 1) % NX;
    let right = y * NX + (x + 1) % NX;
    let down = ((y + NY - 1) % NY) * NX + x;
    let up = ((y + 1) % NY) * NX + x;
    let height = field[i][0];
    let c2 = medium[i][2];
    // One symmetric coefficient per face makes this a negative-semidefinite
    // stiffness operator, including spatially varying hydraulic column depth.
    let flux = |j: usize| 0.5 * (c2 + medium[j][2]) * (field[j][0] - height);
    (flux(left) + flux(right)) / (DX * DX) + (flux(down) + flux(up)) / (DY * DY)
}

impl Ripples {
    pub fn new() -> Self {
        Self {
            state: vec![[0.0; 2]; CELLS],
            predicted: vec![[0.0; 2]; CELLS],
            advected: vec![[0.0; 2]; CELLS],
            medium: vec![[0.0; 4]; CELLS],
            pending: Duration::ZERO,
            active: false,
        }
    }

    pub fn deposit(
        &mut self,
        point: [f32; 2],
        strength: f32,
        size: f32,
    ) {
        if !point.iter().all(|p| p.is_finite())
            || !strength.is_finite()
            || strength <= 0.0
            || !size.is_finite()
            || size <= 0.0
        {
            return;
        }
        let strength = strength.min(1.0);
        let source_sigma = (0.055 + 0.035 * strength.sqrt()) * size.clamp(0.25, 3.0);
        let sigma = source_sigma.max(1.25 * DX.max(DY));
        // Gaussian filtering of a compensated Laplacian impulse preserves its
        // low-frequency pressure moment with this fourth-power amplitude scale.
        // The actual hydraulic/drop diameter is not changed by optical filtering.
        let source_scale = (source_sigma / sigma).powi(4);
        let center = point.map(|p| p.rem_euclid(1.0) * PATCH_LENGTH);
        for y in 0..NY {
            for x in 0..NX {
                let dx = ((x as f32 + 0.5) * DX - center[0] + 1.0).rem_euclid(PATCH_LENGTH) - 1.0;
                let dy = ((y as f32 + 0.5) * DY - center[1] + 1.0).rem_euclid(PATCH_LENGTH) - 1.0;
                let q = (dx * dx + dy * dy) / (sigma * sigma);
                if q < 20.0 {
                    // Compensated pressure footprint: depression plus a displaced
                    // annulus, in vertical m/s. Its small optical energy is empirical.
                    self.state[y * NX + x][1] +=
                        0.05 * strength * source_scale * (q - 2.0) * (-0.5 * q).exp();
                }
            }
        }
        self.remove_mean();
        self.active = true;
    }

    fn remove_mean(&mut self) {
        let mean = self
            .state
            .iter()
            .fold([0.0_f64; 2], |sum, value| {
                [sum[0] + f64::from(value[0]), sum[1] + f64::from(value[1])]
            })
            .map(|sum| (sum / CELLS as f64) as f32);
        for value in &mut self.state {
            value[0] -= mean[0];
            value[1] -= mean[1];
        }
    }

    pub fn advance(
        &mut self,
        duration: Duration,
        water: impl Fn([f32; 2]) -> ([f32; 2], f32),
    ) {
        self.pending += duration;
        if !self.active {
            self.pending = Duration::from_nanos((self.pending.as_nanos() % TICK.as_nanos()) as u64);
            return;
        }
        if self.pending < TICK {
            return;
        }
        for y in 0..NY {
            for x in 0..NX {
                let (velocity, depth) =
                    water([(x as f32 + 0.5) / NX as f32, (y as f32 + 0.5) / NY as f32]);
                let depth = if depth.is_finite() {
                    depth.max(0.001)
                } else {
                    0.16
                };
                let velocity = velocity.map(|v| if v.is_finite() { v } else { 0.0 });
                self.medium[y * NX + x] = [
                    velocity[0],
                    velocity[1],
                    (GRAVITY / K + SURFACE_TENSION / DENSITY * K) * (K * depth).tanh(),
                    depth,
                ];
            }
        }
        while self.pending >= TICK {
            self.pending -= TICK;
            self.tick();
        }
    }

    fn tick(&mut self) {
        let dt = TICK.as_secs_f32();
        // Signed optical waves need no concentration-positivity limiter.
        // Cubic reconstruction preserves their resolved peaks; for a uniform
        // translation its Fourier gain is <= 1. Nonuniform focusing is bounded
        // by the depth/amplitude projection below, applied to both components.
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let [u, v, ..] = self.medium[i];
                self.advected[i] =
                    transported(&self.state, x as f32 - u * dt / DX, y as f32 - v * dt / DY);
            }
        }
        // Velocity-Verlet is stable for this oscillator: the worst-depth CFL
        // c*dt*sqrt(dx^-2+dy^-2) is <0.25, well below its limit 1.
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let speed =
                    self.advected[i][1] + 0.5 * dt * force(&self.advected, &self.medium, x, y);
                self.predicted[i] = [self.advected[i][0] + dt * speed, speed];
            }
        }
        let damping = (-2.0 * KINEMATIC_VISCOSITY * K * K * dt).exp();
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                self.state[i] = [
                    self.predicted[i][0] * damping,
                    (self.predicted[i][1] + 0.5 * dt * force(&self.predicted, &self.medium, x, y))
                        * damping,
                ];
            }
        }
        self.remove_mean();
        // Apply one common energy-reducing scale AFTER mean removal. This keeps
        // the optical datum and every local depth/steepness bound together.
        let mut scale = 1.0_f32;
        for (value, medium) in self.state.iter().zip(&self.medium) {
            let limit = (0.04 * medium[3]).min(0.003);
            if value[0].abs() > limit {
                scale = scale.min(limit / value[0].abs());
            }
        }
        if scale < 1.0 {
            for value in &mut self.state {
                value[0] *= scale;
                value[1] *= scale;
            }
        }
    }

    pub fn sample(
        &self,
        point: [f32; 2],
        resolution: [usize; 2],
    ) -> f32 {
        if !self.active || !point.iter().all(|p| p.is_finite()) {
            return 0.0;
        }
        let weight = spectral_weight(
            PATCH_LENGTH / (WAVELENGTH * resolution[0].min(resolution[1]).max(1) as f32),
        );
        if weight == 0.0 {
            return 0.0;
        }
        let center = [point[0] * NX as f32 - 0.5, point[1] * NY as f32 - 0.5];
        let filter = [
            0.25 * NX as f32 / resolution[0].max(1) as f32,
            0.25 * NY as f32 / resolution[1].max(1) as f32,
        ];
        let mut height = 0.0;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                height += Footprint::new(center[0] + sx * filter[0], center[1] + sy * filter[1])
                    .sample(&self.state)[0];
            }
        }
        height * 0.25 * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medium(point: [f32; 2]) -> ([f32; 2], f32) {
        ([0.15 * (std::f32::consts::TAU * point[1]).sin(), 0.0], 0.16)
    }

    fn mode(mode_x: usize) -> Ripples {
        let mut field = Ripples::new();
        field.active = true;
        for y in 0..NY {
            for x in 0..NX {
                field.state[y * NX + x][0] = 0.0001
                    * (std::f32::consts::TAU * mode_x as f32 * (x as f32 + 0.5) / NX as f32).cos();
            }
        }
        field
    }

    fn coefficient(
        field: &Ripples,
        mode_x: usize,
    ) -> [f64; 2] {
        let mut result = [0.0; 2];
        for y in 0..NY {
            for x in 0..NX {
                let phase = std::f64::consts::TAU * mode_x as f64 * (x as f64 + 0.5) / NX as f64;
                let (s, c) = phase.sin_cos();
                result[0] += f64::from(field.state[y * NX + x][0]) * c * 2.0 / CELLS as f64;
                result[1] += f64::from(field.state[y * NX + x][0]) * s * 2.0 / CELLS as f64;
            }
        }
        result
    }

    fn energy(
        field: &Ripples,
        depth: f32,
    ) -> f64 {
        let c2 = (GRAVITY / K + SURFACE_TENSION / DENSITY * K) * (K * depth).tanh();
        let mut sum = 0.0_f64;
        for y in 0..NY {
            for x in 0..NX {
                let a = field.state[y * NX + x];
                let dx = (field.state[y * NX + (x + 1) % NX][0] - a[0]) / DX;
                let dy = (field.state[((y + 1) % NY) * NX + x][0] - a[0]) / DY;
                sum += f64::from(a[1] * a[1] + c2 * (dx * dx + dy * dy)) * 0.5;
            }
        }
        sum * f64::from(DX * DY)
    }

    #[test]
    fn impact_is_zero_mean_and_larger_size_has_a_wider_footprint() {
        let moment = |size| {
            let mut field = Ripples::new();
            field.deposit([0.5, 0.5], 0.24, size);
            assert!(field.active);
            let sum = field.state.iter().map(|p| f64::from(p[1])).sum::<f64>();
            assert!(
                sum.abs() < 1e-6,
                "optical impulse must not add a DC mode: {sum}"
            );
            let mut norm = 0.0;
            let mut weighted = 0.0;
            for y in 0..NY {
                for x in 0..NX {
                    let r2 = ((x as f64 + 0.5) * f64::from(DX) - 1.0).powi(2)
                        + ((y as f64 + 0.5) * f64::from(DY) - 1.0).powi(2);
                    let power = f64::from(field.state[y * NX + x][1]).powi(2);
                    norm += power;
                    weighted += power * r2;
                }
            }
            assert!(norm > 1e-8, "rain must excite real optical wave momentum");
            weighted / norm
        };
        assert!(moment(2.0) > 2.5 * moment(0.7));
    }

    #[test]
    fn rest_and_invalid_sources_cannot_invent_optical_waves() {
        let mut field = Ripples::new();
        for size in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            field.deposit([0.3, 0.4], 0.2, size);
        }
        field.deposit([f32::NAN, 0.2], 0.2, 1.0);
        field.advance(Duration::from_secs(1), medium);
        assert!(field.state.iter().all(|state| *state == [0.0; 2]));
        assert_eq!(field.sample([0.4, 0.3], [96, 64]), 0.0);
    }

    #[test]
    fn uniform_depth_modes_follow_dispersion_and_have_bounded_energy() {
        let depth = 0.16;
        let mut field = mode(10);
        let initial_energy = energy(&field, depth);
        let c2 = (GRAVITY / K + SURFACE_TENSION / DENSITY * K) * (K * depth).tanh();
        let discrete_k = 2.0 * (std::f32::consts::PI * 10.0 / NX as f32).sin() / DX;
        let omega = c2.sqrt() * discrete_k;
        for _ in 0..25 {
            field.advance(TICK, |_| ([0.0; 2], depth));
        }
        let expected = 0.0001
            * f64::from(omega * 0.2).cos()
            * f64::from((-2.0 * KINEMATIC_VISCOSITY * K * K * 0.2).exp());
        assert!(
            (coefficient(&field, 10)[0] - expected).abs() < 3e-7,
            "wave must propagate at finite-depth narrow-band frequency"
        );
        for _ in 0..225 {
            field.advance(TICK, |_| ([0.0; 2], depth));
            assert!(
                energy(&field, depth) <= initial_energy * 1.005,
                "unforced stable mode gained energy"
            );
        }
        assert!(energy(&field, depth) < initial_energy);
    }

    #[test]
    fn shallow_water_reduces_the_detail_phase_speed() {
        let mut shallow = mode(10);
        let mut deep = mode(10);
        shallow.advance(Duration::from_millis(80), |_| ([0.0; 2], 0.012));
        deep.advance(Duration::from_millis(80), |_| ([0.0; 2], 0.16));
        assert!(coefficient(&shallow, 10)[0] - coefficient(&deep, 10)[0] > 0.000015);
    }

    #[test]
    fn uniform_current_advects_the_wave_phase_in_metres_per_second() {
        let mut stationary = mode(3);
        let mut moving = mode(3);
        let dt = Duration::from_millis(64);
        stationary.advance(dt, |_| ([0.0; 2], 0.16));
        moving.advance(dt, |_| ([0.4, 0.0], 0.16));
        let a = coefficient(&stationary, 3);
        let b = coefficient(&moving, 3);
        let phase = b[1].atan2(b[0]) - a[1].atan2(a[0]);
        let expected = std::f64::consts::TAU * 3.0 / 2.0 * 0.4 * 0.064;
        assert!(
            (phase - expected).abs() < 0.004,
            "phase {phase}, expected {expected}"
        );
    }

    #[test]
    fn periodic_sampling_is_safe_just_below_zero() {
        let field = mode(3);
        for point in [[-f32::EPSILON, -f32::EPSILON], [-f32::MIN_POSITIVE, 0.0], [
            NX as f32, NY as f32,
        ]] {
            let sample = transported(&field.state, point[0], point[1]);
            assert!(sample.iter().all(|v| v.is_finite()));
            assert!((sample[0] - field.state[0][0]).abs() < 1e-7);
            let sample = Footprint::new(point[0], point[1]).sample(&field.state);
            assert!(sample.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn cubic_uniform_translation_cannot_amplify_any_fourier_mode() {
        for phase in 0..=32 {
            let t = phase as f32 / 32.0;
            let weights: [f32; 4] = std::array::from_fn(|tap| {
                let mut basis = [0.0; 4];
                basis[tap] = 1.0;
                cubic(basis, t)
            });
            for mode in 0..=128 {
                let angle = std::f64::consts::PI * mode as f64 / 128.0;
                let mut response = [0.0_f64; 2];
                for (tap, weight) in weights.iter().enumerate() {
                    let phase = (tap as f64 - 1.0) * angle;
                    response[0] += f64::from(*weight) * phase.cos();
                    response[1] += f64::from(*weight) * phase.sin();
                }
                assert!(response[0].hypot(response[1]) <= 1.0 + 2e-7);
            }
        }
    }

    #[test]
    fn current_transport_retains_the_selected_wave_band() {
        let mut stationary = Ripples::new();
        stationary.active = true;
        for y in 0..NY {
            for x in 0..NX {
                stationary.state[y * NX + x][0] =
                    0.0001 * (std::f32::consts::TAU * 11.0 * (y as f32 + 0.5) / NY as f32).cos();
            }
        }
        let mut moving = stationary.clone();
        stationary.advance(Duration::from_millis(200), |_| ([0.0; 2], 0.16));
        moving.advance(Duration::from_millis(200), |_| ([0.0, 0.4], 0.16));
        let amplitude = |field: &Ripples| {
            let mut coefficient = [0.0_f64; 2];
            for y in 0..NY {
                let phase = std::f64::consts::TAU * 11.0 * (y as f64 + 0.5) / NY as f64;
                coefficient[0] += f64::from(field.state[y * NX][0]) * phase.cos();
                coefficient[1] += f64::from(field.state[y * NX][0]) * phase.sin();
            }
            coefficient[0].hypot(coefficient[1])
        };
        let retention = amplitude(&moving) / amplitude(&stationary);
        assert!(
            retention > 0.80 && retention <= 1.005,
            "current erased or amplified the selected 0.182 m wave band: {retention}"
        );
    }

    #[test]
    fn storm_projection_preserves_zero_mean_and_local_depth_bounds() {
        let mut field = Ripples::new();
        for impact in 0..40 {
            field.deposit([0.1 + 0.01 * impact as f32, 0.4], 1.0, 3.0);
        }
        for _ in 0..50 {
            field.advance(TICK, |point| ([0.7, -0.3], 0.006 + 0.15 * point[0]));
            let mean = field.state.iter().map(|p| f64::from(p[0])).sum::<f64>() / CELLS as f64;
            assert!(mean.abs() < 1e-9);
            for (value, medium) in field.state.iter().zip(&field.medium) {
                assert!(value.iter().all(|v| v.is_finite()));
                assert!(value[0].abs() <= (0.04 * medium[3]).min(0.003) + 1e-8);
            }
        }
    }

    #[test]
    fn fixed_tick_partition_and_pause_are_exact() {
        let mut whole = Ripples::new();
        whole.deposit([0.45, 0.37], 0.3, 1.0);
        let mut split = whole.clone();
        whole.advance(Duration::from_millis(177), medium);
        for millis in [1, 15, 80, 81] {
            split.advance(Duration::from_millis(millis), medium);
        }
        assert_eq!(whole.state, split.state);
        assert_eq!(whole.pending, split.pending);
        let before = split.state.clone();
        split.advance(Duration::ZERO, medium);
        assert_eq!(split.state, before);
        assert!(whole.state.iter().any(|v| v[0].abs() > 1e-6));
    }

    #[test]
    fn minimum_diameter_sources_do_not_inject_grid_scale_aliases() {
        let mut field = Ripples::new();
        field.deposit([0.431, 0.381], 0.24, 0.25);
        let mut nyquist = 0.0_f64;
        let mut absolute = 0.0_f64;
        for y in 0..NY {
            for x in 0..NX {
                let value = f64::from(field.state[y * NX + x][1]);
                nyquist += if y % 2 == 0 { value } else { -value };
                absolute += value.abs();
            }
        }
        assert!(absolute > 0.0);
        let fraction = nyquist.abs() / absolute;
        assert!(
            fraction < 0.02,
            "subcell drop injected unresolved checkerboard pressure: {fraction}"
        );
    }

    #[test]
    fn optical_detail_is_filtered_when_the_viewport_cannot_resolve_it() {
        let mut field = Ripples::new();
        field.deposit([0.5, 0.5], 0.3, 1.0);
        field.advance(Duration::from_millis(48), |_| ([0.0; 2], 0.16));
        assert!(field.sample([0.5, 0.5], [180, 120]).abs() > 0.00005);
        assert_eq!(field.sample([0.5, 0.5], [4, 4]), 0.0);
    }
}
