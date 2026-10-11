//! A pump-fed, depth-averaged whirlpool on a fixed physical grid.
//!
//! The pressure bowl, inward flow and advected vorticity belong to one shallow-
//! water state. A resolved suction disk removes water and local momentum; an
//! outer feed supplies the same accepted volume with tangential momentum.
//! The pump, quadratic bottom drag and localized porous-intake resistance are
//! explicit external momentum sources. The latter represents an intake/grate,
//! removing angular momentum before unresolved vertical suction dries the core.
//! A rough outer return reservoir damps bulk momentum before the periodic seam;
//! it removes no water or pigment and does not impose a moving velocity target.
//! This is not a resolved drain pipe, air core, or three-dimensional turbulence.
use std::time::Duration;

use crate::{
    dye::{
        Dye,
        HydraulicStage,
        NX,
        NY,
    },
    rapids::{
        interface,
        minmod,
    },
};

const LENGTH: f32 = 2.0;
const DX: f32 = LENGTH / NX as f32;
const DY: f32 = LENGTH / NY as f32;
const CELLS: usize = NX * NY;
const REFERENCE: f32 = 0.16;
const G: f32 = 9.81;
pub(crate) const TICK: Duration = Duration::from_millis(4);
// Leave room for wave-speed changes in the SSP predictor. Both stages still
// enforce their original CFL limits; this avoids trial steps on the boundary.
const CFL_RESERVE: f32 = 0.90;
const MIN_DEPTH: f32 = 0.005;
const DISCHARGE: f32 = 0.008;
// For this compact cubic kernel the RMS uptake radius is R/sqrt(5).
// R=.24 resolves that radius with >3 coarse-y cells. Localized intake resistance
// supplies the additional angular-momentum loss needed for a wet finite core.
const DRAIN_RADIUS: f32 = 0.24;
const CORE_RADIUS: f32 = 0.14;
const PEAK_SWIRL: f32 = 0.55;
const FEED_RADIUS: f32 = 0.74;
const FEED_WIDTH: f32 = 0.16;
const FEED_SPEED: f32 = 0.40;
const BED_DRAG: f32 = 0.008;
// Empirical Darcy resistance of the unresolved porous intake, in s^-1.
const INTAKE_RESISTANCE: f32 = 2.5;
const RIM_RESISTANCE: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
    h: f32,
    mx: f32,
    my: f32,
}

impl Cell {
    fn primitive(self) -> [f32; 3] { [self.h, self.mx / self.h, self.my / self.h] }

    fn valid(self) -> bool {
        self.h >= MIN_DEPTH && self.h.is_finite() && self.mx.is_finite() && self.my.is_finite()
    }
}

#[derive(Debug)]
struct Stage {
    depth: Vec<f32>,
    flux_x: Vec<[f32; 3]>,
    flux_y: Vec<[f32; 3]>,
    discharge_x: Vec<f32>,
    discharge_y: Vec<f32>,
    drain: Vec<f32>,
    feed: Vec<f32>,
    max_dt: f32,
}

impl Stage {
    fn new() -> Self {
        Self {
            depth: vec![0.0; CELLS],
            flux_x: vec![[0.0; 3]; (NX + 1) * NY],
            flux_y: vec![[0.0; 3]; NX * (NY + 1)],
            discharge_x: vec![0.0; (NX + 1) * NY],
            discharge_y: vec![0.0; NX * (NY + 1)],
            drain: vec![0.0; CELLS],
            feed: vec![0.0; CELLS],
            max_dt: 0.0,
        }
    }

    fn pigment_stage(&self) -> HydraulicStage<'_> {
        HydraulicStage {
            depth: &self.depth,
            discharge_x: &self.discharge_x,
            discharge_y: &self.discharge_y,
            drain: &self.drain,
            feed: &self.feed,
            feed_concentration: [0.0; 3],
        }
    }
}

#[derive(Debug)]
struct Workspace {
    primitive: Vec<[f32; 3]>,
    slope_x: Vec<[f32; 3]>,
    slope_y: Vec<[f32; 3]>,
}

impl Workspace {
    fn new() -> Self {
        Self {
            primitive: vec![[0.0; 3]; CELLS],
            slope_x: vec![[0.0; 3]; CELLS],
            slope_y: vec![[0.0; 3]; CELLS],
        }
    }
}

#[derive(Debug)]
struct Pump {
    sink_kernel: Vec<f32>,
    feed_kernel: Vec<f32>,
    feed_velocity: Vec<[f32; 2]>,
    resistance: Vec<f32>,
    direction: Vec<[f32; 2]>,
    noise: [f32; 4],
    random: u32,
    spin: f32,
}

fn smoothstep(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

fn quintic(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
}

fn compact(q: f32) -> f32 { (1.0 - q).max(0.0).powi(3) }

fn tangential_velocity(
    r: f32,
    spin: f32,
) -> f32 {
    let window = 1.0 - quintic((r - 0.72) / 0.20);
    2.0 * CORE_RADIUS * PEAK_SWIRL * spin * r / (r * r + CORE_RADIUS * CORE_RADIUS) * window
}

fn normalize_kernel(kernel: &mut [f32]) {
    let integral = kernel.iter().map(|&v| v as f64).sum::<f64>() * (DX * DY) as f64;
    for weight in kernel {
        *weight = (*weight as f64 / integral) as f32;
    }
}

impl Pump {
    fn new(spin: f32) -> Self {
        let mut pump = Self {
            sink_kernel: vec![0.0; CELLS],
            feed_kernel: vec![0.0; CELLS],
            feed_velocity: vec![[0.0; 2]; CELLS],
            resistance: vec![0.0; CELLS],
            direction: vec![[0.0; 2]; CELLS],
            noise: [0.0; 4],
            random: 0x1794_BDCA,
            spin,
        };
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let px = (x as f32 + 0.5) * DX - 1.0;
                let py = (y as f32 + 0.5) * DY - 1.0;
                let r = px.hypot(py);
                let direction = [px / r, py / r];
                pump.direction[i] = direction;
                pump.sink_kernel[i] = compact(r * r / (DRAIN_RADIUS * DRAIN_RADIUS));
                pump.resistance[i] = INTAKE_RESISTANCE * pump.sink_kernel[i]
                    + RIM_RESISTANCE * quintic((r - 0.82) / 0.15);
                // A fixed uneven feed supplies shear. With no angular input the
                // pump is axisymmetric and cannot choose a circulation sign.
                let cos2 = direction[0] * direction[0] - direction[1] * direction[1];
                let sin3 = direction[1] * (3.0 - 4.0 * direction[1] * direction[1]);
                pump.feed_kernel[i] = compact(((r - FEED_RADIUS) / FEED_WIDTH).powi(2))
                    * (1.0 + spin.abs() * (0.40 * cos2 + 0.16 * sin3));
            }
        }
        normalize_kernel(&mut pump.sink_kernel);
        normalize_kernel(&mut pump.feed_kernel);
        pump.resolve_feed();
        pump
    }

    fn resolve_feed(&mut self) {
        let mut mean = [0.0_f64; 2];
        let weight = self.feed_kernel.iter().map(|&w| w as f64).sum::<f64>();
        for (i, direction) in self.direction.iter().enumerate() {
            let [x, y] = *direction;
            let sin3 = y * (3.0 - 4.0 * y * y);
            let cos3 = x * (4.0 * x * x - 3.0);
            let perturbation =
                self.noise[0] * x + self.noise[1] * y + self.noise[2] * cos3 + self.noise[3] * sin3;
            let speed = FEED_SPEED * self.spin * (1.0 + 0.045 * perturbation);
            self.feed_velocity[i] = [-speed * y, speed * x];
            for (axis, value) in mean.iter_mut().enumerate() {
                *value += self.feed_velocity[i][axis] as f64 * self.feed_kernel[i] as f64;
            }
        }
        // The pump supplies torque through opposed feed sectors, not a net
        // horizontal jet. Remove the weighted translational component introduced
        // by asymmetric inlet fluctuations; this retains their local shear.
        let mean = mean.map(|v| (v / weight) as f32);
        for velocity in &mut self.feed_velocity {
            for axis in 0..2 {
                velocity[axis] -= mean[axis];
            }
        }
    }

    fn tick(&mut self) {
        // Smooth, seeded inlet fluctuations have a 0.65 s physical correlation
        // time. They perturb supplied momentum, never prescribe vortex paths.
        let decay = (-TICK.as_secs_f32() / 0.65).exp();
        let innovation = (1.0 - decay * decay).sqrt() * 3.0_f32.sqrt();
        for value in &mut self.noise {
            self.random ^= self.random << 13;
            self.random ^= self.random >> 17;
            self.random ^= self.random << 5;
            let random = (self.random >> 8) as f32 / 16_777_216.0;
            *value = (*value * decay + (2.0 * random - 1.0) * innovation).clamp(-2.5, 2.5);
        }
        self.resolve_feed();
    }
}

fn prepare_stage(
    cells: &[Cell],
    stage: &mut Stage,
    work: &mut Workspace,
    pump: &Pump,
) {
    let mut wave_rate = 0.0_f32;
    let mut removed = 0.0_f64;
    for (i, cell) in cells.iter().enumerate() {
        let primitive = cell.primitive();
        stage.depth[i] = cell.h;
        work.primitive[i] = primitive;
        let wave = (G * cell.h).sqrt();
        wave_rate =
            wave_rate.max((primitive[1].abs() + wave) / DX + (primitive[2].abs() + wave) / DY);
        // Reduce the accepted suction smoothly before a column could become dry.
        stage.drain[i] = DISCHARGE * pump.sink_kernel[i] * smoothstep((cell.h - 0.015) / 0.045);
        removed += stage.drain[i] as f64 * (DX * DY) as f64;
    }
    for (feed, &weight) in stage.feed.iter_mut().zip(&pump.feed_kernel) {
        *feed = (removed * weight as f64) as f32;
    }
    for y in 0..NY {
        for x in 0..NX {
            let i = y * NX + x;
            let west = y * NX + (x + NX - 1) % NX;
            let east = y * NX + (x + 1) % NX;
            let south = ((y + NY - 1) % NY) * NX + x;
            let north = ((y + 1) % NY) * NX + x;
            for (a, b, slope) in [
                (west, east, &mut work.slope_x[i]),
                (south, north, &mut work.slope_y[i]),
            ] {
                for (k, out) in slope.iter_mut().enumerate() {
                    let center = work.primitive[i][k];
                    *out = minmod(
                        1.3 * (center - work.primitive[a][k]),
                        0.5 * (work.primitive[b][k] - work.primitive[a][k]),
                        1.3 * (work.primitive[b][k] - center),
                    );
                }
                slope[0] = slope[0].clamp(-1.8 * cells[i].h, 1.8 * cells[i].h);
            }
        }
    }
    for y in 0..NY {
        for x in 0..NX {
            let i = y * NX + x;
            for axis in 0..2 {
                let (j, slopes, face) = if axis == 0 {
                    (y * NX + (x + 1) % NX, &work.slope_x, y * (NX + 1) + x + 1)
                } else {
                    (((y + 1) % NY) * NX + x, &work.slope_y, (y + 1) * NX + x)
                };
                let mut left = work.primitive[i];
                let mut right = work.primitive[j];
                for k in 0..3 {
                    left[k] += 0.5 * slopes[i][k];
                    right[k] -= 0.5 * slopes[j][k];
                }
                let (flux, ..) = interface(left, right, 0.0, 0.0, axis);
                if axis == 0 {
                    stage.flux_x[face] = flux;
                    stage.discharge_x[face] = flux[0];
                } else {
                    stage.flux_y[face] = flux;
                    stage.discharge_y[face] = flux[0];
                }
            }
        }
    }
    for y in 0..NY {
        stage.flux_x[y * (NX + 1)] = stage.flux_x[y * (NX + 1) + NX];
        stage.discharge_x[y * (NX + 1)] = stage.discharge_x[y * (NX + 1) + NX];
    }
    for x in 0..NX {
        stage.flux_y[x] = stage.flux_y[NY * NX + x];
        stage.discharge_y[x] = stage.discharge_y[NY * NX + x];
    }
    let mut material_rate = 0.0_f32;
    let mut damping_rate = 0.0_f32;
    for y in 0..NY {
        for x in 0..NX {
            let i = y * NX + x;
            let outgoing = (stage.discharge_x[y * (NX + 1) + x + 1].max(0.0)
                - stage.discharge_x[y * (NX + 1) + x].min(0.0))
                / DX
                + (stage.discharge_y[(y + 1) * NX + x].max(0.0)
                    - stage.discharge_y[y * NX + x].min(0.0))
                    / DY;
            material_rate = material_rate.max((outgoing + stage.drain[i]) / cells[i].h);
            damping_rate = damping_rate.max(
                stage.drain[i] / cells[i].h
                    + pump.resistance[i]
                    + BED_DRAG * cells[i].mx.hypot(cells[i].my) / (cells[i].h * cells[i].h),
            );
        }
    }
    // The 0.40 material bound leaves room for MC face concentration <=2*C.
    stage.max_dt = (0.34 / wave_rate.max(1e-6))
        .min(0.40 / material_rate.max(1e-6))
        .min(0.40 / damping_rate.max(1e-6));
}

fn euler(
    cells: &[Cell],
    stage: &Stage,
    pump: &Pump,
    dt: f32,
    output: &mut [Cell],
) {
    for y in 0..NY {
        for x in 0..NX {
            let i = y * NX + x;
            let cell = cells[i];
            let left = stage.flux_x[y * (NX + 1) + x];
            let right = stage.flux_x[y * (NX + 1) + x + 1];
            let lower = stage.flux_y[y * NX + x];
            let upper = stage.flux_y[(y + 1) * NX + x];
            let flux = std::array::from_fn::<_, 3, _>(|k| {
                (right[k] - left[k]) / DX + (upper[k] - lower[k]) / DY
            });
            let u = cell.mx / cell.h;
            let v = cell.my / cell.h;
            let drag = BED_DRAG * u.hypot(v);
            output[i] = Cell {
                h: cell.h + dt * (stage.feed[i] - stage.drain[i] - flux[0]),
                mx: cell.mx
                    + dt * (stage.feed[i] * pump.feed_velocity[i][0]
                        - stage.drain[i] * u
                        - pump.resistance[i] * cell.mx
                        - drag * u
                        - flux[1]),
                my: cell.my
                    + dt * (stage.feed[i] * pump.feed_velocity[i][1]
                        - stage.drain[i] * v
                        - pump.resistance[i] * cell.my
                        - drag * v
                        - flux[2]),
            };
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DrainSample {
    /// Free-surface height relative to the 0.16 m reference level, in metres.
    pub elevation: f32,
    pub depth: f32,
    pub foam: f32,
    pub velocity: [f32; 2],
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct StepCounts {
    ticks: usize,
    accepted: usize,
    attempts: usize,
    predictor_retries: usize,
}

#[derive(Debug)]
pub struct Drain {
    cells: Vec<Cell>,
    stage_cells: Vec<Cell>,
    next_cells: Vec<Cell>,
    stage0: Stage,
    stage1: Stage,
    work: Workspace,
    pump: Pump,
    drained_volume: f64,
    fed_volume: f64,
    pending: Duration,
    #[cfg(test)]
    step_counts: StepCounts,
}

impl Default for Drain {
    fn default() -> Self { Self::new() }
}

impl Drain {
    pub fn new() -> Self { Self::with_spin(1.0) }

    fn with_spin(spin: f32) -> Self {
        // Integrate d(eta)/dr=v_theta^2/(g*r), including the compact outer
        // taper, so the initial bowl and rotating velocity share a pressure law.
        const RADIAL: usize = 512;
        let dr = 0.92 / RADIAL as f32;
        let mut depression = [0.0; RADIAL + 1];
        for i in (0..RADIAL).rev() {
            let r = (i as f32 + 0.5) * dr;
            depression[i] = depression[i + 1] + dr * tangential_velocity(r, spin).powi(2) / (G * r);
        }
        let mut cells = vec![Cell::default(); CELLS];
        let mut mean_depression = 0.0_f64;
        for y in 0..NY {
            for x in 0..NX {
                let px = (x as f32 + 0.5) * DX - 1.0;
                let py = (y as f32 + 0.5) * DY - 1.0;
                let r = px.hypot(py);
                let q = (r / dr).min(RADIAL as f32);
                let lo = (q as usize).min(RADIAL - 1);
                let dip = depression[lo] + (depression[lo + 1] - depression[lo]) * (q - lo as f32);
                mean_depression += dip as f64 / CELLS as f64;
                cells[y * NX + x].h = REFERENCE - dip;
            }
        }
        for (i, cell) in cells.iter_mut().enumerate() {
            cell.h += mean_depression as f32;
            let px = (i % NX) as f32 * DX + 0.5 * DX - 1.0;
            let py = (i / NX) as f32 * DY + 0.5 * DY - 1.0;
            let r = px.hypot(py);
            let speed = tangential_velocity(r, spin);
            cell.mx = -cell.h * speed * py / r;
            cell.my = cell.h * speed * px / r;
        }
        Self {
            cells,
            stage_cells: vec![Cell::default(); CELLS],
            next_cells: vec![Cell::default(); CELLS],
            stage0: Stage::new(),
            stage1: Stage::new(),
            work: Workspace::new(),
            pump: Pump::new(spin),
            drained_volume: 0.0,
            fed_volume: 0.0,
            pending: Duration::ZERO,
            #[cfg(test)]
            step_counts: StepCounts::default(),
        }
    }

    pub fn advance(
        &mut self,
        duration: Duration,
        dye: &mut Dye,
    ) {
        self.pending = self.pending.saturating_add(duration);
        while self.pending >= TICK {
            self.pending -= TICK;
            #[cfg(test)]
            {
                self.step_counts.ticks += 1;
            }
            self.pump.tick();
            let mut remaining = TICK.as_secs_f32();
            let mut substeps = 0;
            while remaining > 1e-8 {
                substeps += 1;
                assert!(substeps <= 64, "drain hydraulic substep budget exhausted");
                prepare_stage(&self.cells, &mut self.stage0, &mut self.work, &self.pump);
                let mut dt = remaining.min(CFL_RESERVE * self.stage0.max_dt);
                let mut accepted = false;
                for _ in 0..16 {
                    #[cfg(test)]
                    {
                        self.step_counts.attempts += 1;
                    }
                    euler(
                        &self.cells,
                        &self.stage0,
                        &self.pump,
                        dt,
                        &mut self.stage_cells,
                    );
                    if self.stage_cells.iter().all(|cell| cell.valid()) {
                        prepare_stage(
                            &self.stage_cells,
                            &mut self.stage1,
                            &mut self.work,
                            &self.pump,
                        );
                        if dt <= self.stage1.max_dt {
                            euler(
                                &self.stage_cells,
                                &self.stage1,
                                &self.pump,
                                dt,
                                &mut self.next_cells,
                            );
                            if self.next_cells.iter().all(|cell| cell.valid())
                                && dye.prepare_hydraulic_step(
                                    dt,
                                    self.stage0.pigment_stage(),
                                    self.stage1.pigment_stage(),
                                )
                            {
                                for (cell, next) in self.cells.iter_mut().zip(&self.next_cells) {
                                    cell.h = 0.5 * (cell.h + next.h);
                                    cell.mx = 0.5 * (cell.mx + next.mx);
                                    cell.my = 0.5 * (cell.my + next.my);
                                }
                                let measure = 0.5 * dt as f64 * (DX * DY) as f64;
                                self.drained_volume += measure
                                    * self
                                        .stage0
                                        .drain
                                        .iter()
                                        .chain(&self.stage1.drain)
                                        .map(|&s| s as f64)
                                        .sum::<f64>();
                                self.fed_volume += measure
                                    * self
                                        .stage0
                                        .feed
                                        .iter()
                                        .chain(&self.stage1.feed)
                                        .map(|&s| s as f64)
                                        .sum::<f64>();
                                dye.commit_hydraulic_step();
                                #[cfg(test)]
                                {
                                    self.step_counts.accepted += 1;
                                }
                                accepted = true;
                                break;
                            }
                        } else {
                            #[cfg(test)]
                            {
                                self.step_counts.predictor_retries += 1;
                            }
                            // The predictor already supplies the tighter limit.
                            // A small CFL change need not halve the entire step;
                            // recalculate and validate every stage at the new dt.
                            dt = CFL_RESERVE * self.stage1.max_dt;
                            continue;
                        }
                    }
                    dt *= 0.5;
                }
                assert!(
                    accepted,
                    "drain step failed: dt={dt:e}, h0min={:?}, h1min={:?}, h2min={:?}, \
                     maxdt0={:?}, maxdt1={:?}",
                    self.cells.iter().map(|c| c.h).reduce(f32::min),
                    self.stage_cells.iter().map(|c| c.h).reduce(f32::min),
                    self.next_cells.iter().map(|c| c.h).reduce(f32::min),
                    self.stage0.max_dt,
                    self.stage1.max_dt,
                );
                remaining -= dt;
            }
        }
    }

    pub fn sample(
        &self,
        point: [f32; 2],
    ) -> DrainSample {
        if point.iter().any(|p| !p.is_finite()) {
            return DrainSample {
                depth: REFERENCE,
                ..DrainSample::default()
            };
        }
        let x = point[0].rem_euclid(1.0) * NX as f32 - 0.5;
        let y = point[1].rem_euclid(1.0) * NY as f32 - 0.5;
        let ix = x.floor() as isize;
        let iy = y.floor() as isize;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let mut sample = Cell::default();
        for (col, wx) in [(ix, 1.0 - fx), (ix + 1, fx)] {
            for (row, wy) in [(iy, 1.0 - fy), (iy + 1, fy)] {
                let cell = self.cells[row.rem_euclid(NY as isize) as usize * NX
                    + col.rem_euclid(NX as isize) as usize];
                let weight = wx * wy;
                sample.h += weight * cell.h;
                sample.mx += weight * cell.mx;
                sample.my += weight * cell.my;
            }
        }
        DrainSample {
            elevation: sample.h - REFERENCE,
            depth: sample.h,
            // Subcritical suction does not justify manufacturing whitewater.
            foam: 0.0,
            velocity: [sample.mx / sample.h, sample.my / sample.h],
        }
    }

    pub fn disturb_sized(
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
        let point = point.map(|p| p.rem_euclid(1.0) * LENGTH);
        let sigma = (0.055 + 0.035 * strength.min(1.0).sqrt()) * size.clamp(0.25, 3.0);
        let impulse = 0.08 * strength.min(1.0);
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let x = ((i % NX) as f32 * DX + 0.5 * DX - point[0] + 1.0).rem_euclid(LENGTH) - 1.0;
            let y = ((i / NX) as f32 * DY + 0.5 * DY - point[1] + 1.0).rem_euclid(LENGTH) - 1.0;
            let q = (x * x + y * y) / (sigma * sigma);
            if q <= 20.0 {
                let weight = impulse * (-0.5 * q).exp() / sigma;
                cell.mx += cell.h * weight * x;
                cell.my += cell.h * weight * y;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(flow: &Drain) -> f64 {
        flow.cells.iter().map(|cell| cell.h as f64).sum::<f64>() * (DX * DY) as f64
    }

    fn angular_momentum(flow: &Drain) -> f64 {
        flow.cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let x = (i % NX) as f64 * DX as f64 + 0.5 * DX as f64 - 1.0;
                let y = (i / NX) as f64 * DY as f64 + 0.5 * DY as f64 - 1.0;
                x * cell.my as f64 - y * cell.mx as f64
            })
            .sum::<f64>()
            * (DX * DY) as f64
    }

    #[test]
    fn initialized_whirlpool_has_a_wet_finite_pressure_bowl() {
        let flow = Drain::new();
        let core = flow.sample([0.5, 0.5]);
        let rim = flow.sample([0.85, 0.5]);
        let turn = flow.sample([0.57, 0.5]);
        assert!(rim.elevation - core.elevation > 0.035);
        assert!(core.depth > 0.035 && core.depth < REFERENCE);
        assert!(turn.velocity[1] > 0.40);
        assert!(core.velocity[0].abs() < 1e-5 && core.velocity[1].abs() < 1e-5);
        assert!((volume(&flow) - (REFERENCE * LENGTH * LENGTH) as f64).abs() < 1e-6);
    }

    #[test]
    fn accepted_pump_flow_conserves_volume_and_keeps_every_column_wet() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        let initial = volume(&flow);
        flow.advance(Duration::from_millis(240), &mut dye);
        assert!(flow.drained_volume > 0.001);
        let residual = volume(&flow) - initial - flow.fed_volume + flow.drained_volume;
        assert!(residual.abs() < 2e-6, "volume residual {residual}");
        assert!(
            flow.cells
                .iter()
                .all(|c| c.h > 0.01 && c.h.is_finite() && c.mx.is_finite() && c.my.is_finite())
        );
        assert!(flow.sample([0.60, 0.5]).velocity[0] < -0.001);
    }

    #[test]
    fn symmetric_suction_without_angular_feed_does_not_invent_spin() {
        let mut flow = Drain::with_spin(0.0);
        let mut dye = Dye::new();
        flow.advance(Duration::from_millis(300), &mut dye);
        assert!(flow.drained_volume > 0.001);
        assert!(angular_momentum(&flow).abs() < 1e-7);
        assert!(flow.sample([0.6, 0.5]).velocity[1].abs() < 1e-5);
    }

    #[test]
    fn physical_tick_partition_is_exact() {
        let mut whole = Drain::new();
        let mut split = Drain::new();
        let mut dye_a = Dye::new();
        let mut dye_b = Dye::new();
        for dye in [&mut dye_a, &mut dye_b] {
            dye.deposit([0.56, 0.51], 0.035, 0.006, [0.2, 0.7, 0.1]);
        }
        whole.advance(Duration::from_millis(177), &mut dye_a);
        for millis in [1, 15, 80, 81] {
            split.advance(Duration::from_millis(millis), &mut dye_b);
        }
        assert!(whole.drained_volume > 0.0);
        assert_eq!(whole.cells, split.cells);
        assert_eq!(whole.pending, split.pending);
        assert_eq!(whole.drained_volume, split.drained_volume);
        assert_eq!(whole.fed_volume, split.fed_volume);
        for y in 0..NY {
            for x in 0..NX {
                let point = [(x as f32 + 0.5) / NX as f32, (y as f32 + 0.5) / NY as f32];
                assert_eq!(dye_a.sample(point), dye_b.sample(point));
            }
        }
    }

    #[test]
    fn cfl_planning_avoids_repeated_predictor_work_without_dropping_time() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        dye.deposit([0.56, 0.51], 0.035, 0.006, [0.2, 0.7, 0.1]);
        flow.advance(Duration::from_secs(1), &mut dye);
        flow.disturb_sized([0.58, 0.53], 0.3, 1.0);
        flow.step_counts = StepCounts::default();
        let initial = volume(&flow);
        let drained = flow.drained_volume;
        let fed = flow.fed_volume;
        flow.advance(Duration::from_millis(1003), &mut dye);
        let counts = flow.step_counts;
        eprintln!("hydraulic work: {counts:?}");
        assert_eq!(counts.ticks, 250);
        assert_eq!(flow.pending, Duration::from_millis(3));
        assert!(flow.cells.iter().all(|c| c.valid() && c.h > 0.02));
        let residual =
            volume(&flow) - initial - (flow.fed_volume - fed) + (flow.drained_volume - drained);
        assert!(residual.abs() < 2e-6, "volume residual {residual}");
        assert!((flow.drained_volume - drained - DISCHARGE as f64).abs() < 1e-6);
        // This resolved, wet flow fits two CFL-safe substeps per 4 ms tick.
        // Count actual solver work, never elapsed wall time or mocked callbacks.
        assert!(counts.predictor_retries <= counts.ticks / 20, "{counts:?}");
        assert!(counts.accepted <= counts.ticks * 21 / 10, "{counts:?}");
        assert!(counts.attempts <= counts.ticks * 22 / 10, "{counts:?}");
    }

    #[test]
    fn planned_steps_agree_with_a_small_step_hydraulic_reference() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        flow.advance(Duration::from_secs(1), &mut dye);
        flow.disturb_sized([0.58, 0.53], 0.3, 1.0);
        let mut reference = flow.cells.clone();
        flow.advance(TICK, &mut dye);

        // The pump is fixed within this physical tick. Evolve the same initial
        // water with sixteen independently checked SSP steps, not a saved image
        // or a second copy of the adaptive step-size controller.
        let mut stage = Stage::new();
        let mut workspace = Workspace::new();
        let mut predictor = vec![Cell::default(); CELLS];
        let mut next = vec![Cell::default(); CELLS];
        let dt = TICK.as_secs_f32() / 16.0;
        for _ in 0..16 {
            prepare_stage(&reference, &mut stage, &mut workspace, &flow.pump);
            assert!(dt <= stage.max_dt);
            euler(&reference, &stage, &flow.pump, dt, &mut predictor);
            assert!(predictor.iter().all(|cell| cell.valid()));
            prepare_stage(&predictor, &mut stage, &mut workspace, &flow.pump);
            assert!(dt <= stage.max_dt);
            euler(&predictor, &stage, &flow.pump, dt, &mut next);
            assert!(next.iter().all(|cell| cell.valid()));
            for (cell, next) in reference.iter_mut().zip(&next) {
                cell.h = 0.5 * (cell.h + next.h);
                cell.mx = 0.5 * (cell.mx + next.mx);
                cell.my = 0.5 * (cell.my + next.my);
            }
        }
        let mut depth_error = 0.0_f64;
        let mut velocity_error = 0.0_f64;
        for (cell, reference) in flow.cells.iter().zip(&reference) {
            depth_error += (cell.h as f64 - reference.h as f64).powi(2);
            velocity_error += (cell.mx / cell.h - reference.mx / reference.h).powi(2) as f64
                + (cell.my / cell.h - reference.my / reference.h).powi(2) as f64;
        }
        let depth_rms = (depth_error / CELLS as f64).sqrt();
        let velocity_rms = (velocity_error / CELLS as f64).sqrt();
        eprintln!(
            "small-step reference: depth RMS {depth_rms:e} m, velocity RMS {velocity_rms:e} m/s"
        );
        assert!(depth_rms < 1e-5, "depth RMS {depth_rms:e} m");
        assert!(velocity_rms < 2e-4, "velocity RMS {velocity_rms:e} m/s");
    }

    #[test]
    fn pigment_above_the_suction_leaves_through_the_same_water_sink() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        dye.set_bleach(0.0);
        dye.deposit([0.5, 0.5], 0.025, 0.005, [0.2, 0.7, 0.1]);
        let mass = |dye: &Dye| {
            let mut total = 0.0_f64;
            for y in 0..NY {
                for x in 0..NX {
                    let sample =
                        dye.sample([(x as f32 + 0.5) / NX as f32, (y as f32 + 0.5) / NY as f32]);
                    total += sample.iter().map(|&v| v as f64).sum::<f64>() * (DX * DY) as f64;
                }
            }
            total
        };
        let initial = mass(&dye);
        flow.advance(Duration::from_millis(240), &mut dye);
        let remaining = mass(&dye);
        assert!(
            remaining > 0.0 && remaining < initial * 0.80,
            "{remaining} / {initial}"
        );
    }

    #[test]
    fn sustained_suction_keeps_a_resolved_wet_core() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        flow.advance(Duration::from_secs(25), &mut dye);
        assert!(flow.drained_volume > 0.02);
        assert!(
            flow.cells
                .iter()
                .all(|c| c.valid() && c.mx.hypot(c.my) / c.h < 2.5)
        );
        assert!(flow.sample([0.6, 0.5]).velocity[0] < -0.003);
    }

    #[test]
    fn repeated_large_impacts_preserve_wet_depth_and_water_budget() {
        let mut flow = Drain::new();
        let mut dye = Dye::new();
        let initial = volume(&flow);
        let mut random = 0x631A_CD09_u32;
        for tick in 0..200 {
            let drops = if tick % 25 == 0 { 25 } else { 1 };
            for _ in 0..drops {
                let point = std::array::from_fn(|_| {
                    random ^= random << 13;
                    random ^= random >> 17;
                    random ^= random << 5;
                    (random >> 8) as f32 / 16_777_216.0
                });
                flow.disturb_sized(point, 0.36, 3.0);
            }
            flow.advance(Duration::from_millis(20), &mut dye);
        }
        assert!(flow.cells.iter().all(|c| c.valid() && c.h > 0.02));
        assert!((volume(&flow) - initial - flow.fed_volume + flow.drained_volume).abs() < 3e-6);
    }

    #[test]
    fn annular_feed_injects_torque_without_a_uniform_horizontal_push() {
        let mut pump = Pump::new(1.0);
        pump.noise = [1.2, -0.7, 0.9, 1.5];
        pump.resolve_feed();
        let mut impulse = [0.0_f64; 2];
        let mut torque = 0.0_f64;
        for (i, velocity) in pump.feed_velocity.iter().enumerate() {
            let weight = pump.feed_kernel[i] as f64 * (DX * DY) as f64;
            for axis in 0..2 {
                impulse[axis] += weight * velocity[axis] as f64;
            }
            let x = ((i % NX) as f64 + 0.5) * DX as f64 - 1.0;
            let y = ((i / NX) as f64 + 0.5) * DY as f64 - 1.0;
            torque += weight * (x * velocity[1] as f64 - y * velocity[0] as f64);
        }
        assert!(
            impulse[0].abs() < 1e-7 && impulse[1].abs() < 1e-7,
            "net feed velocity {impulse:?}"
        );
        assert!(torque > 0.20);
    }

    #[test]
    fn impact_adds_finite_local_momentum_without_adding_water() {
        let mut flow = Drain::new();
        let before = flow.cells.clone();
        flow.disturb_sized([0.97, 0.45], 0.3, 1.8);
        assert!(
            flow.cells
                .iter()
                .zip(&before)
                .any(|(a, b)| (a.mx - b.mx).abs() > 1e-5)
        );
        assert!(
            flow.cells
                .iter()
                .zip(&before)
                .all(|(a, b)| a.h == b.h && a.mx.is_finite() && a.my.is_finite())
        );
        assert_eq!(flow.drained_volume, 0.0);
    }
}
