//! Conservative transport of three passive, miscible pigment species.
//!
//! This is a 2 m square column-integrated field, independent of terminal resolution. The
//! caller supplies water velocity in m/s: no motion is invented by this module.
//! A limited finite-volume reconstruction and SSP-RK2 transport are conservative
//! and positivity preserving under the combined advection/diffusion CFL limit.
//! Strain-dependent Smagorinsky diffusivity models unresolved stirring; it is
//! a subgrid closure, not resolved three-dimensional turbulence. Transport and
//! mixing conserve pigment except for open outflow. Separately configurable
//! first-order chromophore bleaching turns colored species into optically
//! untracked colorless products under uniform exposure. This phenomenological
//! reaction is not a calibrated model of real acid or a transport mass sink.
use std::time::Duration;

pub(crate) const NX: usize = 96;
pub(crate) const NY: usize = 64;
const LENGTH: f32 = 2.0;
const DX: f32 = LENGTH / NX as f32;
const DY: f32 = LENGTH / NY as f32;
pub(crate) const TICK: Duration = Duration::from_millis(8);
const MOLECULAR_DIFFUSIVITY: f32 = 1e-9;
const MAX_EDDY_DIFFUSIVITY: f32 = 0.0015;
const CELLS: usize = NX * NY;
pub(crate) const DEFAULT_BLEACH: f32 = 1.0;
pub(crate) const MAX_BLEACH: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Closed periodic transport; configured bleaching remains independent.
    Periodic,
    /// Clear incoming water and advective outflow at x; periodic at y.
    OpenChannel,
    /// Clear incoming water and advective outflow on all four sides.
    OpenPlane,
}

/// One accepted shallow-water Euler stage on the shared 96 × 64 grid.
/// Cell data are row-major x-fast; x faces have (NX+1)×NY entries and y faces
/// NX×(NY+1), including matching periodic seam faces. Discharges are m²/s,
/// sink/feed are nonnegative water-depth rates in m/s, and depth is in metres.
pub(crate) struct HydraulicStage<'a> {
    pub depth: &'a [f32],
    pub discharge_x: &'a [f32],
    pub discharge_y: &'a [f32],
    pub drain: &'a [f32],
    pub feed: &'a [f32],
    pub feed_concentration: [f32; 3],
}

#[derive(Debug, Default)]
struct HydraulicLedger {
    drained: [f64; 3],
    fed: [f64; 3],
    bleached: [f64; 3],
    active: bool,
}

impl HydraulicStage<'_> {
    /// The hydraulic solver must accept both Euler stages with this same bound.
    /// MC concentration reconstruction can double each outgoing advective term;
    /// dt*(outgoing discharge + drain)/h <= .4 therefore leaves a positive margin.
    fn is_safe(
        &self,
        dt: f32,
    ) -> bool {
        if self.depth.len() != CELLS
            || self.drain.len() != CELLS
            || self.feed.len() != CELLS
            || self.discharge_x.len() != (NX + 1) * NY
            || self.discharge_y.len() != NX * (NY + 1)
            || self
                .feed_concentration
                .iter()
                .any(|c| !c.is_finite() || *c < 0.0)
            || self.depth.iter().any(|h| !h.is_finite() || *h <= 0.0)
            || self
                .drain
                .iter()
                .chain(self.feed)
                .any(|q| !q.is_finite() || *q < 0.0)
            || self
                .discharge_x
                .iter()
                .chain(self.discharge_y)
                .any(|q| !q.is_finite())
        {
            return false;
        }
        for y in 0..NY {
            if self.discharge_x[y * (NX + 1)] != self.discharge_x[y * (NX + 1) + NX] {
                return false;
            }
            for x in 0..NX {
                let i = y * NX + x;
                let outgoing = ((self.discharge_x[y * (NX + 1) + x + 1].max(0.0)
                    - self.discharge_x[y * (NX + 1) + x].min(0.0))
                    / DX
                    + (self.discharge_y[(y + 1) * NX + x].max(0.0)
                        - self.discharge_y[y * NX + x].min(0.0))
                        / DY
                    + self.drain[i])
                    / self.depth[i];
                if !outgoing.is_finite() || dt * outgoing > 0.4 {
                    return false;
                }
            }
        }
        (0..NX).all(|x| self.discharge_y[x] == self.discharge_y[NY * NX + x])
    }
}

#[derive(Debug)]
pub struct Dye {
    cells: Vec<[f32; 3]>,
    hydraulic_base: Vec<[f32; 3]>,
    hydraulic_concentration: Vec<[f32; 3]>,
    hydraulic_candidate: Option<HydraulicLedger>,
    stage: Vec<[f32; 3]>,
    derivative: Vec<[f32; 3]>,
    slope_x: Vec<[f32; 3]>,
    slope_y: Vec<[f32; 3]>,
    velocity: Vec<[f32; 2]>,
    depth: Vec<f32>,
    face_x: Vec<f32>,
    face_y: Vec<f32>,
    diffusivity: Vec<f32>,
    diffusion_x: Vec<f32>,
    diffusion_y: Vec<f32>,
    boundary_loss: [f64; 3],
    bleached_mass: [f64; 3],
    drained_mass: [f64; 3],
    fed_mass: [f64; 3],
    bleach: f32,
    pending: Duration,
    boundary: Boundary,
    active: bool,
}

impl Default for Dye {
    fn default() -> Self { Self::new() }
}

impl Dye {
    pub fn new() -> Self {
        Self {
            cells: vec![[0.0; 3]; CELLS],
            hydraulic_base: vec![[0.0; 3]; CELLS],
            hydraulic_concentration: vec![[0.0; 3]; CELLS],
            hydraulic_candidate: None,
            stage: vec![[0.0; 3]; CELLS],
            derivative: vec![[0.0; 3]; CELLS],
            slope_x: vec![[0.0; 3]; CELLS],
            slope_y: vec![[0.0; 3]; CELLS],
            velocity: vec![[0.0; 2]; CELLS],
            depth: vec![1.0; CELLS],
            face_x: vec![0.0; (NX + 1) * NY],
            face_y: vec![0.0; NX * (NY + 1)],
            diffusivity: vec![MOLECULAR_DIFFUSIVITY; CELLS],
            diffusion_x: vec![0.0; (NX + 1) * NY],
            diffusion_y: vec![0.0; NX * (NY + 1)],
            boundary_loss: [0.0; 3],
            bleached_mass: [0.0; 3],
            drained_mass: [0.0; 3],
            fed_mass: [0.0; 3],
            bleach: DEFAULT_BLEACH,
            pending: Duration::ZERO,
            boundary: Boundary::Periodic,
            active: false,
        }
    }

    pub fn is_active(&self) -> bool { self.active }

    /// Move the fixed-size local window by whole cells without resampling:
    /// new[x, y] = old[x + shift[0], y + shift[1]]. Entering cells are clear.
    /// Pigment leaving the represented window is exported to the boundary-loss
    /// ledger exactly once, including corners. Reuse the Euler-stage allocation.
    /// This changes coordinates only: physical time and bleaching do not advance.
    pub(crate) fn shift_cells(
        &mut self,
        shift: [isize; 2],
    ) {
        if shift == [0, 0] {
            return;
        }
        self.hydraulic_candidate = None;
        self.stage.fill([0.0; 3]);
        let area = DX as f64 * DY as f64;
        let mut active = false;
        for y in 0..NY {
            let destination_y = (y as isize).checked_sub(shift[1]);
            for x in 0..NX {
                let source = self.cells[y * NX + x];
                let destination_x = (x as isize).checked_sub(shift[0]);
                match (destination_x, destination_y) {
                    (Some(dx), Some(dy))
                        if (0..NX as isize).contains(&dx) && (0..NY as isize).contains(&dy) =>
                    {
                        self.stage[dy as usize * NX + dx as usize] = source;
                        active |= source.iter().any(|&c| c != 0.0);
                    },
                    _ => {
                        for (lost, concentration) in self.boundary_loss.iter_mut().zip(source) {
                            *lost += concentration as f64 * area;
                        }
                    },
                }
            }
        }
        std::mem::swap(&mut self.cells, &mut self.stage);
        self.active = active;
    }

    /// Prepare pigment using exactly the two accepted hydraulic SSP-RK2 stages.
    /// Reconstruct C=M/h; use the supplied water discharge, suction and feed.
    /// No committed concentrations, clocks or mass ledgers change here. Failure
    /// clears any previous candidate, allowing a smaller hydraulic retry safely.
    /// Drain mode has no added eddy diffusion; its mixing is resolved advection
    /// plus finite-volume numerical diffusion. Production feed concentration is 0.
    pub(crate) fn prepare_hydraulic_step(
        &mut self,
        dt: f32,
        stage0: HydraulicStage<'_>,
        stage1: HydraulicStage<'_>,
    ) -> bool {
        self.hydraulic_candidate = None;
        if !dt.is_finite() || dt < 0.0 {
            return false;
        }
        if dt == 0.0 {
            return true;
        }
        if !stage0.is_safe(dt) || !stage1.is_safe(dt) {
            return false;
        }
        if !self.active
            && stage0.feed_concentration == [0.0; 3]
            && stage1.feed_concentration == [0.0; 3]
        {
            self.stage.fill([0.0; 3]);
            self.hydraulic_candidate = Some(HydraulicLedger::default());
            return true;
        }
        let survival = (-std::f64::consts::LN_2 / 8.0 * self.bleach as f64 * 0.5 * dt as f64).exp();
        let area = DX as f64 * DY as f64;
        let mut ledger = HydraulicLedger::default();
        for i in 0..CELLS {
            for k in 0..3 {
                let before = self.cells[i][k] as f64;
                if !before.is_finite() || before < 0.0 {
                    return false;
                }
                self.hydraulic_base[i][k] = bleach_concentration(self.cells[i][k], survival);
                ledger.bleached[k] += (before - self.hydraulic_base[i][k] as f64) * area;
            }
        }
        let first = self.hydraulic_derivative(&stage0, false);
        for i in 0..CELLS {
            for k in 0..3 {
                let value = self.hydraulic_base[i][k] + dt * self.derivative[i][k];
                if !value.is_finite() || value < 0.0 {
                    return false;
                }
                self.stage[i][k] = value;
            }
        }
        let second = self.hydraulic_derivative(&stage1, true);
        for i in 0..CELLS {
            for k in 0..3 {
                let euler = self.stage[i][k] + dt * self.derivative[i][k];
                if !euler.is_finite() || euler < 0.0 {
                    return false;
                }
                let value = 0.5 * self.hydraulic_base[i][k] + 0.5 * euler;
                self.stage[i][k] = bleach_concentration(value, survival);
                ledger.bleached[k] += (value as f64 - self.stage[i][k] as f64) * area;
                ledger.active |= self.stage[i][k] != 0.0;
            }
        }
        for k in 0..3 {
            ledger.drained[k] = 0.5 * dt as f64 * (first.drained[k] + second.drained[k]);
            ledger.fed[k] = 0.5 * dt as f64 * (first.fed[k] + second.fed[k]);
        }
        self.hydraulic_candidate = Some(ledger);
        true
    }

    /// Commit only after the matching water step is accepted. A duplicate commit
    /// or a commit after rejected preparation is a no-op.
    pub(crate) fn commit_hydraulic_step(&mut self) {
        let Some(ledger) = self.hydraulic_candidate.take() else {
            return;
        };
        std::mem::swap(&mut self.cells, &mut self.stage);
        for k in 0..3 {
            self.drained_mass[k] += ledger.drained[k];
            self.fed_mass[k] += ledger.fed[k];
            self.bleached_mass[k] += ledger.bleached[k];
        }
        self.active = ledger.active;
        self.boundary = Boundary::Periodic;
    }

    fn hydraulic_derivative(
        &mut self,
        water: &HydraulicStage<'_>,
        use_stage: bool,
    ) -> HydraulicLedger {
        let cells = if use_stage {
            &self.stage
        } else {
            &self.hydraulic_base
        };
        // Every neighbor, source and face uses the same rounded quotient in
        // this stage. Compute it once; retain direct division rather than a
        // reciprocal product so the original f32 transport stays bit-exact.
        for (i, cell) in cells.iter().enumerate() {
            self.hydraulic_concentration[i] = std::array::from_fn(|k| cell[k] / water.depth[i]);
        }
        let concentration = |i: usize, k: usize| self.hydraulic_concentration[i][k];
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let west = y * NX + (x + NX - 1) % NX;
                let east = y * NX + (x + 1) % NX;
                let south = ((y + NY - 1) % NY) * NX + x;
                let north = ((y + 1) % NY) * NX + x;
                for k in 0..3 {
                    let c = concentration(i, k);
                    self.slope_x[i][k] =
                        monotonized_central(c - concentration(west, k), concentration(east, k) - c);
                    self.slope_y[i][k] = monotonized_central(
                        c - concentration(south, k),
                        concentration(north, k) - c,
                    );
                }
            }
        }
        self.derivative.fill([0.0; 3]);
        let mut ledger = HydraulicLedger::default();
        let area = DX as f64 * DY as f64;
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                for k in 0..3 {
                    let drained = water.drain[i] * concentration(i, k);
                    let fed = water.feed[i] * water.feed_concentration[k];
                    self.derivative[i][k] += fed - drained;
                    ledger.drained[k] += drained as f64 * area;
                    ledger.fed[k] += fed as f64 * area;
                }
                for axis in 0..2 {
                    let (j, q, spacing, slopes) = if axis == 0 {
                        (
                            y * NX + (x + 1) % NX,
                            water.discharge_x[y * (NX + 1) + x + 1],
                            DX,
                            &self.slope_x,
                        )
                    } else {
                        (
                            ((y + 1) % NY) * NX + x,
                            water.discharge_y[(y + 1) * NX + x],
                            DY,
                            &self.slope_y,
                        )
                    };
                    for (k, (&left_slope, &right_slope)) in
                        slopes[i].iter().zip(&slopes[j]).enumerate()
                    {
                        let upstream = if q >= 0.0 {
                            concentration(i, k) + 0.5 * left_slope
                        } else {
                            concentration(j, k) - 0.5 * right_slope
                        };
                        let flux = q * upstream / spacing;
                        self.derivative[i][k] -= flux;
                        self.derivative[j][k] += flux;
                    }
                }
            }
        }
        ledger
    }

    // Frozen pre-cache implementation. Keep direct quotient evaluations here so
    // a changed arithmetic order or stale stage cache is detectable exactly.
    #[cfg(test)]
    fn prepare_hydraulic_step_reference(
        &mut self,
        dt: f32,
        stage0: HydraulicStage<'_>,
        stage1: HydraulicStage<'_>,
    ) -> bool {
        self.hydraulic_candidate = None;
        if !dt.is_finite() || dt < 0.0 {
            return false;
        }
        if dt == 0.0 {
            return true;
        }
        if !stage0.is_safe(dt) || !stage1.is_safe(dt) {
            return false;
        }
        if !self.active
            && stage0.feed_concentration == [0.0; 3]
            && stage1.feed_concentration == [0.0; 3]
        {
            self.stage.fill([0.0; 3]);
            self.hydraulic_candidate = Some(HydraulicLedger::default());
            return true;
        }
        let survival = (-std::f64::consts::LN_2 / 8.0 * self.bleach as f64 * 0.5 * dt as f64).exp();
        let area = DX as f64 * DY as f64;
        let mut ledger = HydraulicLedger::default();
        for i in 0..CELLS {
            for k in 0..3 {
                let before = self.cells[i][k] as f64;
                if !before.is_finite() || before < 0.0 {
                    return false;
                }
                self.hydraulic_base[i][k] = bleach_concentration(self.cells[i][k], survival);
                ledger.bleached[k] += (before - self.hydraulic_base[i][k] as f64) * area;
            }
        }
        let first = self.hydraulic_derivative_reference(&stage0, false);
        for i in 0..CELLS {
            for k in 0..3 {
                let value = self.hydraulic_base[i][k] + dt * self.derivative[i][k];
                if !value.is_finite() || value < 0.0 {
                    return false;
                }
                self.stage[i][k] = value;
            }
        }
        let second = self.hydraulic_derivative_reference(&stage1, true);
        for i in 0..CELLS {
            for k in 0..3 {
                let euler = self.stage[i][k] + dt * self.derivative[i][k];
                if !euler.is_finite() || euler < 0.0 {
                    return false;
                }
                let value = 0.5 * self.hydraulic_base[i][k] + 0.5 * euler;
                self.stage[i][k] = bleach_concentration(value, survival);
                ledger.bleached[k] += (value as f64 - self.stage[i][k] as f64) * area;
                ledger.active |= self.stage[i][k] != 0.0;
            }
        }
        for k in 0..3 {
            ledger.drained[k] = 0.5 * dt as f64 * (first.drained[k] + second.drained[k]);
            ledger.fed[k] = 0.5 * dt as f64 * (first.fed[k] + second.fed[k]);
        }
        self.hydraulic_candidate = Some(ledger);
        true
    }

    #[cfg(test)]
    fn hydraulic_derivative_reference(
        &mut self,
        water: &HydraulicStage<'_>,
        use_stage: bool,
    ) -> HydraulicLedger {
        let cells = if use_stage {
            &self.stage
        } else {
            &self.hydraulic_base
        };
        let concentration = |i: usize, k: usize| cells[i][k] / water.depth[i];
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let west = y * NX + (x + NX - 1) % NX;
                let east = y * NX + (x + 1) % NX;
                let south = ((y + NY - 1) % NY) * NX + x;
                let north = ((y + 1) % NY) * NX + x;
                for k in 0..3 {
                    let c = concentration(i, k);
                    self.slope_x[i][k] =
                        monotonized_central(c - concentration(west, k), concentration(east, k) - c);
                    self.slope_y[i][k] = monotonized_central(
                        c - concentration(south, k),
                        concentration(north, k) - c,
                    );
                }
            }
        }
        self.derivative.fill([0.0; 3]);
        let mut ledger = HydraulicLedger::default();
        let area = DX as f64 * DY as f64;
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                for k in 0..3 {
                    let drained = water.drain[i] * concentration(i, k);
                    let fed = water.feed[i] * water.feed_concentration[k];
                    self.derivative[i][k] += fed - drained;
                    ledger.drained[k] += drained as f64 * area;
                    ledger.fed[k] += fed as f64 * area;
                }
                for axis in 0..2 {
                    let (j, q, spacing, slopes) = if axis == 0 {
                        (
                            y * NX + (x + 1) % NX,
                            water.discharge_x[y * (NX + 1) + x + 1],
                            DX,
                            &self.slope_x,
                        )
                    } else {
                        (
                            ((y + 1) % NY) * NX + x,
                            water.discharge_y[(y + 1) * NX + x],
                            DY,
                            &self.slope_y,
                        )
                    };
                    for (k, (&left_slope, &right_slope)) in
                        slopes[i].iter().zip(&slopes[j]).enumerate()
                    {
                        let upstream = if q >= 0.0 {
                            concentration(i, k) + 0.5 * left_slope
                        } else {
                            concentration(j, k) - 0.5 * right_slope
                        };
                        let flux = q * upstream / spacing;
                        self.derivative[i][k] -= flux;
                        self.derivative[j][k] += flux;
                    }
                }
            }
        }
        ledger
    }

    /// Set uniform chromophore bleaching strength in 0..=4. At 1, the half-life
    /// is 8 physical seconds; at 0, reaction is disabled exactly. This controls
    /// a separate first-order reaction, not diffusion or unresolved dilution.
    /// Nonfinite input restores the default strength.
    pub fn set_bleach(
        &mut self,
        bleach: f32,
    ) {
        self.hydraulic_candidate = None;
        self.bleach = if bleach.is_finite() {
            bleach.clamp(0.0, MAX_BLEACH)
        } else {
            DEFAULT_BLEACH
        };
    }

    /// Set topology before an initial deposit, without advancing physical time.
    pub fn set_boundary(
        &mut self,
        boundary: Boundary,
    ) {
        self.hydraulic_candidate = None;
        self.boundary = boundary;
    }

    /// Deposit a Gaussian with RMS radius in metres at a normalized position.
    /// `amount` is the total areal pigment integral (concentration * m²), shared
    /// between three nonnegative species weights. The kernel is normalized even
    /// at patch edges, so changing radius cannot change deposited pigment mass.
    /// Subcell radii use half a cell to avoid aliasing; unsupported inputs are
    /// ignored and a single deposit is limited to 1 concentration * m².
    pub fn deposit(
        &mut self,
        point: [f32; 2],
        radius_m: f32,
        amount: f32,
        pigment: [f32; 3],
    ) {
        self.hydraulic_candidate = None;
        if point
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || !radius_m.is_finite()
            || !amount.is_finite()
            || amount <= 0.0
        {
            return;
        }
        let weights = pigment.map(|c| {
            if c.is_finite() {
                c.clamp(0.0, 1e6)
            } else {
                0.0
            }
        });
        let sum: f32 = weights.iter().sum();
        if sum == 0.0 {
            return;
        }
        let species = weights.map(|c| c / sum);
        let sigma = radius_m.clamp(0.5 * DX.min(DY), 0.5);
        let radius = 4.0 * sigma;
        let point = point.map(|p| p * LENGTH);
        let mut integral = 0.0;
        for y in 0..NY {
            let dy = ((y as f32 + 0.5) * DY - point[1]).abs();
            let dy = if self.boundary == Boundary::OpenPlane {
                dy
            } else {
                dy.min(LENGTH - dy)
            };
            for x in 0..NX {
                let dx = ((x as f32 + 0.5) * DX - point[0]).abs();
                let dx = if self.boundary == Boundary::Periodic {
                    dx.min(LENGTH - dx)
                } else {
                    dx
                };
                let kernel = if dx > radius || dy > radius {
                    0.0
                } else {
                    (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
                };
                self.derivative[y * NX + x][0] = kernel;
                integral += kernel;
            }
        }
        let scale = amount.min(1.0) / (integral * DX * DY);
        for i in 0..CELLS {
            let density = self.derivative[i][0] * scale;
            for (c, weight) in self.cells[i].iter_mut().zip(species) {
                *c += density * weight;
            }
        }
        self.active = true;
    }

    /// Advance physical time in fixed 8 ms ticks. Velocity is sampled once per
    /// tick; supplying a changing field between calls naturally changes forcing.
    /// Constant-depth convenience adapter for transport verification.
    /// Finite base velocities are limited to +/-10 m/s as an input guard.
    #[cfg(test)]
    pub fn advance(
        &mut self,
        duration: Duration,
        velocity: impl Fn([f32; 2]) -> [f32; 2],
        boundary: Boundary,
    ) {
        self.advance_with_streamfunction(duration, |p| (velocity(p), 1.0), &[], boundary);
    }

    /// Add a discrete divergence-free velocity from vertex streamfunction ψ in
    /// m²/s, laid out as y*(NX+1)+x. Its face curl is u=-∂yψ, v=∂xψ.
    /// The callback returns base velocity in m/s and wet column depth in metres.
    /// Diffusion follows volume concentration M/h, while M itself is conserved.
    /// Periodic seams use canonical shared vertices. Invalid buffers contribute
    /// no curl. A pathological curl is scaled uniformly to a 10 m/s envelope,
    /// preserving its discrete divergence instead of clipping its components.
    pub fn advance_with_streamfunction(
        &mut self,
        duration: Duration,
        water: impl Fn([f32; 2]) -> ([f32; 2], f32),
        streamfunction: &[f32],
        boundary: Boundary,
    ) {
        self.set_boundary(boundary);
        self.pending += duration;
        if !self.active {
            self.pending = Duration::from_nanos((self.pending.as_nanos() % TICK.as_nanos()) as u64);
            return;
        }
        let streamfunction = (streamfunction.len() == (NX + 1) * (NY + 1)
            && streamfunction.iter().all(|p| p.is_finite()))
        .then_some(streamfunction);
        while self.pending >= TICK && self.active {
            self.pending -= TICK;
            self.tick(&water, streamfunction);
        }
        if !self.active {
            self.pending = Duration::from_nanos((self.pending.as_nanos() % TICK.as_nanos()) as u64);
        }
    }

    /// Bilinear cell-centered sample. OpenPlane clamps interpolation support at
    /// the four edges and returns clear water outside the represented window.
    pub fn sample(
        &self,
        point: [f32; 2],
    ) -> [f32; 3] {
        if !self.active
            || point.iter().any(|p| !p.is_finite())
            || (self.boundary == Boundary::OpenPlane
                && point.iter().any(|p| !(0.0..=1.0).contains(p)))
        {
            return [0.0; 3];
        }
        let x = point[0].clamp(0.0, 1.0) * NX as f32 - 0.5;
        let y = point[1].clamp(0.0, 1.0) * NY as f32 - 0.5;
        let ix = x.floor() as isize;
        let iy = y.floor() as isize;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let mut value = [0.0; 3];
        for (col, wx) in [(ix, 1.0 - fx), (ix + 1, fx)] {
            let col = match self.boundary {
                Boundary::Periodic => col.rem_euclid(NX as isize) as usize,
                Boundary::OpenChannel | Boundary::OpenPlane => {
                    col.clamp(0, NX as isize - 1) as usize
                },
            };
            for (row, wy) in [(iy, 1.0 - fy), (iy + 1, fy)] {
                let row = if self.boundary == Boundary::OpenPlane {
                    row.clamp(0, NY as isize - 1) as usize
                } else {
                    row.rem_euclid(NY as isize) as usize
                };
                for (v, c) in value.iter_mut().zip(self.cells[row * NX + col]) {
                    *v += wx * wy * c;
                }
            }
        }
        value
    }

    fn tick(
        &mut self,
        water: &impl Fn([f32; 2]) -> ([f32; 2], f32),
        streamfunction: Option<&[f32]>,
    ) {
        for y in 0..NY {
            for x in 0..NX {
                let (velocity, depth) =
                    water([(x as f32 + 0.5) / NX as f32, (y as f32 + 0.5) / NY as f32]);
                self.depth[y * NX + x] = if depth.is_finite() && depth > 0.0 {
                    depth.max(1e-6)
                } else {
                    1.0
                };
                let v = velocity.map(|v| {
                    if v.is_finite() {
                        v.clamp(-10.0, 10.0)
                    } else {
                        0.0
                    }
                });
                self.velocity[y * NX + x] = v;
            }
        }
        let max_outgoing = self.resolve_faces(streamfunction);
        self.resolve_diffusivity();
        let max_diffusion_rate = self.resolve_diffusion_faces();
        // Monotonized-central reconstruction bounds outgoing face density by 2*cell
        // density. Include that factor and all four diffusion outflows in CFL.
        let rate = 2.0 * max_outgoing + max_diffusion_rate;
        let max_dt = 0.8 / rate.max(1e-6);
        let survival =
            (-std::f64::consts::LN_2 / 8.0 * self.bleach as f64 * 0.5 * TICK.as_secs_f64()).exp();
        self.bleach_half_step(survival);
        let mut remaining = TICK.as_secs_f32();
        while remaining > 0.0 {
            let dt = remaining.min(max_dt);
            self.transport(dt);
            remaining -= dt;
        }
        self.bleach_half_step(survival);
        self.active = self
            .cells
            .iter()
            .flatten()
            .any(|&concentration| concentration != 0.0);
    }

    fn bleach_half_step(
        &mut self,
        survival: f64,
    ) {
        if survival == 1.0 {
            return;
        }
        let area = DX as f64 * DY as f64;
        for cell in &mut self.cells {
            for (concentration, removed) in cell.iter_mut().zip(&mut self.bleached_mass) {
                let before = *concentration as f64;
                *concentration = bleach_concentration(*concentration, survival);
                // Account for what the stored field actually lost, including
                // f32 rounding, rather than an idealized reaction increment.
                *removed += (before - *concentration as f64) * area;
            }
        }
    }

    fn resolve_faces(
        &mut self,
        streamfunction: Option<&[f32]>,
    ) -> f32 {
        self.face_x.fill(0.0);
        self.face_y.fill(0.0);
        if let Some(psi) = streamfunction {
            let vertex = |x: usize, y: usize| {
                let x = if self.boundary == Boundary::Periodic {
                    x % NX
                } else {
                    x
                };
                let y = if self.boundary == Boundary::OpenPlane {
                    y
                } else {
                    y % NY
                };
                psi[y * (NX + 1) + x]
            };
            for y in 0..NY {
                for x in 0..=NX {
                    self.face_x[y * (NX + 1) + x] = -(vertex(x, y + 1) - vertex(x, y)) / DY;
                }
            }
            for y in 0..=NY {
                for x in 0..NX {
                    self.face_y[y * NX + x] = (vertex(x + 1, y) - vertex(x, y)) / DX;
                }
            }
            let finite = self
                .face_x
                .iter()
                .chain(&self.face_y)
                .all(|v| v.is_finite());
            if finite {
                let maximum = self
                    .face_x
                    .iter()
                    .chain(&self.face_y)
                    .fold(0.0f32, |a, b| a.max(b.abs()));
                let scale = if maximum > 10.0 { 10.0 / maximum } else { 1.0 };
                for face in self.face_x.iter_mut().chain(&mut self.face_y) {
                    *face *= scale;
                }
            } else {
                self.face_x.fill(0.0);
                self.face_y.fill(0.0);
            }
        }
        for y in 0..NY {
            for x in 0..=NX {
                let (left, right) = match self.boundary {
                    Boundary::Periodic => ((x + NX - 1) % NX, x % NX),
                    Boundary::OpenChannel | Boundary::OpenPlane => {
                        (x.saturating_sub(1), x.min(NX - 1))
                    },
                };
                self.face_x[y * (NX + 1) + x] +=
                    0.5 * (self.velocity[y * NX + left][0] + self.velocity[y * NX + right][0]);
            }
        }
        for y in 0..=NY {
            let (lower, upper) = if self.boundary == Boundary::OpenPlane {
                (y.saturating_sub(1), y.min(NY - 1))
            } else {
                ((y + NY - 1) % NY, y % NY)
            };
            for x in 0..NX {
                self.face_y[y * NX + x] +=
                    0.5 * (self.velocity[lower * NX + x][1] + self.velocity[upper * NX + x][1]);
            }
        }
        let mut maximum = 0.0f32;
        for y in 0..NY {
            for x in 0..NX {
                let left = self.face_x[y * (NX + 1) + x];
                let right = self.face_x[y * (NX + 1) + x + 1];
                let lower = self.face_y[y * NX + x];
                let upper = self.face_y[(y + 1) * NX + x];
                self.velocity[y * NX + x] = [0.5 * (left + right), 0.5 * (lower + upper)];
                let outgoing =
                    (right.max(0.0) - left.min(0.0)) / DX + (upper.max(0.0) - lower.min(0.0)) / DY;
                maximum = maximum.max(outgoing);
            }
        }
        maximum
    }

    fn transport(
        &mut self,
        dt: f32,
    ) {
        let first_loss = self.spatial_derivative(false);
        for i in 0..CELLS {
            for k in 0..3 {
                self.stage[i][k] = self.cells[i][k] + dt * self.derivative[i][k];
            }
        }
        let second_loss = self.spatial_derivative(true);
        for i in 0..CELLS {
            for k in 0..3 {
                self.cells[i][k] =
                    0.5 * self.cells[i][k] + 0.5 * (self.stage[i][k] + dt * self.derivative[i][k]);
            }
        }
        for k in 0..3 {
            self.boundary_loss[k] += 0.5 * dt as f64 * (first_loss[k] + second_loss[k]);
        }
    }

    fn resolve_diffusivity(&mut self) -> f32 {
        let mixing_length_squared = 0.3f32.powi(2) * DX * DY;
        let mut maximum = MOLECULAR_DIFFUSIVITY;
        for y in 0..NY {
            let (south, north) = self.y_neighbors(y);
            let y_spacing = if self.boundary == Boundary::OpenPlane {
                (north - south) as f32 * DY
            } else {
                2.0 * DY
            };
            for x in 0..NX {
                let (west, east) = self.x_neighbors(x);
                let v_w = self.velocity[y * NX + west];
                let v_e = self.velocity[y * NX + east];
                let v_s = self.velocity[south * NX + x];
                let v_n = self.velocity[north * NX + x];
                let x_spacing = match self.boundary {
                    Boundary::Periodic => 2.0 * DX,
                    Boundary::OpenChannel | Boundary::OpenPlane => (east - west) as f32 * DX,
                };
                let du_dx = (v_e[0] - v_w[0]) / x_spacing;
                let dv_dy = (v_n[1] - v_s[1]) / y_spacing;
                let shear = (v_n[0] - v_s[0]) / y_spacing + (v_e[1] - v_w[1]) / x_spacing;
                let strain = (2.0 * (du_dx * du_dx + dv_dy * dv_dy) + shear * shear).sqrt();
                let diffusivity = MOLECULAR_DIFFUSIVITY
                    + (mixing_length_squared * strain).min(MAX_EDDY_DIFFUSIVITY);
                self.diffusivity[y * NX + x] = diffusivity;
                maximum = maximum.max(diffusivity);
            }
        }
        maximum
    }

    fn resolve_diffusion_faces(&mut self) -> f32 {
        let conductance = |i: usize, j: usize| {
            let small = self.depth[i].min(self.depth[j]);
            let large = self.depth[i].max(self.depth[j]);
            let harmonic = small / (0.5 + 0.5 * small / large);
            0.5 * (self.diffusivity[i] + self.diffusivity[j]) * harmonic
        };
        for y in 0..NY {
            for x in 0..=NX {
                self.diffusion_x[y * (NX + 1) + x] =
                    if self.boundary != Boundary::Periodic && (x == 0 || x == NX) {
                        0.0
                    } else {
                        conductance(y * NX + (x + NX - 1) % NX, y * NX + x % NX)
                    };
            }
        }
        for y in 0..=NY {
            for x in 0..NX {
                self.diffusion_y[y * NX + x] =
                    if self.boundary == Boundary::OpenPlane && (y == 0 || y == NY) {
                        0.0
                    } else {
                        conductance(((y + NY - 1) % NY) * NX + x, (y % NY) * NX + x)
                    };
            }
        }
        let mut maximum = 0.0f32;
        for y in 0..NY {
            for x in 0..NX {
                let rate = ((self.diffusion_x[y * (NX + 1) + x]
                    + self.diffusion_x[y * (NX + 1) + x + 1])
                    / (DX * DX)
                    + (self.diffusion_y[y * NX + x] + self.diffusion_y[(y + 1) * NX + x])
                        / (DY * DY))
                    / self.depth[y * NX + x];
                maximum = maximum.max(rate);
            }
        }
        maximum
    }

    fn x_neighbors(
        &self,
        x: usize,
    ) -> (usize, usize) {
        match self.boundary {
            Boundary::Periodic => ((x + NX - 1) % NX, (x + 1) % NX),
            Boundary::OpenChannel | Boundary::OpenPlane => {
                (x.saturating_sub(1), (x + 1).min(NX - 1))
            },
        }
    }

    fn y_neighbors(
        &self,
        y: usize,
    ) -> (usize, usize) {
        if self.boundary == Boundary::OpenPlane {
            (y.saturating_sub(1), (y + 1).min(NY - 1))
        } else {
            ((y + NY - 1) % NY, (y + 1) % NY)
        }
    }

    fn spatial_derivative(
        &mut self,
        use_stage: bool,
    ) -> [f64; 3] {
        let cells = if use_stage { &self.stage } else { &self.cells };
        for y in 0..NY {
            let (south, north) = self.y_neighbors(y);
            for x in 0..NX {
                let i = y * NX + x;
                let (west, east) = self.x_neighbors(x);
                for (k, &center) in cells[i].iter().enumerate() {
                    self.slope_x[i][k] = monotonized_central(
                        center - cells[y * NX + west][k],
                        cells[y * NX + east][k] - center,
                    );
                    self.slope_y[i][k] = monotonized_central(
                        center - cells[south * NX + x][k],
                        cells[north * NX + x][k] - center,
                    );
                }
            }
        }
        self.derivative.fill([0.0; 3]);
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                for axis in 0..2 {
                    if (axis == 0 && x == NX - 1 && self.boundary != Boundary::Periodic)
                        || (axis == 1 && y == NY - 1 && self.boundary == Boundary::OpenPlane)
                    {
                        continue;
                    }
                    let (j, spacing, slopes) = if axis == 0 {
                        (y * NX + (x + 1) % NX, DX, &self.slope_x)
                    } else {
                        (((y + 1) % NY) * NX + x, DY, &self.slope_y)
                    };
                    let v = if axis == 0 {
                        self.face_x[y * (NX + 1) + x + 1]
                    } else {
                        self.face_y[(y + 1) * NX + x]
                    };
                    let conductance = if axis == 0 {
                        self.diffusion_x[y * (NX + 1) + x + 1]
                    } else {
                        self.diffusion_y[(y + 1) * NX + x]
                    };
                    for k in 0..3 {
                        let upstream = if v >= 0.0 {
                            cells[i][k] + 0.5 * slopes[i][k]
                        } else {
                            cells[j][k] - 0.5 * slopes[j][k]
                        };
                        let flux = (v * upstream
                            - conductance
                                * (cells[j][k] / self.depth[j] - cells[i][k] / self.depth[i])
                                / spacing)
                            / spacing;
                        self.derivative[i][k] -= flux;
                        self.derivative[j][k] += flux;
                    }
                }
            }
        }
        let mut boundary_loss = [0.0; 3];
        if self.boundary != Boundary::Periodic {
            for y in 0..NY {
                let inlet = y * NX;
                let outlet = inlet + NX - 1;
                for (k, &c) in cells[inlet].iter().enumerate() {
                    let left = self.face_x[y * (NX + 1)].min(0.0) * c;
                    let right = self.face_x[y * (NX + 1) + NX].max(0.0) * cells[outlet][k];
                    self.derivative[inlet][k] += left / DX;
                    self.derivative[outlet][k] -= right / DX;
                    boundary_loss[k] += (right - left) as f64 * DY as f64;
                }
            }
        }
        if self.boundary == Boundary::OpenPlane {
            for x in 0..NX {
                let lower = x;
                let upper = (NY - 1) * NX + x;
                for (k, &c) in cells[lower].iter().enumerate() {
                    let bottom = self.face_y[x].min(0.0) * c;
                    let top = self.face_y[NY * NX + x].max(0.0) * cells[upper][k];
                    self.derivative[lower][k] += bottom / DY;
                    self.derivative[upper][k] -= top / DY;
                    boundary_loss[k] += (top - bottom) as f64 * DX as f64;
                }
            }
        }
        boundary_loss
    }
}

/// Below the normal f32 range, repeated exponential products can round back to
/// the same residue forever. Positive reaction converts that numerical residue
/// completely; callers account for the actual removed mass in their ledgers.
/// No-reaction transport retains the original value, including subnormals.
fn bleach_concentration(
    concentration: f32,
    survival: f64,
) -> f32 {
    if survival == 1.0 {
        return concentration;
    }
    let remaining = (concentration as f64 * survival) as f32;
    if remaining.is_subnormal() {
        0.0
    } else {
        remaining
    }
}

fn monotonized_central(
    a: f32,
    b: f32,
) -> f32 {
    if a > 0.0 && b > 0.0 {
        (0.5 * (a + b)).min(2.0 * a).min(2.0 * b)
    } else if a < 0.0 && b < 0.0 {
        (0.5 * (a + b)).max(2.0 * a).max(2.0 * b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conservative_dye() -> Dye {
        let mut dye = Dye::new();
        dye.set_bleach(0.0);
        dye
    }

    struct HydraulicFixture {
        depth: Vec<f32>,
        discharge_x: Vec<f32>,
        discharge_y: Vec<f32>,
        drain: Vec<f32>,
        feed: Vec<f32>,
    }

    impl HydraulicFixture {
        fn new() -> Self {
            Self {
                depth: vec![0.1; CELLS],
                discharge_x: vec![0.0; (NX + 1) * NY],
                discharge_y: vec![0.0; NX * (NY + 1)],
                drain: vec![0.0; CELLS],
                feed: vec![0.0; CELLS],
            }
        }

        fn stage<'a>(
            &'a self,
            depth: &'a [f32],
            concentration: [f32; 3],
        ) -> HydraulicStage<'a> {
            HydraulicStage {
                depth,
                discharge_x: &self.discharge_x,
                discharge_y: &self.discharge_y,
                drain: &self.drain,
                feed: &self.feed,
                feed_concentration: concentration,
            }
        }

        fn euler_depth(
            &self,
            depth: &[f32],
            dt: f32,
        ) -> Vec<f32> {
            (0..CELLS)
                .map(|i| {
                    let (x, y) = (i % NX, i / NX);
                    depth[i]
                        + dt * (-(self.discharge_x[y * (NX + 1) + x + 1]
                            - self.discharge_x[y * (NX + 1) + x])
                            / DX
                            - (self.discharge_y[(y + 1) * NX + x] - self.discharge_y[y * NX + x])
                                / DY
                            + self.feed[i]
                            - self.drain[i])
                })
                .collect()
        }

        fn advance(
            &mut self,
            dye: &mut Dye,
            dt: f32,
            concentration: [f32; 3],
        ) -> bool {
            let stage = self.euler_depth(&self.depth, dt);
            let euler2 = self.euler_depth(&stage, dt);
            let final_depth: Vec<f32> = self
                .depth
                .iter()
                .zip(&euler2)
                .map(|(a, b)| 0.5 * (a + b))
                .collect();
            let accepted = dye.prepare_hydraulic_step(
                dt,
                self.stage(&self.depth, concentration),
                self.stage(&stage, concentration),
            );
            if accepted {
                dye.commit_hydraulic_step();
                self.depth = final_depth;
            }
            accepted
        }
    }

    #[test]
    fn hydraulic_uniform_concentration_follows_the_same_water_stages() {
        let mut dye = conservative_dye();
        let mut water = HydraulicFixture::new();
        let concentration = [0.2, 0.7, 1.0];
        for y in 0..NY {
            for x in 0..NX {
                let px = std::f32::consts::TAU * x as f32 / NX as f32;
                let py = std::f32::consts::TAU * y as f32 / NY as f32;
                let i = y * NX + x;
                water.depth[i] = 0.1 + 0.03 * px.sin() + 0.02 * py.cos();
                water.drain[i] = 0.002 * (1.0 + px.cos() * py.cos());
                water.feed[i] = 0.001;
                water.discharge_x[y * (NX + 1) + x] = 0.025 + 0.004 * px.sin() * py.cos();
                water.discharge_y[y * NX + x] = 0.01 * px.cos() * py.sin();
                dye.cells[i] = concentration.map(|c| c * water.depth[i]);
            }
            water.discharge_x[y * (NX + 1) + NX] = water.discharge_x[y * (NX + 1)];
        }
        for x in 0..NX {
            water.discharge_y[NY * NX + x] = water.discharge_y[x];
        }
        dye.active = true;
        let initial = mass(&dye);
        for _ in 0..20 {
            assert!(water.advance(&mut dye, 0.008, concentration));
        }
        let mut error = 0.0f32;
        for (i, cell) in dye.cells.iter().enumerate() {
            for k in 0..3 {
                error = error.max((cell[k] / water.depth[i] - concentration[k]).abs());
            }
        }
        eprintln!("hydraulic uniform concentration maximum error={error:.9}");
        assert!(error < 2e-6);
        for (k, &initial_mass) in initial.iter().enumerate() {
            let residual =
                (mass(&dye)[k] + dye.drained_mass[k] - dye.fed_mass[k]) / initial_mass - 1.0;
            assert!(residual.abs() < 2e-6);
        }
    }

    #[test]
    fn hydraulic_concentration_cache_matches_uncached_stages_and_ledgers_exactly() {
        for strength in [0.0, 1.0, 4.0] {
            let mut cached = Dye::new();
            let mut reference = Dye::new();
            let mut water = HydraulicFixture::new();
            let mut seed = 0x913b_a19du32;
            let mut random = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / 16_777_216.0
            };
            let scales = [0.0, f32::MIN_POSITIVE * 0.5, 1e-25, 1e-12, 0.003, 0.1, 10.0];
            for i in 0..CELLS {
                water.depth[i] = 0.025 + 0.25 * random();
                water.drain[i] = 0.002 * random();
                water.feed[i] = 0.001 * random();
                for k in 0..3 {
                    cached.cells[i][k] = scales[(i + 2 * k) % scales.len()] * (0.1 + random());
                }
            }
            for face in water.discharge_x.iter_mut().chain(&mut water.discharge_y) {
                *face = 0.001 * (2.0 * random() - 1.0);
            }
            for y in 0..NY {
                water.discharge_x[y * (NX + 1) + NX] = water.discharge_x[y * (NX + 1)];
            }
            for x in 0..NX {
                water.discharge_y[NY * NX + x] = water.discharge_y[x];
            }
            reference.cells.clone_from(&cached.cells);
            cached.active = true;
            reference.active = true;
            cached.set_bleach(strength);
            reference.set_bleach(strength);
            for (step, dt) in [0.0041, 0.0063, 0.0009, 0.008].into_iter().enumerate() {
                let first_depth = water.euler_depth(&water.depth, dt);
                let second_depth = water.euler_depth(&first_depth, dt);
                let incoming = if step % 2 == 0 {
                    [0.0; 3]
                } else {
                    [0.1, 0.3, 0.2]
                };
                // Every stage must overwrite the entire reusable cache, even
                // when a preceding accepted step had a different wet depth.
                cached.hydraulic_concentration.fill([f32::NAN; 3]);
                let actual = cached.prepare_hydraulic_step(
                    dt,
                    water.stage(&water.depth, incoming),
                    water.stage(&first_depth, incoming),
                );
                let expected = reference.prepare_hydraulic_step_reference(
                    dt,
                    water.stage(&water.depth, incoming),
                    water.stage(&first_depth, incoming),
                );
                assert!(actual && expected);
                cached.commit_hydraulic_step();
                reference.commit_hydraulic_step();
                assert_eq!(
                    cached.cells, reference.cells,
                    "bleach={strength}, step={step}"
                );
                assert_eq!(cached.drained_mass, reference.drained_mass);
                assert_eq!(cached.fed_mass, reference.fed_mass);
                assert_eq!(cached.bleached_mass, reference.bleached_mass);
                assert_eq!(cached.active, reference.active);
                for (depth, end) in water.depth.iter_mut().zip(second_depth) {
                    *depth = 0.5 * (*depth + end);
                }
            }
        }
        eprintln!(
            "cached/uncached hydraulic fields and all ledgers: exactly equal across 12 \
             heterogeneous accepted steps"
        );
    }

    #[test]
    fn hydraulic_clear_feed_dilutes_without_adding_or_removing_species() {
        let mut dye = conservative_dye();
        let mut water = HydraulicFixture::new();
        water.feed.fill(0.01);
        dye.cells.fill([0.04, 0.02, 0.01]);
        dye.active = true;
        let original = dye.cells.clone();
        for _ in 0..125 {
            assert!(water.advance(&mut dye, 0.008, [0.0; 3]));
        }
        assert_eq!(dye.cells, original);
        assert_eq!(dye.fed_mass, [0.0; 3]);
        assert!((dye.cells[0][0] / water.depth[0] - 0.4 / 1.1).abs() < 2e-5);
    }

    #[test]
    fn bleaching_converts_subnormal_residue_but_zero_strength_retains_it() {
        for hydraulic in [false, true] {
            for strength in [1.0, 0.0] {
                let mut dye = Dye::new();
                dye.set_bleach(strength);
                dye.cells
                    .fill([0.5, 0.25, 0.125].map(|c| c * f32::MIN_POSITIVE));
                dye.active = true;
                let original = dye.cells.clone();
                let initial = mass(&dye);
                if hydraulic {
                    let mut water = HydraulicFixture::new();
                    assert!(water.advance(&mut dye, 0.008, [0.0; 3]));
                } else {
                    dye.advance(TICK, |_| [0.0; 2], Boundary::Periodic);
                }
                if strength == 0.0 {
                    assert_eq!(dye.cells, original);
                    assert_eq!(dye.bleached_mass, [0.0; 3]);
                    assert!(dye.is_active());
                } else {
                    assert!(
                        !dye.is_active(),
                        "subnormal residue persists, hydraulic={hydraulic}"
                    );
                    assert_eq!(mass(&dye), [0.0; 3]);
                    for (removed, original) in dye.bleached_mass.into_iter().zip(initial) {
                        assert!((removed / original - 1.0).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn hydraulic_suction_and_bleaching_close_the_species_ledger() {
        let mut dye = Dye::new();
        let mut water = HydraulicFixture::new();
        water.feed.fill(0.002);
        water.drain.fill(0.002);
        dye.deposit([0.5, 0.5], 0.08, 0.02, [0.5, 0.3, 0.2]);
        let initial = mass(&dye);
        for _ in 0..125 {
            assert!(water.advance(&mut dye, 0.008, [0.0; 3]));
        }
        let expected_fraction = (-0.002f64 / 0.1 - std::f64::consts::LN_2 / 8.0).exp();
        for (k, &initial_mass) in initial.iter().enumerate() {
            let remaining = mass(&dye)[k];
            let residual =
                (remaining + dye.drained_mass[k] + dye.bleached_mass[k]) / initial_mass - 1.0;
            eprintln!(
                "hydraulic suction species{k}: active fraction={:.9}, ledger \
                 residual={residual:.12}",
                remaining / initial_mass
            );
            assert!((remaining / initial_mass - expected_fraction).abs() < 2e-6);
            assert!(residual.abs() < 2e-6);
            assert!(dye.drained_mass[k] > 0.01 * initial_mass);
            assert!(dye.bleached_mass[k] > 0.05 * initial_mass);
        }
    }

    #[test]
    fn hydraulic_discharge_transports_species_at_the_water_volume_velocity() {
        let mut dye = conservative_dye();
        let mut water = HydraulicFixture::new();
        water.discharge_x.fill(0.03);
        dye.deposit([0.25, 0.3], 0.04, 0.02, [0.5, 0.3, 0.2]);
        let before = centroid_and_variance(&dye).0;
        for _ in 0..125 {
            assert!(water.advance(&mut dye, 0.008, [0.0; 3]));
        }
        let after = centroid_and_variance(&dye).0;
        eprintln!(
            "hydraulic displacement x={:.9}, expected=.3 m",
            after[0] - before[0]
        );
        assert!((after[0] - before[0] - 0.3).abs() < 0.002);
        assert!((after[1] - before[1]).abs() < 1e-5);
    }

    #[test]
    fn hydraulic_preparation_and_retries_do_not_mutate_committed_state() {
        let mut dye = Dye::new();
        let mut water = HydraulicFixture::new();
        water.drain.fill(0.002);
        water.feed.fill(0.002);
        dye.deposit([0.5, 0.5], 0.04, 0.02, [0.5, 0.3, 0.2]);
        let original = dye.cells.clone();
        let prepare = |dye: &mut Dye, dt| {
            dye.prepare_hydraulic_step(
                dt,
                water.stage(&water.depth, [0.0; 3]),
                water.stage(&water.depth, [0.0; 3]),
            )
        };
        assert!(prepare(&mut dye, 0.008));
        assert_eq!(dye.cells, original);
        assert_eq!(dye.drained_mass, [0.0; 3]);
        assert_eq!(dye.bleached_mass, [0.0; 3]);
        assert!(!prepare(&mut dye, 30.0));
        dye.commit_hydraulic_step();
        assert_eq!(
            dye.cells, original,
            "a rejected retry must discard prior scratch state"
        );
        for dt in [f32::NAN, f32::INFINITY, -0.01] {
            assert!(!prepare(&mut dye, dt));
        }
        assert!(prepare(&mut dye, 0.008));
        dye.commit_hydraulic_step();
        assert_ne!(dye.cells, original);
        let committed = dye.cells.clone();
        let bleached = dye.bleached_mass;
        let drained = dye.drained_mass;
        dye.commit_hydraulic_step();
        assert_eq!(dye.cells, committed);
        assert_eq!(dye.bleached_mass, bleached);
        assert_eq!(dye.drained_mass, drained);
    }

    #[test]
    fn hydraulic_colored_feed_activates_an_empty_field_only_after_commit() {
        let mut dye = conservative_dye();
        let mut water = HydraulicFixture::new();
        water.feed.fill(0.01);
        let concentration = [0.2, 0.7, 1.0];
        let stage_depth = water.euler_depth(&water.depth, 0.008);
        assert!(dye.prepare_hydraulic_step(
            0.008,
            water.stage(&water.depth, concentration),
            water.stage(&stage_depth, concentration)
        ));
        assert!(!dye.is_active());
        assert_eq!(mass(&dye), [0.0; 3]);
        assert_eq!(dye.fed_mass, [0.0; 3]);
        dye.commit_hydraulic_step();
        assert!(dye.is_active());
        for (k, &feed_concentration) in concentration.iter().enumerate() {
            let expected = 4.0 * 0.01 * 0.008 * feed_concentration as f64;
            assert!((mass(&dye)[k] / expected - 1.0).abs() < 1e-6);
            assert!((dye.fed_mass[k] / expected - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn hydraulic_stages_reject_unsafe_data_and_preserve_positive_filaments() {
        let mut dye = conservative_dye();
        let mut water = HydraulicFixture::new();
        dye.deposit([0.25, 0.3], 0.011, 0.02, [0.5, 0.3, 0.2]);
        let original = dye.cells.clone();
        water.depth[0] = 0.0;
        assert!(!dye.prepare_hydraulic_step(
            0.008,
            water.stage(&water.depth, [0.0; 3]),
            water.stage(&water.depth, [0.0; 3])
        ));
        water.depth[0] = 0.1;
        water.discharge_x[0] = f32::NAN;
        assert!(!dye.prepare_hydraulic_step(
            0.008,
            water.stage(&water.depth, [0.0; 3]),
            water.stage(&water.depth, [0.0; 3])
        ));
        water.discharge_x[0] = 0.0;
        let invalid = HydraulicStage {
            depth: &[],
            ..water.stage(&water.depth, [0.0; 3])
        };
        assert!(!dye.prepare_hydraulic_step(0.008, invalid, water.stage(&water.depth, [0.0; 3])));
        dye.commit_hydraulic_step();
        assert_eq!(dye.cells, original);
        water.discharge_x.fill(0.07);
        water.discharge_y.fill(0.025);
        water.drain.fill(0.002);
        water.feed.fill(0.002);
        let initial = mass(&dye);
        for _ in 0..125 {
            assert!(water.advance(&mut dye, 0.008, [0.0; 3]));
        }
        assert!(
            dye.cells
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
        for (k, &initial_mass) in initial.iter().enumerate() {
            let residual = (mass(&dye)[k] + dye.drained_mass[k]) / initial_mass - 1.0;
            assert!(residual.abs() < 2e-6);
        }
    }

    fn mass(dye: &Dye) -> [f64; 3] {
        let mut mass = [0.0; 3];
        for cell in &dye.cells {
            for k in 0..3 {
                mass[k] += cell[k] as f64 * DX as f64 * DY as f64;
            }
        }
        mass
    }

    #[test]
    fn open_plane_cell_shift_preserves_overlap_bit_exactly_and_exports_each_corner_once() {
        for shift in [
            [3, -5],
            [-7, 9],
            [0, 0],
            [NX as isize, 1],
            [1, NY as isize],
            [isize::MIN, isize::MAX],
        ] {
            let mut dye = conservative_dye();
            dye.set_boundary(Boundary::OpenPlane);
            dye.set_bleach(1.7);
            dye.pending = Duration::from_millis(3);
            for (i, cell) in dye.cells.iter_mut().enumerate() {
                *cell = [
                    0.25 + i as f32 / CELLS as f32,
                    0.0,
                    f32::from_bits(i as u32 + 1),
                ];
            }
            dye.active = true;
            let original = dye.cells.clone();
            let initial_mass = mass(&dye);
            let cell_buffer = dye.cells.as_ptr();
            let stage_buffer = dye.stage.as_ptr();
            dye.shift_cells(shift);
            for y in 0..NY {
                for x in 0..NX {
                    let source_x = (x as isize).checked_add(shift[0]);
                    let source_y = (y as isize).checked_add(shift[1]);
                    let expected = match (source_x, source_y) {
                        (Some(sx), Some(sy))
                            if (0..NX as isize).contains(&sx) && (0..NY as isize).contains(&sy) =>
                        {
                            original[sy as usize * NX + sx as usize]
                        },
                        _ => [0.0; 3],
                    };
                    assert_eq!(
                        dye.cells[y * NX + x].map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "shift {shift:?}, cell ({x}, {y})"
                    );
                }
            }
            let remaining = mass(&dye);
            for k in [0, 2] {
                assert!(
                    (remaining[k] + dye.boundary_loss[k] - initial_mass[k]).abs()
                        <= initial_mass[k] * 1e-12,
                    "shift {shift:?}, channel {k}: {}, {}, {}",
                    remaining[k],
                    dye.boundary_loss[k],
                    initial_mass[k]
                );
            }
            assert_eq!(dye.active, dye.cells.iter().flatten().any(|&v| v != 0.0));
            assert_eq!(dye.pending, Duration::from_millis(3));
            assert_eq!(dye.bleach, 1.7);
            assert_eq!(dye.boundary, Boundary::OpenPlane);
            assert!(dye.cells.as_ptr() == cell_buffer || dye.cells.as_ptr() == stage_buffer);
            assert!(dye.stage.as_ptr() == cell_buffer || dye.stage.as_ptr() == stage_buffer);
        }
    }

    #[test]
    fn open_plane_shift_invalidates_uncommitted_hydraulic_state() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.04, 0.01, [1.0, 0.0, 0.0]);
        let water = HydraulicFixture::new();
        assert!(dye.prepare_hydraulic_step(
            0.008,
            water.stage(&water.depth, [0.0; 3]),
            water.stage(&water.depth, [0.0; 3])
        ));
        dye.shift_cells([4, 0]);
        assert!(dye.hydraulic_candidate.is_none());
        let shifted = dye.cells.clone();
        dye.commit_hydraulic_step();
        assert_eq!(dye.cells, shifted);
        assert!(dye.hydraulic_candidate.is_none());
    }

    #[test]
    fn open_plane_deposition_and_sampling_do_not_clone_across_any_edge() {
        let mut dye = conservative_dye();
        dye.set_boundary(Boundary::OpenPlane);
        dye.deposit([0.0, 0.0], 0.06, 0.01, [1.0, 0.0, 0.0]);
        assert!(dye.sample([0.0, 0.0])[0] > 0.0);
        assert!((mass(&dye)[0] - 0.01).abs() < 1e-7);
        for point in [[0.02, 0.98], [0.98, 0.02], [0.98, 0.98], [-0.01, 0.0], [
            0.0, -0.01,
        ]] {
            assert_eq!(dye.sample(point), [0.0; 3], "point {point:?}");
        }
        dye.cells[0] = [3.0, 2.0, 1.0];
        assert_eq!(dye.sample([0.0, 0.0]), dye.cells[0]);
    }

    #[test]
    fn open_plane_outflow_on_both_axes_closes_mass_ledger_without_recycling() {
        for axis in 0..2 {
            for direction in [-1.0, 1.0] {
                let mut dye = conservative_dye();
                dye.set_boundary(Boundary::OpenPlane);
                let mut point = [0.5; 2];
                point[axis] = if direction > 0.0 { 0.88 } else { 0.12 };
                dye.deposit(point, 0.045, 0.02, [0.3, 0.5, 0.2]);
                let initial = mass(&dye);
                let mut velocity = [0.0; 2];
                velocity[axis] = 0.7 * direction;
                dye.advance(Duration::from_secs(1), |_| velocity, Boundary::OpenPlane);
                let remaining = mass(&dye);
                for k in 0..3 {
                    assert!(
                        (remaining[k] + dye.boundary_loss[k] - initial[k]).abs()
                            < initial[k] * 2e-5,
                        "axis {axis}, direction {direction}, channel {k}"
                    );
                    assert!(dye.boundary_loss[k] > initial[k] * 0.95);
                }
                point[axis] = if direction > 0.0 { 0.02 } else { 0.98 };
                assert_eq!(dye.sample(point), [0.0; 3]);
                assert!(
                    dye.cells
                        .iter()
                        .flatten()
                        .all(|&v| v.is_finite() && v >= 0.0)
                );
            }
        }
    }

    #[test]
    fn open_plane_diffusion_never_crosses_an_opposite_boundary() {
        let mut dye = conservative_dye();
        dye.set_boundary(Boundary::OpenPlane);
        dye.cells[0] = [1.0; 3];
        dye.active = true;
        dye.diffusivity.fill(0.001);
        dye.resolve_diffusion_faces();
        let initial = mass(&dye);
        dye.transport(0.001);
        assert_eq!(dye.cells[NX - 1], [0.0; 3]);
        assert_eq!(dye.cells[(NY - 1) * NX], [0.0; 3]);
        assert_eq!(dye.boundary_loss, [0.0; 3]);
        assert!(dye.cells[1][0] > 0.0 && dye.cells[NX][0] > 0.0);
        for (after, before) in mass(&dye).into_iter().zip(initial) {
            assert!((after - before).abs() < before * 1e-6);
        }
    }

    #[test]
    fn open_plane_strain_uses_one_sided_spacing_on_all_boundaries() {
        let mut dye = conservative_dye();
        dye.set_boundary(Boundary::OpenPlane);
        for y in 0..NY {
            for x in 0..NX {
                let px = (x as f32 + 0.5) * DX;
                let py = (y as f32 + 0.5) * DY;
                dye.velocity[y * NX + x] = [0.4 * px - 0.3 * py, 0.7 * px + 0.2 * py];
            }
        }
        dye.resolve_diffusivity();
        let strain = (2.0 * (0.4f32.powi(2) + 0.2f32.powi(2)) + 0.4f32.powi(2)).sqrt();
        let expected = MOLECULAR_DIFFUSIVITY + 0.3f32.powi(2) * DX * DY * strain;
        for (i, &actual) in dye.diffusivity.iter().enumerate() {
            assert!(
                (actual - expected).abs() < 1e-8,
                "cell {i}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn open_plane_streamfunction_retains_boundary_vertices_and_interior_uniform_mixture() {
        let mut dye = conservative_dye();
        dye.set_boundary(Boundary::OpenPlane);
        let mut psi = vec![0.0; (NX + 1) * (NY + 1)];
        for y in 0..=NY {
            for x in 0..=NX {
                let px = x as f32 * DX;
                let py = y as f32 * DY;
                psi[y * (NX + 1) + x] = 0.03 * px * py + 0.08 * px - 0.1 * py;
            }
        }
        dye.resolve_faces(Some(&psi));
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let divergence = (dye.face_x[y * (NX + 1) + x + 1] - dye.face_x[y * (NX + 1) + x])
                    / DX
                    + (dye.face_y[(y + 1) * NX + x] - dye.face_y[y * NX + x]) / DY;
                assert!(divergence.abs() < 1e-5, "cell {i}: divergence {divergence}");
                assert!((dye.velocity[i][0] - (0.1 - 0.03 * (x as f32 + 0.5) * DX)).abs() < 1e-5);
                assert!((dye.velocity[i][1] - (0.08 + 0.03 * (y as f32 + 0.5) * DY)).abs() < 1e-5);
            }
        }
        dye.cells.fill([0.3, 0.7, 1.0]);
        dye.active = true;
        dye.advance_with_streamfunction(TICK, |_| ([0.0; 2], 1.0), &psi, Boundary::OpenPlane);
        for y in 3..NY - 3 {
            for x in 3..NX - 3 {
                for (actual, expected) in dye.cells[y * NX + x].into_iter().zip([0.3, 0.7, 1.0]) {
                    assert!((actual - expected).abs() < 2e-6);
                }
            }
        }
    }

    #[test]
    fn default_bleaching_has_an_eight_second_half_life() {
        let mut dye = Dye::new();
        dye.deposit([0.5, 0.5], 0.08, 0.02, [0.5, 0.3, 0.2]);
        let initial = mass(&dye);
        dye.advance(Duration::from_secs(8), |_| [0.0; 2], Boundary::Periodic);
        for (remaining, initial) in mass(&dye).into_iter().zip(initial) {
            eprintln!(
                "default bleaching after 8 s: remaining fraction={:.9}",
                remaining / initial
            );
            assert!((remaining / initial - 0.5).abs() < 2e-5);
        }
    }

    #[test]
    fn bleaching_rate_four_has_a_two_second_half_life_and_zero_disables_it() {
        let mut dye = Dye::new();
        dye.set_bleach(4.0);
        dye.deposit([0.5, 0.5], 0.08, 0.02, [0.5, 0.3, 0.2]);
        let initial = mass(&dye);
        dye.advance(Duration::from_secs(2), |_| [0.0; 2], Boundary::Periodic);
        for (remaining, initial) in mass(&dye).into_iter().zip(initial) {
            eprintln!(
                "bleaching 4 after 2 s: remaining fraction={:.9}",
                remaining / initial
            );
            assert!((remaining / initial - 0.5).abs() < 2e-5);
        }
        dye.set_bleach(0.0);
        let after_disable = mass(&dye);
        let ledger_after_disable = dye.bleached_mass;
        dye.advance(Duration::from_secs(2), |_| [0.0; 2], Boundary::Periodic);
        for (remaining, original) in mass(&dye).into_iter().zip(after_disable) {
            assert!((remaining / original - 1.0).abs() < 1e-6);
        }
        assert_eq!(dye.bleached_mass, ledger_after_disable);
    }

    #[test]
    fn bleaching_preserves_the_mass_normalized_transported_shape() {
        let mut clear = conservative_dye();
        let mut fading = Dye::new();
        for dye in [&mut clear, &mut fading] {
            dye.deposit([0.45, 0.53], 0.07, 0.02, [0.5, 0.3, 0.2]);
            dye.advance_with_streamfunction(
                Duration::from_secs(2),
                |_| ([0.2, 0.1], 0.1),
                &periodic_streamfunction(),
                Boundary::Periodic,
            );
        }
        let clear_mass = mass(&clear);
        let fading_mass = mass(&fading);
        assert!(fading_mass[0] < 0.9 * clear_mass[0]);
        for k in 0..3 {
            let error: f64 = clear
                .cells
                .iter()
                .zip(&fading.cells)
                .map(|(a, b)| {
                    (a[k] as f64 / clear_mass[k] - b[k] as f64 / fading_mass[k]).abs()
                        * DX as f64
                        * DY as f64
                })
                .sum();
            eprintln!("bleached/conservative normalized shape L1, species {k}: {error:.9}");
            assert!(error < 2e-5);
        }
    }

    #[test]
    fn repeated_stationary_sources_approach_a_finite_colored_mass() {
        let mut dye = Dye::new();
        let amount = 0.001f32;
        let duration = Duration::from_millis(200);
        let steps = 200;
        for _ in 0..steps {
            dye.deposit([0.5, 0.5], 0.08, amount, [0.5, 0.3, 0.2]);
            dye.advance(duration, |_| [0.0; 2], Boundary::Periodic);
        }
        let survival = (-std::f64::consts::LN_2 / 8.0 * duration.as_secs_f64()).exp();
        let equilibrium = amount as f64 * survival / (1.0 - survival);
        let expected = equilibrium * (1.0 - survival.powi(steps));
        let total: f64 = mass(&dye).into_iter().sum();
        eprintln!(
            "repeated sources at 40 s: active mass={total:.9}, expected={expected:.9}, \
             asymptote={equilibrium:.9}"
        );
        assert!(total < equilibrium);
        assert!((total / expected - 1.0).abs() < 1e-4);
        let removed: f64 = dye.bleached_mass.iter().sum();
        assert!(((total + removed) / (steps as f64 * amount as f64) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn active_bleached_and_exported_species_close_the_mass_ledger() {
        for boundary in [Boundary::Periodic, Boundary::OpenChannel] {
            let mut dye = Dye::new();
            dye.set_boundary(boundary);
            dye.deposit([0.78, 0.5], 0.07, 0.02, [0.5, 0.3, 0.2]);
            let initial = mass(&dye);
            dye.advance(Duration::from_secs(3), |_| [0.25, 0.0], boundary);
            let remaining = mass(&dye);
            for k in 0..3 {
                let residual =
                    (remaining[k] + dye.bleached_mass[k] + dye.boundary_loss[k]) / initial[k] - 1.0;
                eprintln!(
                    "bleach ledger {boundary:?} species {k}: relative residual={residual:.12}"
                );
                assert!(residual.abs() < 1e-5);
                assert!(dye.bleached_mass[k] > 0.05 * initial[k]);
                if boundary == Boundary::OpenChannel {
                    assert!(dye.boundary_loss[k] > 0.5 * initial[k]);
                } else {
                    assert_eq!(dye.boundary_loss[k], 0.0);
                }
            }
            assert!(
                dye.cells
                    .iter()
                    .flatten()
                    .all(|c| c.is_finite() && *c >= 0.0)
            );
        }
    }

    fn centroid_and_variance(dye: &Dye) -> ([f64; 2], [f64; 2]) {
        let mut sum = 0.0;
        let mut mean = [0.0; 2];
        let mut second = [0.0; 2];
        for y in 0..NY {
            for x in 0..NX {
                let c = dye.cells[y * NX + x][0] as f64;
                let p = [(x as f64 + 0.5) * DX as f64, (y as f64 + 0.5) * DY as f64];
                sum += c;
                for axis in 0..2 {
                    mean[axis] += p[axis] * c;
                    second[axis] += p[axis] * p[axis] * c;
                }
            }
        }
        for axis in 0..2 {
            mean[axis] /= sum;
            second[axis] = second[axis] / sum - mean[axis] * mean[axis];
        }
        (mean, second)
    }

    #[test]
    fn deposit_preserves_the_requested_pigment_mass_and_radius() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.06, 0.012, [2.0, 1.0, 0.0]);
        let m = mass(&dye);
        assert!((m[0] - 0.008).abs() < 1e-7, "{m:?}");
        assert!((m[1] - 0.004).abs() < 1e-7);
        assert_eq!(m[2], 0.0);
        let (center, variance) = centroid_and_variance(&dye);
        for axis in 0..2 {
            assert!((center[axis] - 1.0).abs() < 1e-6);
            assert!((variance[axis] - 0.06f64.powi(2)).abs() < 0.0001);
        }
        assert!(dye.sample([0.5, 0.5])[0] > 0.2);
    }

    #[test]
    fn periodic_mixing_preserves_all_species_for_thirty_two_seconds() {
        let mut dye = conservative_dye();
        dye.deposit([0.37, 0.41], 0.05, 0.02, [0.6, 0.3, 0.1]);
        let initial = mass(&dye);
        dye.advance(
            Duration::from_secs(32),
            |_| [0.17, -0.13],
            Boundary::Periodic,
        );
        let final_mass = mass(&dye);
        for k in 0..3 {
            let relative = final_mass[k] / initial[k] - 1.0;
            eprintln!("32-second periodic mass change, species {k}: {relative:.12}");
            assert!(
                relative.abs() < 0.0001,
                "species {k}: relative mass change {relative}"
            );
        }
    }

    #[test]
    fn arbitrarily_dilute_species_are_not_discarded() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.1, 1e-12, [1.0, 0.0, 0.0]);
        let initial = mass(&dye)[0];
        dye.advance(TICK, |_| [0.0; 2], Boundary::Periodic);
        assert!(
            dye.is_active(),
            "a finite pigment mass must remain represented"
        );
        assert!((mass(&dye)[0] / initial - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_translated_narrow_filament_keeps_its_width_and_shape() {
        let mut dye = conservative_dye();
        let sigma = 0.035f32;
        let start = 0.55f32;
        for y in 0..NY {
            for x in 0..NX {
                let p = (x as f32 + 0.5) * DX;
                dye.cells[y * NX + x][0] = (-0.5 * ((p - start) / sigma).powi(2)).exp();
            }
        }
        dye.active = true;
        let initial_mass = mass(&dye)[0];
        let initial_variance = centroid_and_variance(&dye).1[0];
        dye.advance(Duration::from_secs(1), |_| [0.4, 0.0], Boundary::Periodic);
        let (center, variance) = centroid_and_variance(&dye);
        let relative_spread = (variance[0] - initial_variance) / initial_variance;
        let mass_scale = initial_mass / mass(&dye)[0];
        let mut shape_error = 0.0f64;
        for y in 0..NY {
            for x in 0..NX {
                let p = (x as f32 + 0.5) * DX;
                let exact = (-0.5 * ((p - start - 0.4) / sigma).powi(2)).exp() as f64;
                shape_error += (dye.cells[y * NX + x][0] as f64 * mass_scale - exact).abs()
                    * DX as f64
                    * DY as f64;
            }
        }
        shape_error /= initial_mass;
        eprintln!(
            "translation: relative variance growth={relative_spread:.9}, mass-normalized \
             L1={shape_error:.9}, center_error={:.9}",
            center[0] - (start + 0.4) as f64
        );
        // At only 1.68 cells per sigma this front is under-resolved. Require a
        // material reduction from minmod's 1.965 variance growth / 0.526 L1;
        // the resolved-filament test below imposes a tighter absolute bound.
        assert!(
            relative_spread < 0.70,
            "narrow filament broadened by {relative_spread:.6}"
        );
        assert!(
            shape_error < 0.32,
            "relative L1 shape error {shape_error:.6}"
        );
        assert!((center[0] - (start + 0.4) as f64).abs() < 0.002);
    }

    #[test]
    fn a_resolved_translated_filament_has_small_variance_error() {
        let mut dye = conservative_dye();
        let sigma = 0.06f32;
        for y in 0..NY {
            for x in 0..NX {
                let p = (x as f32 + 0.5) * DX;
                dye.cells[y * NX + x][0] = (-0.5 * ((p - 0.55) / sigma).powi(2)).exp();
            }
        }
        dye.active = true;
        let initial = centroid_and_variance(&dye).1[0];
        dye.advance(Duration::from_secs(1), |_| [0.4, 0.0], Boundary::Periodic);
        let variance_error = (centroid_and_variance(&dye).1[0] - initial) / initial;
        eprintln!("resolved translation: relative variance growth={variance_error:.9}");
        assert!(variance_error < 0.15, "variance growth={variance_error}");
    }

    fn periodic_streamfunction() -> Vec<f32> {
        let mut streamfunction = vec![0.0; (NX + 1) * (NY + 1)];
        for y in 0..NY {
            for x in 0..NX {
                streamfunction[y * (NX + 1) + x] = 0.025
                    * (std::f32::consts::TAU * x as f32 / NX as f32).sin()
                    * (std::f32::consts::TAU * y as f32 / NY as f32).sin();
            }
        }
        for y in 0..NY {
            streamfunction[y * (NX + 1) + NX] = streamfunction[y * (NX + 1)];
        }
        for x in 0..=NX {
            streamfunction[NY * (NX + 1) + x] = streamfunction[x];
        }
        streamfunction
    }

    #[test]
    fn discrete_streamfunction_moves_pigment_without_compressing_a_uniform_mixture() {
        let psi = periodic_streamfunction();
        let mut uniform = conservative_dye();
        uniform.cells.fill([0.3, 0.7, 1.0]);
        uniform.active = true;
        uniform.advance_with_streamfunction(
            Duration::from_secs(1),
            |_| ([0.0; 2], 1.0),
            &psi,
            Boundary::Periodic,
        );
        let mut error = 0.0f32;
        for cell in &uniform.cells {
            for (actual, expected) in cell.iter().zip([0.3, 0.7, 1.0]) {
                error = error.max((actual - expected).abs());
            }
        }
        eprintln!("uniform discrete curl concentration max error: {error:.12}");
        assert!(
            error < 2e-5,
            "discrete curl compressed a uniform tracer: {error}"
        );
        let mut blob = conservative_dye();
        blob.deposit([0.35, 0.35], 0.04, 0.01, [1.0, 0.0, 0.0]);
        let original = blob.cells.clone();
        let initial = mass(&blob)[0];
        blob.advance_with_streamfunction(
            Duration::from_millis(500),
            |_| ([0.0; 2], 1.0),
            &psi,
            Boundary::Periodic,
        );
        let displacement: f32 = blob
            .cells
            .iter()
            .zip(original)
            .map(|(a, b)| (a[0] - b[0]).abs())
            .sum();
        assert!(
            displacement > 1.0,
            "streamfunction must actually transport pigment: {displacement}"
        );
        assert!((mass(&blob)[0] / initial - 1.0).abs() < 1e-4);
        assert!(
            blob.cells
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
    }

    #[test]
    fn deposits_follow_periodic_seams_without_wrapping_across_open_ends() {
        let mut periodic = conservative_dye();
        periodic.deposit([0.0, 0.0], 0.06, 0.01, [1.0, 0.0, 0.0]);
        assert!(periodic.sample([0.98, 0.98])[0] > 0.05);
        assert!((mass(&periodic)[0] - 0.01).abs() < 1e-7);
        let mut open = conservative_dye();
        open.set_boundary(Boundary::OpenChannel);
        open.deposit([0.0, 0.0], 0.06, 0.01, [1.0, 0.0, 0.0]);
        assert!(open.sample([0.02, 0.98])[0] > 0.05);
        assert_eq!(open.sample([0.98, 0.02]), [0.0; 3]);
        assert!((mass(&open)[0] - 0.01).abs() < 1e-7);
    }

    #[test]
    fn uniform_flow_transports_mass_at_its_physical_velocity() {
        let mut dye = conservative_dye();
        dye.deposit([0.3, 0.45], 0.07, 0.01, [1.0, 0.0, 0.0]);
        let initial_mass = mass(&dye)[0];
        let initial_center = centroid_and_variance(&dye).0;
        dye.advance(Duration::from_secs(1), |_| [0.28, 0.12], Boundary::Periodic);
        let (center, _) = centroid_and_variance(&dye);
        assert!(
            (center[0] - initial_center[0] - 0.28).abs() < 0.002,
            "{center:?}"
        );
        assert!((center[1] - initial_center[1] - 0.12).abs() < 0.002);
        assert!((mass(&dye)[0] / initial_mass - 1.0).abs() < 1e-4);
    }

    #[test]
    fn diffusion_spreads_pigment_without_creating_or_destroying_it() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.05, 0.01, [1.0, 0.0, 0.0]);
        let initial_mass = mass(&dye)[0];
        let variance = centroid_and_variance(&dye).1;
        let diffusivity = 0.0003;
        dye.diffusivity.fill(diffusivity);
        dye.resolve_diffusion_faces();
        for _ in 0..125 {
            dye.transport(0.008);
        }
        let after = centroid_and_variance(&dye).1;
        for axis in 0..2 {
            let expected = variance[axis] + 2.0 * diffusivity as f64;
            assert!((after[axis] - expected).abs() < 2e-7);
        }
        assert!((mass(&dye)[0] / initial_mass - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unequal_depth_does_not_diffuse_a_uniform_volume_concentration() {
        let mut dye = conservative_dye();
        for y in 0..NY {
            for x in 0..NX {
                let h = 0.1 + 0.08 * (std::f32::consts::TAU * (x as f32 + 0.5) / NX as f32).sin();
                let i = y * NX + x;
                dye.depth[i] = h;
                dye.cells[i] = [0.2 * h, 0.7 * h, h];
            }
        }
        dye.diffusivity.fill(0.0004);
        dye.resolve_diffusion_faces();
        let original = dye.cells.clone();
        for _ in 0..200 {
            dye.transport(0.005);
        }
        let error = dye
            .cells
            .iter()
            .zip(original)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0.0f32, f32::max);
        eprintln!("uniform volume concentration, unequal depth: max areal error={error:.12}");
        assert!(
            error < 1e-7,
            "diffusion followed areal mass instead of concentration: {error}"
        );
    }

    #[test]
    fn depth_weighted_diffusion_moves_down_the_volume_concentration_gradient() {
        let mut dye = conservative_dye();
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                dye.depth[i] = if x < NX / 2 { 0.002 } else { 0.7 };
                dye.cells[i] = [0.02, 0.005, 0.009];
            }
        }
        dye.diffusivity.fill(MAX_EDDY_DIFFUSIVITY);
        let rate = dye.resolve_diffusion_faces();
        let initial = mass(&dye);
        let dt = (0.8 / rate).min(0.002);
        for _ in 0..200 {
            dye.transport(dt);
        }
        let final_mass = mass(&dye);
        let shallow = dye.cells[NY / 2 * NX + NX / 2 - 1][0];
        let deep = dye.cells[NY / 2 * NX + NX / 2][0];
        assert!(
            shallow < 0.02 && deep > 0.02,
            "volume concentrations must mix: {shallow}, {deep}"
        );
        for k in 0..3 {
            assert!((final_mass[k] / initial[k] - 1.0).abs() < 2e-5);
        }
        assert!(
            dye.cells
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
    }

    #[test]
    fn open_boundary_mass_loss_matches_the_integrated_flux() {
        for direction in [-1.0, 1.0] {
            let mut dye = conservative_dye();
            dye.deposit(
                [if direction > 0.0 { 0.88 } else { 0.12 }, 0.5],
                0.045,
                0.02,
                [0.3, 0.5, 0.2],
            );
            let initial = mass(&dye);
            dye.advance_with_streamfunction(
                Duration::from_secs(1),
                |_| ([0.7 * direction, 0.1], 0.08),
                &periodic_streamfunction(),
                Boundary::OpenChannel,
            );
            let remaining = mass(&dye);
            for k in 0..3 {
                let residual = (remaining[k] + dye.boundary_loss[k] - initial[k]) / initial[k];
                eprintln!(
                    "open flow {direction}, species {k}: relative mass ledger \
                     residual={residual:.12}"
                );
                assert!(residual.abs() < 2e-5);
                assert!(dye.boundary_loss[k] > 0.95 * initial[k]);
            }
            assert!(
                dye.cells
                    .iter()
                    .flatten()
                    .all(|c| c.is_finite() && *c >= 0.0)
            );
        }
    }

    #[test]
    fn invalid_streamfunctions_fall_back_to_the_base_velocity() {
        for psi in [
            vec![0.0; 2],
            vec![f32::NAN; (NX + 1) * (NY + 1)],
            vec![f32::INFINITY; (NX + 1) * (NY + 1)],
        ] {
            let mut actual = conservative_dye();
            let mut expected = conservative_dye();
            actual.deposit([0.4, 0.4], 0.04, 0.01, [1.0, 0.2, 0.3]);
            expected.deposit([0.4, 0.4], 0.04, 0.01, [1.0, 0.2, 0.3]);
            actual.advance_with_streamfunction(
                Duration::from_millis(40),
                |_| ([0.2, -0.1], 1.0),
                &psi,
                Boundary::Periodic,
            );
            expected.advance(
                Duration::from_millis(40),
                |_| [0.2, -0.1],
                Boundary::Periodic,
            );
            assert_eq!(actual.cells, expected.cells);
        }
    }

    #[test]
    fn a_sheared_gaussian_matches_analytic_covariance_and_remains_positive() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.06, 0.01, [1.0, 0.0, 0.0]);
        let initial = centroid_and_variance(&dye).1;
        let shear = 1.2f32;
        let diffusivity = (MOLECULAR_DIFFUSIVITY + 0.3f32.powi(2) * DX * DY * shear) as f64;
        dye.advance(
            Duration::from_secs(1),
            |point| [shear * (point[1] * LENGTH - 1.0), 0.0],
            Boundary::Periodic,
        );
        let (center, variance) = centroid_and_variance(&dye);
        let expected_x = initial[0]
            + (shear as f64).powi(2) * initial[1]
            + 2.0 * diffusivity
            + 2.0 / 3.0 * diffusivity * (shear as f64).powi(2);
        let expected_y = initial[1] + 2.0 * diffusivity;
        let expected_covariance = shear as f64 * (initial[1] + diffusivity);
        let mut covariance = 0.0;
        let mut total = 0.0;
        for y in 0..NY {
            for x in 0..NX {
                let c = dye.cells[y * NX + x][0] as f64;
                covariance += c
                    * ((x as f64 + 0.5) * DX as f64 - center[0])
                    * ((y as f64 + 0.5) * DY as f64 - center[1]);
                total += c;
            }
        }
        covariance /= total;
        let error_x = variance[0] / expected_x - 1.0;
        let error_y = variance[1] / expected_y - 1.0;
        let error_covariance = covariance / expected_covariance - 1.0;
        eprintln!(
            "linear shear: relative variance x error={error_x:.9}, y={error_y:.9}, \
             covariance={error_covariance:.9}"
        );
        assert!(error_x.abs() < 0.08 && error_y.abs() < 0.002 && error_covariance.abs() < 0.02);
        assert!(
            dye.cells
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
    }

    #[test]
    fn unresolved_mixing_responds_to_strain_but_not_rigid_rotation() {
        let mut dye = conservative_dye();
        let shear = 2.0;
        for y in 0..NY {
            for x in 0..NX {
                dye.velocity[y * NX + x] = [shear * (y as f32 + 0.5) * DY, 0.0];
            }
        }
        dye.resolve_diffusivity();
        let center = NY / 2 * NX + NX / 2;
        let expected = MOLECULAR_DIFFUSIVITY + 0.3f32.powi(2) * DX * DY * shear;
        assert!((dye.diffusivity[center] - expected).abs() < 1e-8);
        for y in 0..NY {
            for x in 0..NX {
                dye.velocity[y * NX + x] = [
                    -shear * ((y as f32 + 0.5) * DY - 1.0),
                    shear * ((x as f32 + 0.5) * DX - 1.0),
                ];
            }
        }
        dye.resolve_diffusivity();
        assert!((dye.diffusivity[center] - MOLECULAR_DIFFUSIVITY).abs() < 1e-8);
    }

    #[test]
    fn open_channel_boundary_strain_uses_the_actual_one_sided_spacing() {
        let mut dye = conservative_dye();
        dye.set_boundary(Boundary::OpenChannel);
        for y in 0..NY {
            for x in 0..NX {
                let position = (x as f32 + 0.5) * DX;
                dye.velocity[y * NX + x] = [0.4 * position, 0.7 * position];
            }
        }
        dye.resolve_diffusivity();
        let strain = (2.0 * 0.4f32.powi(2) + 0.7f32.powi(2)).sqrt();
        let expected = MOLECULAR_DIFFUSIVITY + 0.3f32.powi(2) * DX * DY * strain;
        for x in 0..NX {
            let actual = dye.diffusivity[NY / 2 * NX + x];
            assert!(
                (actual - expected).abs() < 1e-8,
                "column {x}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn invalid_inputs_and_empty_updates_cannot_create_pigment_or_index_outside_the_grid() {
        let mut dye = conservative_dye();
        dye.deposit([f32::NAN, 0.5], 0.04, 0.01, [1.0; 3]);
        dye.deposit([0.5, 0.5], f32::INFINITY, 0.01, [1.0; 3]);
        dye.deposit([0.5, 0.5], 0.04, 0.01, [f32::NAN; 3]);
        dye.advance(
            Duration::from_secs(100),
            |_| panic!("empty field must skip sampling"),
            Boundary::Periodic,
        );
        assert!(!dye.is_active());
        assert_eq!(mass(&dye), [0.0; 3]);
        dye.deposit([0.0, 1.0], 0.0001, 0.01, [1.0, 0.0, 0.0]);
        for point in [[0.0, 0.0], [1.0, 1.0], [-f32::MAX, f32::MAX], [
            f32::NAN,
            0.5,
        ]] {
            assert!(dye.sample(point).iter().all(|c| c.is_finite() && *c >= 0.0));
        }
        dye.advance(TICK, |_| [f32::NAN, f32::INFINITY], Boundary::Periodic);
        assert!(
            dye.cells
                .iter()
                .flatten()
                .all(|c| c.is_finite() && *c >= 0.0)
        );
    }

    #[test]
    fn clear_water_inflow_does_not_recycle_downstream_pigment() {
        let mut dye = conservative_dye();
        dye.deposit([0.88, 0.5], 0.04, 0.01, [1.0, 0.0, 0.0]);
        let initial = mass(&dye)[0];
        dye.advance(
            Duration::from_secs(1),
            |_| [0.7, 0.0],
            Boundary::OpenChannel,
        );
        assert!(mass(&dye)[0] < initial * 0.005, "{:#?}", mass(&dye));
        assert_eq!(dye.sample([0.02, 0.5]), [0.0; 3]);
    }

    #[test]
    fn variable_flow_is_positive_and_conservative() {
        let mut dye = conservative_dye();
        dye.deposit([0.5, 0.5], 0.035, 0.02, [0.6, 0.2, 0.2]);
        let initial = mass(&dye);
        dye.advance(
            Duration::from_secs(2),
            |p| {
                let x = std::f32::consts::TAU * p[0];
                let y = std::f32::consts::TAU * p[1];
                [1.2 * x.sin() * y.cos(), -1.2 * x.cos() * y.sin()]
            },
            Boundary::Periodic,
        );
        for cell in &dye.cells {
            for c in cell {
                assert!(c.is_finite() && *c >= 0.0, "{cell:?}");
            }
        }
        let final_mass = mass(&dye);
        for k in 0..3 {
            assert!((final_mass[k] / initial[k] - 1.0).abs() < 2e-4);
        }
    }

    #[test]
    fn fixed_ticks_are_independent_of_display_frame_partition() {
        let mut a = conservative_dye();
        let mut b = conservative_dye();
        a.deposit([0.4, 0.4], 0.04, 0.01, [0.3, 0.4, 0.3]);
        b.deposit([0.4, 0.4], 0.04, 0.01, [0.3, 0.4, 0.3]);
        a.advance(
            Duration::from_millis(731),
            |_| [0.3, -0.1],
            Boundary::Periodic,
        );
        for _ in 0..73 {
            b.advance(
                Duration::from_millis(10),
                |_| [0.3, -0.1],
                Boundary::Periodic,
            );
        }
        b.advance(
            Duration::from_millis(1),
            |_| [0.3, -0.1],
            Boundary::Periodic,
        );
        assert_eq!(a.cells, b.cells);
    }
}
