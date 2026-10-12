use std::{
    f32::consts::TAU,
    time::{
        Duration,
        Instant,
    },
};

use ndarray::Array2;

use crate::{
    drain::{
        Drain,
        TICK as DRAIN_TICK,
    },
    dye::{
        Boundary,
        Dye,
        NX as DYE_WIDTH,
        NY as DYE_HEIGHT,
        TICK as TRANSPORT_TICK,
    },
    physics::{
        DEPTH,
        PATCH_LENGTH,
        angular_frequency,
    },
    rapids::Rapids,
    ripples::Ripples,
    settings::SETTINGS,
    stirring::Stirring,
    vortex::VortexFrame,
    waves::{
        PressureImpact,
        RadialWave,
        spectral_weight,
    },
    world_flow::{
        CURRENT,
        WorldFlow,
    },
};

const MAX_WAVES: usize = 128;
// Broad crossing waves with quieter fine resonances. Shorter modes carry less
// height and drift slowly, avoiding a rapidly sparkling surface.
const SWELL_MODES: [(f32, f32, f32); 10] = [
    (1.0, 0.0, 0.0063),
    (0.0, 1.0, 0.0049),
    (2.0, 1.0, 0.0042),
    (-1.0, 2.0, 0.00315),
    (2.4, 1.8, 0.0028),
    (-4.5, 3.4, 0.0021),
    (3.0, -4.0, 0.00175),
    (7.2, 2.9, 0.00126),
    (-8.6, -7.2, 0.0007),
    (13.0, -13.7, 0.00042),
];

#[inline]
fn stokes_bound(
    fundamental: f32,
    amplitude: f32,
    coefficient: f32,
) -> f32 {
    coefficient * (2.0 * fundamental * fundamental - amplitude * amplitude)
}

#[inline]
fn sample_surface(
    levels: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> f32 {
    let fx = (x * width as f32).clamp(0.0, (width - 1) as f32);
    let fy = (y * height as f32).clamp(0.0, (height - 1) as f32);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let a = levels[x0 * height + y0] * (1.0 - tx) + levels[x1 * height + y0] * tx;
    let b = levels[x0 * height + y1] * (1.0 - tx) + levels[x1 * height + y1] * tx;
    a + (b - a) * ty
}

/// Extend the analytic carrier on demand at the same reconstruction lattice.
/// Swirl backtraces can leave the material window; clamping those taps would
/// tether a wave to the moving border and change it whenever the window shifts.
#[inline]
fn sample_extended_surface(
    levels: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    outside: &impl Fn(f32, f32) -> f32,
) -> f32 {
    let fx = x * width as f32;
    let fy = y * height as f32;
    if fx >= 0.0 && fy >= 0.0 && fx <= (width - 1) as f32 && fy <= (height - 1) as f32 {
        return sample_surface(levels, width, height, x, y);
    }
    let ix = fx.floor() as isize;
    let iy = fy.floor() as isize;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let fetch = |col: isize, row: isize| {
        if (0..width as isize).contains(&col) && (0..height as isize).contains(&row) {
            levels[col as usize * height + row as usize]
        } else {
            outside(col as f32 / width as f32, row as f32 / height as f32)
        }
    };
    let a = fetch(ix, iy) * (1.0 - tx) + fetch(ix + 1, iy) * tx;
    let b = fetch(ix, iy + 1) * (1.0 - tx) + fetch(ix + 1, iy + 1) * tx;
    a + (b - a) * ty
}

#[derive(Debug)]
struct PhysicalWaveCache {
    key: (u32, u32, usize, usize),
    wave: Option<RadialWave>,
}

#[derive(Debug)]
pub struct WaveCenter {
    pub x: f32,
    pub y: f32,
    pub strength: f32,
    pub size: f32,
    pub time: Instant,
    physical: Option<PressureImpact>,
    physical_cache: Option<Box<PhysicalWaveCache>>,
}

impl WaveCenter {
    pub fn new(
        x: f32,
        y: f32,
        strength: f32,
        time: Instant,
    ) -> Self {
        Self {
            x,
            y,
            strength,
            size: 1.0,
            time,
            physical: None,
            physical_cache: None,
        }
    }
}

#[derive(Debug)]
pub struct Water {
    pub levels: Array2<f32>,
    pub depths: Vec<f32>,
    pub foam: Vec<f32>,
    pub surface_velocity: Vec<[f32; 2]>,
    pub dye: Vec<[f32; 3]>,
    pub time: Instant,
    pub wave_centers: Vec<WaveCenter>,
    elapsed: Duration,
    coupled_elapsed: Duration,
    coupled_pending: Duration,
    swell_rows: Vec<[(f32, f32); SWELL_MODES.len()]>,
    advected_levels: Vec<f32>,
    rapids: Option<Rapids>,
    rapids_requested: bool,
    rapids_blend: f32,
    drain: Option<Drain>,
    drain_ripples: Option<Ripples>,
    drain_requested: bool,
    drain_blend: f32,
    dissolved: Dye,
    stirring: Stirring,
    transport_streamfunction: Vec<f32>,
    drop_sequence: u32,
    current_displacement: [f64; 2],
    current_enabled: bool,
    world_enabled: bool,
    world_seed: u64,
    world_origin: [f64; 2],
    world_cells: [i64; 2],
    world_flow: Option<WorldFlow>,
}

impl Water {
    pub fn new(
        width: u16,
        height: u16,
    ) -> Self {
        let mut water = Self {
            levels: Array2::zeros((width as usize, height as usize)),
            depths: vec![DEPTH; width as usize * height as usize],
            foam: vec![0.0; width as usize * height as usize],
            surface_velocity: vec![[0.0; 2]; width as usize * height as usize],
            dye: vec![[0.0; 3]; width as usize * height as usize],
            time: Instant::now(),
            wave_centers: Vec::with_capacity(MAX_WAVES),
            elapsed: Duration::ZERO,
            coupled_elapsed: Duration::ZERO,
            coupled_pending: Duration::ZERO,
            swell_rows: Vec::new(),
            advected_levels: Vec::new(),
            rapids: None,
            rapids_requested: false,
            rapids_blend: 0.0,
            drain: None,
            drain_ripples: None,
            drain_requested: false,
            drain_blend: 0.0,
            dissolved: Dye::new(),
            stirring: Stirring::new(),
            transport_streamfunction: vec![0.0; (DYE_WIDTH + 1) * (DYE_HEIGHT + 1)],
            drop_sequence: 0,
            current_displacement: [0.0; 2],
            current_enabled: true,
            world_enabled: false,
            world_seed: 0,
            world_origin: [-1.0; 2],
            world_cells: [0; 2],
            world_flow: None,
        };
        water.update(Duration::ZERO);
        water
    }

    /// Select streamed Classic geometry. Camera travel can be paused without
    /// disabling this model: simply leave its requested world centre unchanged.
    pub fn configure_world(
        &mut self,
        seed: u64,
        enabled: bool,
    ) {
        self.world_seed = seed;
        self.world_enabled = enabled;
        self.refresh_world_flow();
        self.dissolved.set_boundary(self.transport_boundary());
    }

    fn refresh_world_flow(&mut self) {
        self.world_flow = self.world_enabled.then(|| {
            WorldFlow::new(
                self.world_origin,
                self.world_seed,
                self.elapsed.as_secs_f64(),
                [self.width() as usize, self.height() as usize],
            )
        });
    }

    fn streaming_active(&self) -> bool {
        self.world_enabled
            && !self.rapids_requested
            && !self.drain_requested
            && self.rapids_blend == 0.0
            && self.drain_blend == 0.0
    }

    /// Rebase by whole pigment cells: overlap is copied exactly, never sampled.
    /// Analytic waves and persistent impact/wake positions keep world identity.
    /// Exiting pigment is an open-window loss; entering water starts clear.
    pub fn set_world_center(
        &mut self,
        center: [f64; 2],
    ) {
        if !self.streaming_active() || center.iter().any(|p| !p.is_finite() || p.abs() > 1e12) {
            return;
        }
        let spacing = [
            PATCH_LENGTH as f64 / DYE_WIDTH as f64,
            PATCH_LENGTH as f64 / DYE_HEIGHT as f64,
        ];
        let cells: [i64; 2] = std::array::from_fn(|i| (center[i] / spacing[i]).round() as i64);
        let shift: [isize; 2] = std::array::from_fn(|i| (cells[i] - self.world_cells[i]) as isize);
        if shift == [0; 2] {
            return;
        }
        let origin: [f64; 2] = std::array::from_fn(|i| -1.0 + cells[i] as f64 * spacing[i]);
        let delta: [f32; 2] = std::array::from_fn(|i| (origin[i] - self.world_origin[i]) as f32);
        self.dissolved.shift_cells(shift);
        self.stirring.shift_origin(delta);
        for center in &mut self.wave_centers {
            center.x -= delta[0] / PATCH_LENGTH;
            center.y -= delta[1] / PATCH_LENGTH;
        }
        self.world_cells = cells;
        self.world_origin = origin;
        self.refresh_world_flow();
    }

    pub fn world_origin(&self) -> [f64; 2] { self.world_origin }

    /// Column-current sample at a resolved lattice node, in metres per second.
    /// The native volume owns its impact wake; this supplies the ambient flow.
    #[allow(dead_code)] // Called by the separately built native-core crate.
    pub fn native_transport_velocity(
        &self,
        x: usize,
        y: usize,
    ) -> [f32; 2] {
        if x >= self.width() as usize || y >= self.height() as usize {
            return [0.0; 2];
        }
        let mut velocity = self.surface_velocity[x * self.height() as usize + y];
        let blend = self.material_blends().0;
        if let Some(rapids) = self.rapids.as_ref().filter(|_| blend > 0.0) {
            let point = [
                x as f32 / self.width() as f32,
                y as f32 / self.height() as f32,
            ];
            let mean = rapids.depth_averaged_velocity(point);
            let surface = rapids.sample(point).velocity;
            for axis in 0..2 {
                velocity[axis] += (mean[axis] - surface[axis]) * blend;
            }
        }
        velocity
    }

    /// Change future through-flow without resetting accumulated wave phase.
    pub fn set_current(
        &mut self,
        enabled: bool,
    ) {
        self.current_enabled = enabled;
    }

    fn through_current(&self) -> [f32; 2] {
        if self.world_enabled && self.current_enabled {
            CURRENT
        } else {
            [0.0; 2]
        }
    }

    /// Keep impacts and simulation time while resampling the continuous surface.
    pub fn resize(
        &mut self,
        width: u16,
        height: u16,
    ) {
        if self.width() != width || self.height() != height {
            self.levels = Array2::zeros((width as usize, height as usize));
            self.refresh_world_flow();
            self.update(Duration::ZERO);
        }
    }

    #[inline]
    pub fn height(&self) -> u16 { self.levels.shape()[1] as u16 }

    #[inline]
    pub fn width(&self) -> u16 { self.levels.shape()[0] as u16 }

    pub fn has_rapids(&self) -> bool { self.rapids_blend > 0.0 }

    pub fn has_drain(&self) -> bool { self.drain_blend > 0.0 }

    pub fn has_numerical_flow(&self) -> bool { self.has_rapids() || self.has_drain() }

    pub fn set_drain(
        &mut self,
        enabled: bool,
    ) {
        if enabled {
            if self.drain.is_none() {
                self.drain = Some(Drain::new());
                self.drain_ripples = Some(Ripples::new());
            }
            self.rapids_requested = false;
        }
        self.drain_requested = enabled;
        self.dissolved.set_boundary(self.transport_boundary());
    }

    pub fn has_dye(&self) -> bool { self.dissolved.is_active() }

    pub fn set_bleach(
        &mut self,
        value: f32,
    ) {
        self.dissolved.set_bleach(value);
    }

    pub fn elapsed(&self) -> Duration { self.elapsed }

    pub fn set_rapids(
        &mut self,
        enabled: bool,
    ) {
        if enabled && self.rapids.is_none() {
            self.rapids = Some(Rapids::new());
        }
        if enabled {
            self.drain_requested = false;
        }
        self.rapids_requested = enabled;
        self.dissolved.set_boundary(self.transport_boundary());
    }

    fn transport_boundary(&self) -> Boundary {
        if self.world_enabled && !self.drain_requested && self.rapids_blend == 0.0 {
            Boundary::OpenPlane
        } else if !self.drain_requested && self.rapids_blend > 0.5 {
            Boundary::OpenChannel
        } else {
            Boundary::Periodic
        }
    }

    /// Hydro and pigment consume the same integer ticks. In particular, a long
    /// display frame never forces old pigment ticks with a future flow field.
    fn advance_coupled(
        &mut self,
        delay: Duration,
    ) {
        self.coupled_pending += delay;
        let tick = if self.drain_requested {
            DRAIN_TICK
        } else {
            TRANSPORT_TICK
        };
        while self.coupled_pending >= tick {
            self.coupled_pending -= tick;
            self.coupled_elapsed += tick;
            let through = self.through_current();
            if self.streaming_active() {
                for (displacement, velocity) in self.current_displacement.iter_mut().zip(through) {
                    *displacement += velocity as f64 * tick.as_secs_f64();
                }
                if let Some(flow) = &self.world_flow {
                    for center in &mut self.wave_centers {
                        let point = [center.x, center.y];
                        let a = flow.velocity(point);
                        let middle = std::array::from_fn(|axis| {
                            point[axis]
                                + (a[axis] + through[axis]) * tick.as_secs_f32() * 0.5
                                    / PATCH_LENGTH
                        });
                        let b = flow.velocity(middle);
                        center.x += (b[0] + through[0]) * tick.as_secs_f32() / PATCH_LENGTH;
                        center.y += (b[1] + through[1]) * tick.as_secs_f32() / PATCH_LENGTH;
                    }
                }
            }
            if let Some(flow) = self
                .rapids
                .as_mut()
                .filter(|_| self.rapids_requested || self.rapids_blend > 0.0)
            {
                flow.advance(tick);
            }
            let step = tick.as_secs_f32() / 0.45;
            self.rapids_blend = if self.rapids_requested {
                (self.rapids_blend + step).min(1.0)
            } else {
                (self.rapids_blend - step).max(0.0)
            };
            self.drain_blend = if self.drain_requested {
                (self.drain_blend + step).min(1.0)
            } else {
                (self.drain_blend - step).max(0.0)
            };
            // The pressure-impulse closure follows the local mean flow. Its
            // short waves are superposed after hydraulic blending, so they must
            // also advect in the numerical modes rather than remain pinned.
            if !self.streaming_active() {
                for center in self
                    .wave_centers
                    .iter_mut()
                    .filter(|c| c.physical.is_some())
                {
                    let velocity = |point| {
                        if self.drain_requested {
                            self.drain
                                .as_ref()
                                .map_or([0.0; 2], |flow| flow.sample(point).velocity)
                        } else if self.rapids_requested {
                            self.rapids
                                .as_ref()
                                .map_or([0.0; 2], |flow| flow.depth_averaged_velocity(point))
                        } else {
                            [0.0; 2]
                        }
                    };
                    let point = [center.x, center.y];
                    let a = velocity(point);
                    let middle = std::array::from_fn(|axis| {
                        point[axis] + a[axis] * tick.as_secs_f32() * 0.5 / PATCH_LENGTH
                    });
                    let b = velocity(middle);
                    center.x += b[0] * tick.as_secs_f32() / PATCH_LENGTH;
                    center.y += b[1] * tick.as_secs_f32() / PATCH_LENGTH;
                }
            }
            let boundary = self.transport_boundary();
            self.dissolved.set_boundary(boundary);
            if self.drain_requested {
                // Accepted hydraulic stages own both water and pigment. The
                // display transition must never mix or duplicate these fluxes.
                if let Some(flow) = self.drain.as_mut() {
                    flow.advance(tick, &mut self.dissolved);
                    if let Some(ripples) = self.drain_ripples.as_mut() {
                        ripples.advance(tick, |point| {
                            let sample = flow.sample(point);
                            (sample.velocity, sample.depth)
                        });
                    }
                }
                continue;
            }
            // An inactive Drain retains its last hydraulic state while the
            // displayed surface fades out; it cannot remove hidden pigment.
            if self.dissolved.is_active() || self.stirring.is_active() {
                let flow = self.rapids.as_ref();
                let blend = self.rapids_blend * self.rapids_blend * (3.0 - 2.0 * self.rapids_blend);
                let world = self.world_flow.as_ref();
                let vortex = (blend < 1.0 && world.is_none()).then(|| {
                    VortexFrame::new(self.coupled_elapsed.as_secs_f64(), DYE_WIDTH, DYE_HEIGHT)
                });
                let mean_velocity = |point| {
                    let mut velocity = vortex.as_ref().map_or([0.0; 2], |vortex| {
                        vortex.periodic_velocity(point).map(|v| v * (1.0 - blend))
                    });
                    if let Some(world) = world {
                        let current = world.velocity(point);
                        for axis in 0..2 {
                            velocity[axis] += (current[axis] + through[axis]) * (1.0 - blend);
                        }
                    }
                    if let Some(flow) = flow.filter(|_| blend > 0.0) {
                        let current = flow.depth_averaged_velocity(point);
                        for axis in 0..2 {
                            velocity[axis] += current[axis] * blend;
                        }
                    }
                    velocity
                };
                self.stirring
                    .advance(TRANSPORT_TICK, mean_velocity, boundary);
                if !self.dissolved.is_active() {
                    continue;
                }
                // The discrete curl of this shared vertex field gives dye
                // exactly divergence-free periodic stirring at its own faces.
                if let Some(world) = world {
                    for y in 0..=DYE_HEIGHT {
                        for x in 0..=DYE_WIDTH {
                            let point = [x as f32 / DYE_WIDTH as f32, y as f32 / DYE_HEIGHT as f32];
                            self.transport_streamfunction[y * (DYE_WIDTH + 1) + x] =
                                world.streamfunction(point) * (1.0 - blend);
                        }
                    }
                } else if let Some(vortex) = &vortex {
                    for y in 0..=DYE_HEIGHT {
                        for x in 0..=DYE_WIDTH {
                            let point = [x as f32 / DYE_WIDTH as f32, y as f32 / DYE_HEIGHT as f32];
                            self.transport_streamfunction[y * (DYE_WIDTH + 1) + x] =
                                vortex.periodic_streamfunction(point) * (1.0 - blend);
                        }
                    }
                } else {
                    self.transport_streamfunction.fill(0.0);
                }
                self.stirring.fill_streamfunction(
                    &mut self.transport_streamfunction,
                    DYE_WIDTH,
                    DYE_HEIGHT,
                    boundary,
                );
                self.dissolved.advance_with_streamfunction(
                    TRANSPORT_TICK,
                    |point| {
                        if let Some(flow) = flow.filter(|_| blend > 0.0) {
                            let current = flow.depth_averaged_velocity(point);
                            let velocity = std::array::from_fn(|axis| {
                                current[axis] * blend + through[axis] * (1.0 - blend)
                            });
                            let depth = DEPTH + (flow.sample(point).depth - DEPTH) * blend;
                            (velocity, depth)
                        } else {
                            (through, DEPTH)
                        }
                    },
                    &self.transport_streamfunction,
                    boundary,
                );
            }
        }
    }

    fn material_blends(&self) -> (f32, f32) {
        let smooth = |blend: f32| blend * blend * (3.0 - 2.0 * blend);
        let rapids = smooth(self.rapids_blend);
        let drain = smooth(self.drain_blend);
        // Mutually exclusive targets keep the sum <= 1 analytically. Normalize
        // only possible roundoff when crossing between numerical modes.
        let total = (rapids + drain).max(1.0);
        (rapids / total, drain / total)
    }

    fn resolve_materials(&mut self) {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let (rapids_blend, drain_blend) = self.material_blends();
        let classic_blend = (1.0 - rapids_blend - drain_blend).max(0.0);
        let levels = self.levels.as_slice_mut().unwrap();
        self.depths.resize(levels.len(), DEPTH);
        self.foam.resize(levels.len(), 0.0);
        self.surface_velocity.resize(levels.len(), [0.0; 2]);
        self.dye.resize(levels.len(), [0.0; 3]);
        let dyed = self.dissolved.is_active();
        let through = if self.world_enabled && self.current_enabled {
            CURRENT
        } else {
            [0.0; 2]
        };
        for x in 0..width {
            for y in 0..height {
                let point = [x as f32 / width as f32, y as f32 / height as f32];
                let i = x * height + y;
                self.dye[i] = if dyed {
                    self.dissolved.sample(point)
                } else {
                    [0.0; 3]
                };
                self.depths[i] = (DEPTH + levels[i]).max(0.001) * classic_blend;
                levels[i] *= classic_blend;
                self.foam[i] = 0.0;
                self.surface_velocity[i] = self.world_flow.as_ref().map_or([0.0; 2], |flow| {
                    let velocity = flow.velocity(point);
                    std::array::from_fn(|axis| (velocity[axis] + through[axis]) * classic_blend)
                });
                if let Some(flow) = self.rapids.as_ref().filter(|_| rapids_blend > 0.0) {
                    let sample = flow.sample(point);
                    levels[i] += sample.elevation * rapids_blend;
                    self.depths[i] += sample.depth * rapids_blend;
                    self.foam[i] += sample.foam * rapids_blend;
                    for axis in 0..2 {
                        self.surface_velocity[i][axis] += sample.velocity[axis] * rapids_blend;
                    }
                }
                if let Some(flow) = self.drain.as_ref().filter(|_| drain_blend > 0.0) {
                    let sample = flow.sample(point);
                    let detail = self
                        .drain_ripples
                        .as_ref()
                        .map_or(0.0, |ripples| ripples.sample(point, [width, height]));
                    levels[i] += (sample.elevation + detail) * drain_blend;
                    self.depths[i] += sample.depth * drain_blend;
                    self.foam[i] += sample.foam * drain_blend;
                    for axis in 0..2 {
                        self.surface_velocity[i][axis] += sample.velocity[axis] * drain_blend;
                    }
                }
            }
        }
    }

    /// Classic waves are analytic; numerical modes use fixed physical ticks.
    /// Viewport changes resample persistent fields without re-integrating.
    pub fn update(
        &mut self,
        delay: Duration,
    ) {
        self.time += delay;
        self.elapsed += delay;
        self.advance_coupled(delay);
        if let Some(flow) = self.world_flow.as_mut() {
            flow.set_time(self.elapsed.as_secs_f64());
        }
        self.wave_centers.retain(|wave| {
            self.time.saturating_duration_since(wave.time) < SETTINGS.attenuation_time
        });
        if self.wave_centers.len() > MAX_WAVES {
            self.wave_centers
                .drain(..self.wave_centers.len() - MAX_WAVES);
        }

        let width = self.width() as usize;
        let height = self.height() as usize;
        if width == 0 || height == 0 {
            self.depths.clear();
            self.foam.clear();
            self.surface_velocity.clear();
            self.dye.clear();
            return;
        }
        if self.rapids_blend == 1.0 || self.drain_blend == 1.0 {
            self.resolve_materials();
            self.resolve_physical_impacts();
            return;
        }
        let world_offset = if self.world_enabled {
            std::array::from_fn(|axis| {
                (self.world_origin[axis] + 1.0 - self.current_displacement[axis])
                    / PATCH_LENGTH as f64
            })
        } else {
            [0.0; 2]
        };
        let inv_width = 1.0 / width as f32;
        let inv_height = 1.0 / height as f32;
        let levels = self.levels.as_slice_mut().unwrap();
        let filtered_modes = SWELL_MODES.map(|(kx, ky, amplitude)| {
            let weight = spectral_weight((kx.abs() * inv_width).max(ky.abs() * inv_height));
            (
                kx,
                ky,
                angular_frequency((kx * kx + ky * ky).sqrt() * TAU / PATCH_LENGTH, DEPTH),
                amplitude * weight,
            )
        });
        // Second-order Stokes bound harmonics share twice the carrier phase.
        // These modes have kh >= 2.2 and small steepness; no free harmonic speed.
        let bound_modes = filtered_modes.map(|(kx, ky, ..)| {
            let k = (kx * kx + ky * ky).sqrt() * TAU / PATCH_LENGTH;
            let t = (k * DEPTH).tanh();
            let amplitude_coefficient = k * (3.0 - t * t) / (4.0 * t * t * t);
            let filter = spectral_weight(2.0 * (kx.abs() * inv_width).max(ky.abs() * inv_height));
            (amplitude_coefficient, filter)
        });

        let phases: [f32; SWELL_MODES.len()] = std::array::from_fn(|mode| {
            let (kx, ky, frequency, _) = filtered_modes[mode];
            (-self.elapsed.as_secs_f64() * frequency as f64
                + mode as f64 * 1.9
                + (world_offset[0] * kx as f64 + world_offset[1] * ky as f64)
                    * std::f64::consts::TAU)
                .rem_euclid(std::f64::consts::TAU) as f32
        });
        let continuous_swell = |x: f32, y: f32| {
            let mut swell = 0.0;
            for mode in 0..SWELL_MODES.len() {
                let (kx, ky, _, amplitude) = filtered_modes[mode];
                let fundamental = amplitude * ((x * kx + y * ky) * TAU + phases[mode]).sin();
                let (coefficient, filter) = bound_modes[mode];
                swell += fundamental + filter * stokes_bound(fundamental, amplitude, coefficient);
            }
            swell
        };

        // sin(x + y) = sin(x)cos(y) + cos(x)sin(y). Trigonometry scales
        // with rows + columns, while all cells retain continuous capillary motion.
        self.swell_rows
            .resize(height, [(0.0, 0.0); SWELL_MODES.len()]);
        for (y, row) in self.swell_rows.iter_mut().enumerate() {
            for (mode, &(_, ky, ..)) in filtered_modes.iter().enumerate() {
                let phase = phases[mode];
                row[mode] = (y as f32 * inv_height * ky * TAU + phase).sin_cos();
            }
        }
        for x in 0..width {
            let x_phases = filtered_modes.map(|(kx, _, _, amplitude)| {
                let (sine, cosine) = (x as f32 * inv_width * kx * TAU).sin_cos();
                (sine * amplitude, cosine * amplitude)
            });
            for (y, row) in self.swell_rows.iter().enumerate() {
                let mut swell = 0.0;
                for mode in 0..SWELL_MODES.len() {
                    let fundamental =
                        x_phases[mode].0 * row[mode].1 + x_phases[mode].1 * row[mode].0;
                    let amplitude = filtered_modes[mode].3;
                    let (coefficient, filter) = bound_modes[mode];
                    swell +=
                        fundamental + filter * stokes_bound(fundamental, amplitude, coefficient);
                }
                levels[x * height + y] = swell;
            }
        }

        let vortices = self
            .world_flow
            .is_none()
            .then(|| VortexFrame::new(self.elapsed.as_secs_f64(), width, height));
        self.advected_levels.resize(levels.len(), 0.0);
        let read_swell = |x, y| {
            if self.world_enabled {
                sample_extended_surface(levels, width, height, x, y, &continuous_swell)
            } else {
                sample_surface(levels, width, height, x, y)
            }
        };
        for x in 0..width {
            for y in 0..height {
                let point = [x as f32 * inv_width, y as f32 * inv_height];
                let deformation = self.world_flow.as_ref().map_or_else(
                    || vortices.as_ref().unwrap().sample(point),
                    |flow| flow.sample(point),
                );
                // Spatial antialias reconstruction enlarged by flow shear.
                // Physical viscosity is applied to the spectral impulse modes;
                // this filter does not accumulate or depend on display cadence.
                let dx = (0.45 * inv_width * deformation.stretch).max(0.007);
                let dy = (0.45 * inv_height * deformation.stretch).max(0.007);
                let [sx, sy] = deformation.source;
                let displaced = (read_swell(sx - dx, sy - dy)
                    + read_swell(sx + dx, sy - dy)
                    + read_swell(sx - dx, sy + dy)
                    + read_swell(sx + dx, sy + dy))
                    * 0.25;
                self.advected_levels[x * height + y] = displaced + deformation.elevation;
            }
        }
        levels.copy_from_slice(&self.advected_levels);
        let lifetime = SETTINGS.attenuation_time.as_secs_f32();
        let sources: Vec<_> = self
            .wave_centers
            .iter()
            .filter(|center| center.physical.is_none())
            .collect();
        let requests: Vec<_> = sources
            .iter()
            .map(|center| {
                (
                    center.strength,
                    center.size,
                    self.time
                        .saturating_duration_since(center.time)
                        .as_secs_f32(),
                )
            })
            .collect();
        let profiles = RadialWave::for_grid_waves(&requests, lifetime, width, height);
        for ((center, &(_, _, age)), wave) in sources.into_iter().zip(&requests).zip(profiles) {
            let Some(wave) = wave else {
                continue;
            };
            let position = vortices.as_ref().map_or([center.x, center.y], |vortices| {
                vortices.transport([center.x, center.y], age)
            });
            let radius = wave.radius;
            let radius2 = radius * radius;
            let x_min = ((position[0] - radius) * width as f32).floor().max(0.0) as usize;
            let x_max =
                (((position[0] + radius) * width as f32).ceil() + 1.0).min(width as f32) as usize;
            let y_min = ((position[1] - radius) * height as f32).floor().max(0.0) as usize;
            let y_max =
                (((position[1] + radius) * height as f32).ceil() + 1.0).min(height as f32) as usize;
            for x in x_min..x_max {
                let dx = x as f32 * inv_width - position[0];
                let dx2 = dx * dx;
                let column = &mut levels[x * height..(x + 1) * height];
                for (y, level) in column.iter_mut().enumerate().take(y_max).skip(y_min) {
                    let dy = y as f32 * inv_height - position[1];
                    let distance2 = dx2 + dy * dy;
                    if distance2 < radius2 {
                        *level += wave.sample(distance2.sqrt());
                    }
                }
            }
        }
        // Fix the zero-frequency mode: pressure bowls redistribute volume
        // inside the finite patch, without nonlinear height compression.
        if !self.world_enabled {
            let mean =
                levels.iter().map(|height| *height as f64).sum::<f64>() / levels.len() as f64;
            for level in levels {
                *level -= mean as f32;
            }
        }
        self.resolve_materials();
        self.resolve_physical_impacts();
    }

    fn resolve_physical_impacts(&mut self) {
        if !self
            .wave_centers
            .iter()
            .any(|center| center.physical.is_some())
        {
            return;
        }
        let width = self.width() as usize;
        let height = self.height() as usize;
        let lifetime = SETTINGS.attenuation_time.as_secs_f32();
        let mut requests = Vec::new();
        let mut missing = Vec::new();
        for (index, center) in self.wave_centers.iter().enumerate() {
            let Some(impact) = center.physical else {
                continue;
            };
            let age = self
                .time
                .saturating_duration_since(center.time)
                .as_secs_f32();
            let key = (age.to_bits(), lifetime.to_bits(), width, height);
            if center
                .physical_cache
                .as_ref()
                .is_none_or(|cache| cache.key != key)
            {
                requests.push((impact, age));
                missing.push((index, key));
            }
        }
        // Reuse only the radial transform, never the spatial field. New contacts,
        // expired/evicted sources and world recentering still rebuild the field.
        // At a fixed physical instant the transform depends only on immutable
        // impact parameters, age, lifetime and the grid's spectral cutoff.
        let profiles = RadialWave::for_pressure_impacts(&requests, lifetime, width, height);
        for ((index, key), wave) in missing.into_iter().zip(profiles) {
            let value = PhysicalWaveCache { key, wave };
            match &mut self.wave_centers[index].physical_cache {
                Some(cache) => **cache = value,
                slot @ None => *slot = Some(Box::new(value)),
            }
        }
        let levels = self
            .levels
            .as_slice_mut()
            .expect("water field is contiguous");
        let mut dy_squared = vec![0.0; height];
        for center in &self.wave_centers {
            let Some(wave) = center
                .physical_cache
                .as_ref()
                .and_then(|cache| cache.wave.as_ref())
            else {
                continue;
            };
            for (y, distance2) in dy_squared.iter_mut().enumerate() {
                let dy = y as f32 / height as f32 - center.y;
                *distance2 = dy * dy;
            }
            for x in 0..width {
                let dx = x as f32 / width as f32 - center.x;
                let dx2 = dx * dx;
                let offset = x * height;
                let column = &mut levels[offset..offset + height];
                let depths = &mut self.depths[offset..offset + height];
                for (y, (level, depth)) in column.iter_mut().zip(depths).enumerate() {
                    let eta = wave.sample((dx2 + dy_squared[y]).sqrt());
                    *level += eta;
                    // Same geometric displacement in height and depth preserves
                    // bed elevation. The conservative hydro state and its fluxes
                    // are untouched by this linear small-wave closure.
                    *depth += eta;
                }
            }
        }
    }

    /// Interpolating geometry shared by native contact and GPU ray optics.
    /// Coordinates are normalized to the resident patch; derivatives are per
    /// metre. Clamped ghost samples match the Metal/WGSL tensor Catmull-Rom.
    pub(crate) fn sample_cubic_surface(
        &self,
        point: [f32; 2],
    ) -> [f32; 3] {
        let width = self.width() as usize;
        let height = self.height() as usize;
        if width == 0
            || height == 0
            || !point
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        {
            return [0.0; 3];
        }
        let p = [point[0] * width as f32, point[1] * height as f32];
        let cell = p.map(|value| value.floor() as isize);
        let t = [p[0] - cell[0] as f32, p[1] - cell[1] as f32];
        let coefficients = |h: [f32; 4]| {
            [
                0.5 * (-h[0] + 3.0 * h[1] - 3.0 * h[2] + h[3]),
                h[0] - 2.5 * h[1] + 2.0 * h[2] - 0.5 * h[3],
                0.5 * (h[2] - h[0]),
                h[1],
            ]
        };
        let mut values = [0.0; 4];
        let mut dx = [0.0; 4];
        for row in 0..4 {
            let y = (cell[1] + row as isize - 1).clamp(0, height as isize - 1) as usize;
            let samples = std::array::from_fn(|column| {
                let x = (cell[0] + column as isize - 1).clamp(0, width as isize - 1) as usize;
                self.levels[[x, y]]
            });
            let a = coefficients(samples);
            values[row] = ((a[0] * t[0] + a[1]) * t[0] + a[2]) * t[0] + a[3];
            dx[row] = (3.0 * a[0] * t[0] + 2.0 * a[1]) * t[0] + a[2];
        }
        let a = coefficients(values);
        let b = coefficients(dx);
        [
            ((a[0] * t[1] + a[1]) * t[1] + a[2]) * t[1] + a[3],
            (((b[0] * t[1] + b[1]) * t[1] + b[2]) * t[1] + b[3]) * width as f32 / PATCH_LENGTH,
            ((3.0 * a[0] * t[1] + 2.0 * a[1]) * t[1] + a[2]) * height as f32 / PATCH_LENGTH,
        ]
    }

    /// Native ballistic contact. Inputs use metres, m/s, and physical time.
    /// The sphere's relative vertical approach speed is projected on the local
    /// surface normal. No duplicate terminal wave, pigment, or empirical kick.
    #[allow(dead_code)]
    pub fn add_ballistic_drop(
        &mut self,
        point: [f32; 2],
        radius: f32,
        vertical_speed: f32,
        contact_age: Duration,
    ) {
        let width = self.width() as usize;
        let height = self.height() as usize;
        if width < 2
            || height < 2
            || !point
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        {
            return;
        }
        let [surface, sx, sy] = self.sample_cubic_surface(point);
        let normal_speed = vertical_speed / (1.0 + sx * sx + sy * sy).sqrt();
        // The bed remains bilinear height minus bilinear depth. Only the tiny
        // reconstruction offset changes the wet column under the cubic surface.
        let bilinear_height = sample_surface(
            self.levels.as_slice().unwrap(),
            width,
            height,
            point[0],
            point[1],
        );
        let depth = (sample_surface(&self.depths, width, height, point[0], point[1]) + surface
            - bilinear_height)
            .max(0.001);
        let Some(impact) = PressureImpact::new(radius, normal_speed, depth) else {
            return;
        };
        if self.wave_centers.len() >= MAX_WAVES {
            self.wave_centers
                .drain(..self.wave_centers.len() - MAX_WAVES + 1);
        }
        let contact_age = contact_age.min(self.elapsed);
        // Contact occurs inside the accepted frame. Propagate that fractional
        // age spatially as well as spectrally; otherwise faster currents make
        // the wave source depend on the display/physics frame partition.
        let velocity = |point: [f32; 2]| {
            let x = (point[0] * width as f32).clamp(0.0, (width - 1) as f32);
            let y = (point[1] * height as f32).clamp(0.0, (height - 1) as f32);
            let x0 = x as usize;
            let y0 = y as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let tx = x - x0 as f32;
            let ty = y - y0 as f32;
            let a = self.native_transport_velocity(x0, y0);
            let b = self.native_transport_velocity(x1, y0);
            let c = self.native_transport_velocity(x0, y1);
            let d = self.native_transport_velocity(x1, y1);
            std::array::from_fn::<_, 2, _>(|axis| {
                (a[axis] * (1.0 - tx) + b[axis] * tx) * (1.0 - ty)
                    + (c[axis] * (1.0 - tx) + d[axis] * tx) * ty
            })
        };
        let dt = contact_age.as_secs_f32() / PATCH_LENGTH;
        let a = velocity(point);
        let b = velocity(std::array::from_fn(|axis| point[axis] + a[axis] * dt * 0.5));
        let position: [f32; 2] = std::array::from_fn(|axis| point[axis] + b[axis] * dt);
        let mut source = WaveCenter::new(position[0], position[1], 0.0, self.time - contact_age);
        source.physical = Some(impact);
        self.wave_centers.push(source);
    }

    /// Add a deterministic impact at normalized coordinates on the water plane.
    pub fn add_drop(
        &mut self,
        x: f32,
        y: f32,
        strength: f32,
    ) {
        self.add_drop_sized(x, y, strength, 1.0);
    }

    pub fn add_drop_sized(
        &mut self,
        x: f32,
        y: f32,
        strength: f32,
        size: f32,
    ) {
        self.add_drop_impulse(x, y, strength, size, true);
    }

    /// Native volumetric pigment has its own conserved mass field. Deposit the
    /// identical hydraulic impulse and wake without creating a second dye mass.
    #[allow(dead_code)] // Called by the separately built native-core crate.
    pub fn add_drop_sized_uncolored(
        &mut self,
        x: f32,
        y: f32,
        strength: f32,
        size: f32,
    ) {
        self.add_drop_impulse(x, y, strength, size, false);
    }

    fn add_drop_impulse(
        &mut self,
        x: f32,
        y: f32,
        strength: f32,
        size: f32,
        colored: bool,
    ) {
        if !x.is_finite() || !y.is_finite() || !strength.is_finite() || strength <= 0.0 {
            return;
        }
        if !size.is_finite() || size <= 0.0 {
            return;
        }
        let size = size.clamp(0.25, 3.0);
        let point = [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)];
        if self.drain_requested {
            if let Some(flow) = self.drain.as_mut() {
                flow.disturb_sized(point, strength.min(1.0), size);
            }
            if let Some(ripples) = self.drain_ripples.as_mut() {
                ripples.deposit(point, strength.min(1.0), size);
            }
        } else {
            let blend = self.rapids_blend * self.rapids_blend * (3.0 - 2.0 * self.rapids_blend);
            let mut ambient_velocity = if let Some(world) = &self.world_flow {
                let current = world.velocity(point);
                let through = self.through_current();
                std::array::from_fn(|axis| (current[axis] + through[axis]) * (1.0 - blend))
            } else if blend < 1.0 {
                VortexFrame::new(self.coupled_elapsed.as_secs_f64(), DYE_WIDTH, DYE_HEIGHT)
                    .periodic_velocity(point)
                    .map(|v| v * (1.0 - blend))
            } else {
                [0.0; 2]
            };
            if let Some(flow) = self.rapids.as_ref().filter(|_| blend > 0.0) {
                let current = flow.depth_averaged_velocity(point);
                for axis in 0..2 {
                    ambient_velocity[axis] += current[axis] * blend;
                }
            }
            let classic_radius = 0.035 + 0.045 * strength.min(1.0).sqrt();
            let rapids_radius = 0.055 + 0.035 * strength.min(1.0).sqrt();
            let impulse_radius = (classic_radius + (rapids_radius - classic_radius) * blend) * size;
            // The finite wake is aligned with the pre-impact column current. Its
            // support follows the pressure footprint, not the smaller pigment spot.
            self.stirring.deposit(
                point,
                impulse_radius,
                strength.min(1.0),
                ambient_velocity,
                self.transport_boundary(),
            );
            if let Some(flow) = self.rapids.as_mut().filter(|_| self.rapids_requested) {
                if size == 1.0 {
                    flow.disturb(point, strength.min(1.0));
                } else {
                    flow.disturb_sized(point, strength.min(1.0), size);
                }
            }
        }
        if colored {
            // Independent pigment species, mixed as concentrations rather than RGB
            // screen colors. A larger source carries proportionally more dye volume.
            let phase = self.drop_sequence.wrapping_mul(37) % 101;
            self.drop_sequence = self.drop_sequence.wrapping_add(1);
            let species = match phase % 6 {
                0 => [0.0, 1.0, 0.05],
                1 => [0.7, 0.3, 0.0],
                2 => [1.0, 0.0, 0.05],
                3 => [0.45, 0.0, 0.55],
                4 => [0.0, 0.1, 0.9],
                _ => [0.15, 0.75, 0.1],
            };
            let radius = 0.025 * size * (0.65 + strength.min(1.0).sqrt());
            let amount = 0.006 * (strength.min(1.0) / 0.2) * size.powi(3);
            self.dissolved.deposit(point, radius, amount, species);
        }
        if self.wave_centers.len() >= MAX_WAVES {
            self.wave_centers
                .drain(..self.wave_centers.len() - MAX_WAVES + 1);
        }
        let mut center = WaveCenter::new(
            x.clamp(0.0, 1.0),
            y.clamp(0.0, 1.0),
            strength.min(1.0),
            self.time,
        );
        center.size = size;
        self.wave_centers.push(center);
    }

    #[cfg(test)]
    pub fn add_random_drop(&mut self) {
        if self.width() == 0 || self.height() == 0 {
            return;
        }
        self.add_drop(
            rand::random(),
            rand::random(),
            0.08 + rand::random::<f32>().powi(2) * 0.4,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_surface_preserves_metric_planes_saddles_and_resolved_node_amplitudes() {
        let mut water = Water::new(32, 24);
        for curved in [false, true] {
            for ix in 0..32 {
                for iy in 0..24 {
                    let x = ix as f32 * PATCH_LENGTH / 32.0;
                    let y = iy as f32 * PATCH_LENGTH / 24.0;
                    let height = 0.3
                        + 0.04 * x
                        + 0.07 * y
                        + if curved {
                            0.02 * x * y + 0.03 * x * x - 0.04 * y * y
                        } else {
                            0.0
                        };
                    water.levels[[ix, iy]] = height;
                }
            }
            for point in [[0.25, 0.5], [0.37, 0.61], [0.72, 0.27]] {
                let [x, y] = point.map(|value| value * PATCH_LENGTH);
                let expected = [
                    0.3 + 0.04 * x
                        + 0.07 * y
                        + if curved {
                            0.02 * x * y + 0.03 * x * x - 0.04 * y * y
                        } else {
                            0.0
                        },
                    0.04 + if curved { 0.02 * y + 0.06 * x } else { 0.0 },
                    0.07 + if curved { 0.02 * x - 0.08 * y } else { 0.0 },
                ];
                for (actual, expected) in
                    water.sample_cubic_surface(point).into_iter().zip(expected)
                {
                    assert!(
                        (actual - expected).abs() < 3e-6,
                        "{point:?}: {actual} vs {expected}"
                    );
                }
            }
        }
        for x in 0..32 {
            for y in 0..24 {
                water.levels[[x, y]] = 0.025 * (TAU * x as f32 / 3.0).cos();
            }
        }
        for x in 1..31 {
            let actual = water.sample_cubic_surface([x as f32 / 32.0, 0.5])[0];
            assert_eq!(
                actual,
                water.levels[[x, 12]],
                "resolved wave node lost amplitude"
            );
        }
    }

    #[test]
    fn cubic_surface_uses_finite_clamped_ghost_samples_at_patch_edges() {
        let mut water = Water::new(8, 6);
        for x in 0..8 {
            for y in 0..6 {
                water.levels[[x, y]] = x as f32 * 0.25;
            }
        }
        assert_eq!(water.sample_cubic_surface([0.0, 0.5]), [0.0, 0.5, 0.0]);
        assert_eq!(water.sample_cubic_surface([1.0, 0.5]), [1.75, 0.0, 0.0]);
        // The last resident cell has repeated right-hand ghost samples, not a
        // coordinate clamped prematurely to the final lattice node.
        let edge = water.sample_cubic_surface([7.5 / 8.0, 0.5]);
        assert!((edge[0] - 1.765625).abs() < 1e-7);
        assert!((edge[1] + 0.125).abs() < 1e-7);
        for height in [0.0, -0.3, 0.2] {
            water.levels.fill(height);
            for point in [[0.0, 0.0], [1.0, 1.0], [0.37, 0.91]] {
                let surface = water.sample_cubic_surface(point);
                assert!(surface.iter().all(|value| value.is_finite()));
                assert!((surface[0] - height).abs() < 1e-7);
                assert!(surface[1].abs() < 1e-6 && surface[2].abs() < 1e-6);
            }
        }
        assert_eq!(Water::new(0, 0).sample_cubic_surface([0.5, 0.5]), [0.0; 3]);
    }

    #[test]
    fn ballistic_pressure_uses_the_same_cubic_normal_and_wet_column() {
        let mut water = Water::new(32, 24);
        water.levels.fill(0.0);
        water.depths.fill(0.7);
        for y in 0..24 {
            water.levels[[12, y]] = 0.08;
            water.levels[[14, y]] = 0.02;
        }
        let point = [12.25 / 32.0, 0.5];
        // Independent polynomial of the samples [0,.08,0,.02].
        let surface = 0.13 * 0.25_f32.powi(3) - 0.21 * 0.25_f32.powi(2) + 0.08;
        let sx = (3.0 * 0.13 * 0.25_f32.powi(2) - 2.0 * 0.21 * 0.25) * 16.0;
        let expected =
            PressureImpact::new(0.00825, 2.9 / (1.0 + sx * sx).sqrt(), 0.7 + surface - 0.06)
                .unwrap();
        water.add_ballistic_drop(point, 0.00825, 2.9, Duration::ZERO);
        let actual = water.wave_centers[0].physical.unwrap();
        for age in [0.01, 0.1, 0.4] {
            let actual = RadialWave::for_pressure_impact(actual, age, 10.0, 32, 24).unwrap();
            let expected = RadialWave::for_pressure_impact(expected, age, 10.0, 32, 24).unwrap();
            for radius in [0.0, 0.03, 0.2] {
                assert!((actual.sample(radius) - expected.sample(radius)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn native_column_current_uses_rapids_momentum_and_keeps_classic_current() {
        let mut water = Water::new(48, 32);
        water.configure_world(37, true);
        water.update(Duration::ZERO);
        for (x, y) in [(8, 12), (28, 21)] {
            assert_eq!(
                water.native_transport_velocity(x, y),
                water.surface_velocity[x * 32 + y]
            );
        }
        let enabled = water.native_transport_velocity(8, 12);
        water.set_current(false);
        water.update(Duration::ZERO);
        let disabled = water.native_transport_velocity(8, 12);
        for axis in 0..2 {
            assert!((enabled[axis] - disabled[axis] - CURRENT[axis]).abs() < 1e-6);
        }
        water.set_rapids(true);
        water.update(Duration::from_millis(464));
        assert_eq!(water.rapids_blend, 1.0);
        let flow = water.rapids.as_ref().unwrap();
        let mut maximum_difference = 0.0_f32;
        for x in 0..48 {
            let point = [x as f32 / 48.0, 0.5];
            let expected = flow.depth_averaged_velocity(point);
            let surface = water.surface_velocity[x * 32 + 16];
            for (actual, expected) in water
                .native_transport_velocity(x, 16)
                .into_iter()
                .zip(expected)
            {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "volume must follow momentum, not surface foam"
                );
            }
            maximum_difference = maximum_difference.max((expected[0] - surface[0]).abs());
        }
        assert!(
            maximum_difference > 0.001,
            "fixture must distinguish the two flow fields"
        );
    }

    #[test]
    fn uncolored_drop_preserves_water_impulse_without_duplicate_pigment() {
        for mode in 0..3 {
            let mut colored = Water::new(48, 32);
            let mut uncolored = Water::new(48, 32);
            for water in [&mut colored, &mut uncolored] {
                water.set_drain(mode == 1);
                water.set_rapids(mode == 2);
                water.update(Duration::from_millis(32));
            }
            colored.add_drop_sized(0.45, 0.55, 0.2, 1.5);
            uncolored.add_drop_sized_uncolored(0.45, 0.55, 0.2, 1.5);
            for water in [&mut colored, &mut uncolored] {
                water.update(Duration::from_millis(64));
            }
            assert_eq!(colored.levels, uncolored.levels);
            assert_eq!(colored.surface_velocity, uncolored.surface_velocity);
            assert_eq!(colored.foam, uncolored.foam);
            assert!(colored.has_dye());
            assert!(!uncolored.has_dye());
            assert!(uncolored.dye.iter().all(|value| *value == [0.0; 3]));
        }
    }

    #[test]
    fn ballistic_profile_cache_reuses_only_unchanged_time_and_grid() {
        let mut water = Water::new(48, 32);
        water.set_current(false);
        water.update(Duration::from_millis(480));
        water.add_ballistic_drop([0.5, 0.5], 0.00825, 2.9, Duration::from_millis(6));
        water.update(Duration::ZERO);
        let once = water.levels.clone();
        let built = crate::waves::profile_build_count();
        water.update(Duration::ZERO);
        assert_eq!(water.levels, once);
        assert_eq!(
            crate::waves::profile_build_count(),
            built,
            "same-time display/contact refresh must reuse the radial transform"
        );

        water.add_ballistic_drop([0.6, 0.4], 0.012, 2.9, Duration::from_millis(4));
        water.update(Duration::ZERO);
        assert_eq!(
            crate::waves::profile_build_count(),
            built + 1,
            "a new contact must build only its own profile"
        );
        let before_step = crate::waves::profile_build_count();
        water.update(Duration::from_millis(16));
        assert_eq!(
            crate::waves::profile_build_count(),
            before_step + 2,
            "advancing physical time must rebuild both profiles"
        );

        // Test resolution invalidation without also changing physical time.
        water.levels = Array2::zeros((64, 48));
        water.depths.resize(64 * 48, DEPTH);
        let before_resize = crate::waves::profile_build_count();
        water.resolve_physical_impacts();
        assert_eq!(
            crate::waves::profile_build_count(),
            before_resize + 2,
            "a changed Nyquist filter requires rebuilding both profiles"
        );
    }

    #[test]
    fn ballistic_surface_closure_is_visible_in_all_modes_and_preserves_the_bed() {
        for mode in 0..3 {
            let mut water = Water::new(192, 128);
            water.configure_world(37, true);
            water.set_current(false);
            water.set_drain(mode == 1);
            water.set_rapids(mode == 2);
            water.update(Duration::from_millis(480));
            let before = water.levels.clone();
            let depths = water.depths.clone();
            let velocity = water.surface_velocity.clone();
            water.add_ballistic_drop([0.5, 0.5], 0.00825, 2.9, Duration::from_millis(6));
            water.update(Duration::ZERO);
            let peak = water
                .levels
                .iter()
                .zip(before.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(peak > 0.0001, "mode={mode}, first-contact peak={peak}");
            for (i, depth) in depths.iter().enumerate() {
                let before_bed = before.as_slice().unwrap()[i] - depth;
                let after_bed = water.levels.as_slice().unwrap()[i] - water.depths[i];
                assert!((before_bed - after_bed).abs() < 1e-6);
                assert!(water.depths[i] > 0.0);
            }
            assert_eq!(velocity, water.surface_velocity);
            assert!(!water.has_dye());
            assert_eq!(water.wave_centers.len(), 1);
            assert!(water.wave_centers[0].physical.is_some());
            let once = water.levels.clone();
            water.update(Duration::ZERO);
            assert_eq!(
                water.levels, once,
                "display refresh must not reinject or accumulate displacement"
            );
        }
    }

    #[test]
    fn ballistic_surface_closure_preserves_its_world_position_on_rebase() {
        let mut water = Water::new(192, 128);
        water.configure_world(37, true);
        water.set_current(false);
        water.update(Duration::from_millis(480));
        water.add_ballistic_drop([0.5, 0.5], 0.00825, 2.9, Duration::from_millis(40));
        water.update(Duration::ZERO);
        let before = water.levels.clone();
        let old_origin = water.world_origin();
        let old_position = [water.wave_centers[0].x, water.wave_centers[0].y];
        water.set_world_center([0.25, -0.25]);
        water.update(Duration::ZERO);
        for axis in 0..2 {
            let coordinate = if axis == 0 {
                water.wave_centers[0].x
            } else {
                water.wave_centers[0].y
            };
            assert!(
                (f64::from(coordinate) * 2.0 + water.world_origin()[axis]
                    - (f64::from(old_position[axis]) * 2.0 + old_origin[axis]))
                    .abs()
                    < 1e-7
            );
        }
        for x in 60..150 {
            for y in 25..90 {
                assert!((before[[x, y]] - water.levels[[x - 24, y + 16]]).abs() < 0.00001);
            }
        }
    }

    #[test]
    fn ballistic_contact_substep_age_advects_the_source_in_metric_units() {
        let mut water = Water::new(192, 128);
        water.update(Duration::from_millis(480));
        water.surface_velocity.fill([0.4, -0.2]);
        water.add_ballistic_drop([0.5, 0.5], 0.00825, 2.9, Duration::from_millis(40));
        let source = &water.wave_centers[0];
        assert!((source.x - (0.5 + 0.4 * 0.04 / PATCH_LENGTH)).abs() < 1e-7);
        assert!((source.y - (0.5 - 0.2 * 0.04 / PATCH_LENGTH)).abs() < 1e-7);
        assert_eq!(
            water.time.duration_since(source.time),
            Duration::from_millis(40)
        );
    }

    #[test]
    fn recentered_edge_vortices_do_not_sample_a_clamped_patch() {
        let mut water = Water::new(96, 64);
        water.configure_world(7, true);
        water.set_current(false);
        water.update(Duration::from_millis(400));
        let before = water.levels.clone();
        water.set_world_center([0.25, -0.25]);
        water.update(Duration::ZERO);
        let mut maximum = 0.0_f32;
        for x in 12..96 {
            for y in 0..56 {
                maximum = maximum.max((before[[x, y]] - water.levels[[x - 12, y + 8]]).abs());
            }
        }
        assert!(
            maximum < 0.00001,
            "recenter changed overlapping edge by {maximum} m"
        );
    }

    #[test]
    fn recenter_does_not_pump_retained_world_surface_or_resample_pigment() {
        let mut water = Water::new(96, 64);
        water.configure_world(19, true);
        water.add_drop_sized(0.45, 0.55, 0.2, 1.5);
        water.update(Duration::from_millis(160));
        let before = water.levels.clone();
        let dye = water.dye.clone();
        water.set_world_center([0.25, -0.25]);
        water.update(Duration::ZERO);
        let mut maximum = 0.0_f32;
        for x in 30..75 {
            for y in 15..45 {
                maximum = maximum.max((before[[x, y]] - water.levels[[x - 12, y + 8]]).abs());
                for (before, after) in dye[x * 64 + y].iter().zip(water.dye[(x - 12) * 64 + y + 8])
                {
                    assert!((before - after).abs() < 2e-5);
                }
            }
        }
        assert!(
            maximum < 0.00001,
            "window translation changed retained surface by {maximum} m"
        );
    }

    #[test]
    fn world_transport_is_cadence_independent_and_zero_size_safe() {
        let mut whole = Water::new(24, 16);
        let mut split = Water::new(24, 16);
        for water in [&mut whole, &mut split] {
            water.configure_world(7, true);
            water.add_drop_sized(0.42, 0.54, 0.3, 1.5);
        }
        whole.update(Duration::from_millis(240));
        for _ in 0..30 {
            split.update(Duration::from_millis(8));
        }
        assert_eq!(whole.current_displacement, split.current_displacement);
        assert_eq!(whole.dye, split.dye);
        assert_eq!(whole.levels, split.levels);
        let before = whole.levels.clone();
        whole.resize(0, 0);
        whole.resize(24, 16);
        assert_eq!(whole.levels, before);
        whole.set_world_center([f64::NAN, 0.0]);
        assert_eq!(whole.world_origin(), [-1.0; 2]);
        whole.set_world_center([10_000.0, -10_000.0]);
        whole.update(Duration::ZERO);
        assert!(whole.levels.iter().all(|v| v.is_finite()));
        assert!(whole.dye.iter().all(|v| *v == [0.0; 3]));
    }

    #[test]
    fn streamed_recenter_preserves_pigment_and_impacts_in_world_coordinates() {
        let mut water = Water::new(96, 64);
        water.configure_world(7, true);
        water.set_current(false);
        water.add_drop_sized(0.45, 0.55, 0.2, 1.5);
        water.update(Duration::from_millis(80));
        let pigment = water.dissolved.sample([0.45, 0.55]);
        let impact = [water.wave_centers[0].x, water.wave_centers[0].y];
        let before = water.world_origin();
        water.set_world_center([0.25, -0.25]);
        let after = water.world_origin();
        assert_eq!(after, [-0.75, -1.25]);
        let offset = [
            (after[0] - before[0]) as f32 / PATCH_LENGTH,
            (after[1] - before[1]) as f32 / PATCH_LENGTH,
        ];
        let translated = water.dissolved.sample([0.45 - offset[0], 0.55 - offset[1]]);
        for channel in 0..3 {
            assert!((pigment[channel] - translated[channel]).abs() < 0.00001);
        }
        assert_eq!(water.wave_centers[0].x, impact[0] - offset[0]);
        assert_eq!(water.wave_centers[0].y, impact[1] - offset[1]);
    }

    #[test]
    fn world_current_switch_preserves_phase_and_moves_both_pigment_and_impacts() {
        let mut water = Water::new(96, 64);
        water.configure_world(7, true);
        water.add_drop_sized(0.43, 0.57, 0.2, 1.0);
        let position = [water.wave_centers[0].x, water.wave_centers[0].y];
        water.update(Duration::from_millis(160));
        assert_ne!([water.wave_centers[0].x, water.wave_centers[0].y], position);
        assert!(water.current_displacement[0] > 0.0);
        let levels = water.levels.clone();
        let displacement = water.current_displacement;
        water.set_current(false);
        water.update(Duration::ZERO);
        assert_eq!(
            water.levels, levels,
            "a current toggle cannot relocate waves"
        );
        assert_eq!(water.current_displacement, displacement);
        water.update(Duration::from_millis(80));
        assert_eq!(water.current_displacement, displacement);
    }

    #[test]
    fn numerical_modes_suspend_streaming_and_leave_the_hydraulic_state_intact() {
        let mut water = Water::new(32, 24);
        water.configure_world(7, true);
        water.set_drain(true);
        water.update(Duration::from_millis(480));
        let origin = water.world_origin();
        let before = water.drain.as_ref().unwrap().sample([0.45, 0.55]);
        water.set_world_center([50.0, 30.0]);
        assert_eq!(water.world_origin(), origin);
        let after = water.drain.as_ref().unwrap().sample([0.45, 0.55]);
        assert_eq!(before.depth, after.depth);
        assert_eq!(before.velocity, after.velocity);
    }

    #[test]
    fn selected_drain_advances_hydraulics_and_pigment_after_four_milliseconds() {
        let mut water = Water::new(19, 13);
        water.set_drain(true);
        water.add_drop_sized(0.64, 0.43, 0.24, 1.0);
        let before = water.drain.as_ref().unwrap().sample([0.64, 0.43]);
        let pigment = water.dissolved.sample([0.64, 0.43]);
        water.update(Duration::from_millis(4));
        let after = water.drain.as_ref().unwrap().sample([0.64, 0.43]);
        assert_ne!(
            after.depth, before.depth,
            "a Drain tick must not wait for an unrelated 8 ms transport batch"
        );
        assert_ne!(
            water.dissolved.sample([0.64, 0.43]),
            pigment,
            "Drain hydraulic and scalar stages must advance together"
        );
    }

    #[test]
    fn developed_drain_rain_keeps_optical_wave_detail_separate_from_depth() {
        let mut water = Water::new(96, 64);
        water.set_drain(true);
        water.update(Duration::from_secs(4));
        water.add_drop_sized(0.64, 0.43, 0.24, 1.0);
        water.update(Duration::from_millis(48));
        let mut peak = 0.0_f32;
        for x in 0..96 {
            for y in 0..64 {
                let index = x * 64 + y;
                let hydraulic = water
                    .drain
                    .as_ref()
                    .unwrap()
                    .sample([x as f32 / 96.0, y as f32 / 64.0]);
                assert_eq!(
                    water.depths[index], hydraulic.depth,
                    "optical waves cannot change hydraulic column depth"
                );
                assert_eq!(water.surface_velocity[index], hydraulic.velocity);
                assert_eq!(water.foam[index], hydraulic.foam);
                peak = peak.max((water.levels[[x, y]] - hydraulic.elevation).abs());
            }
        }
        assert!(
            peak > 0.0002,
            "developed Drain drops lost their small-wave optical response: {peak} m"
        );
        assert!(
            peak <= 0.00301,
            "optical closure must remain small amplitude: {peak} m"
        );
        let before = water.levels.clone();
        let dye = water.dye.clone();
        water.resize(0, 0);
        water.update(Duration::ZERO);
        water.resize(96, 64);
        assert_eq!(
            water.levels, before,
            "resize cannot integrate or diffuse detail"
        );
        assert_eq!(water.dye, dye);
    }

    #[test]
    fn drain_selection_is_exclusive_and_does_not_snap_or_reset() {
        let mut water = Water::new(23, 17);
        water.add_drop_sized(0.4, 0.5, 0.2, 1.0);
        water.update(Duration::from_millis(64));
        let initial = water.levels.clone();
        let pigment = water.dye.clone();
        water.set_drain(true);
        water.update(Duration::ZERO);
        assert_eq!(
            water.levels, initial,
            "mode selection cannot snap the surface"
        );
        assert_eq!(water.dye, pigment, "mode selection cannot reset pigment");
        water.update(Duration::from_millis(8));
        assert!(
            water.has_drain(),
            "selected Drain must begin its visual transition"
        );
        assert!(water.has_numerical_flow());
        water.set_rapids(false);
        water.update(Duration::from_millis(456));
        assert!(water.has_drain(), "disabling Rapids must not disable Drain");
        assert!(!water.has_rapids());
        water.set_rapids(true);
        water.update(Duration::from_millis(464));
        assert!(
            !water.has_drain(),
            "enabling Rapids must retire the Drain display"
        );
        assert!(water.has_rapids());
        water.set_drain(false);
        water.update(Duration::from_millis(8));
        assert!(
            water.has_rapids(),
            "disabling Drain must not disable Rapids"
        );
        water.set_drain(true);
        water.update(Duration::from_millis(464));
        assert!(water.has_drain());
        assert!(!water.has_rapids());
    }

    #[test]
    fn drain_alone_advances_hydraulics_and_pigment_once_per_shared_tick() {
        let mut water = Water::new(19, 13);
        water.set_drain(true);
        water.add_drop_sized(0.38, 0.47, 0.24, 1.3);
        let mut expected_flow = crate::drain::Drain::new();
        let mut expected_dye = Dye::new();
        expected_flow.disturb_sized([0.38, 0.47], 0.24, 1.3);
        expected_dye.deposit(
            [0.38, 0.47],
            0.025 * 1.3 * (0.65 + 0.24_f32.sqrt()),
            0.006 * (0.24 / 0.2) * 1.3_f32.powi(3),
            [0.0, 1.0, 0.05],
        );
        water.update(Duration::from_millis(1003));
        for _ in 0..125 {
            expected_flow.advance(TRANSPORT_TICK, &mut expected_dye);
        }
        for x in 0..water.width() as usize {
            for y in 0..water.height() as usize {
                let point = [
                    x as f32 / water.width() as f32,
                    y as f32 / water.height() as f32,
                ];
                let index = x * water.height() as usize + y;
                let expected = expected_flow.sample(point);
                assert_eq!(
                    water.levels[[x, y]],
                    expected.elevation,
                    "Drain must own surface geometry"
                );
                assert_eq!(water.depths[index], expected.depth);
                assert_eq!(water.foam[index], expected.foam);
                assert_eq!(water.surface_velocity[index], expected.velocity);
                assert_eq!(
                    water.dye[index],
                    expected_dye.sample(point),
                    "Drain dye was skipped, reset or advanced twice"
                );
            }
        }
    }

    #[test]
    fn drain_state_freezes_when_inactive_and_resumes_without_reinitializing() {
        let sample_state = |flow: &Drain| {
            [[0.5, 0.5], [0.57, 0.5], [0.3, 0.7], [0.83, 0.22]].map(|point| {
                let sample = flow.sample(point);
                (sample.elevation, sample.depth, sample.foam, sample.velocity)
            })
        };
        let mut water = Water::new(23, 17);
        water.set_drain(true);
        water.add_drop_sized(0.3, 0.7, 0.24, 1.3);
        water.update(Duration::from_millis(600));
        let evolved = sample_state(water.drain.as_ref().unwrap());
        assert_ne!(
            evolved,
            sample_state(&Drain::new()),
            "test must retain an evolved hydraulic state"
        );
        water.set_drain(false);
        water.update(Duration::from_millis(600));
        assert_eq!(
            sample_state(water.drain.as_ref().unwrap()),
            evolved,
            "inactive Drain cannot evolve or drain unseen pigment"
        );
        let pigment = water.dye.clone();
        water.set_drain(true);
        water.update(Duration::ZERO);
        assert_eq!(
            sample_state(water.drain.as_ref().unwrap()),
            evolved,
            "re-entry cannot reset the initialized Drain"
        );
        assert_eq!(water.dye, pigment);
        water.update(TRANSPORT_TICK);
        assert_ne!(
            sample_state(water.drain.as_ref().unwrap()),
            evolved,
            "selected Drain must resume evolution"
        );
    }

    #[test]
    fn drain_partition_pause_and_empty_resize_preserve_the_same_state() {
        let make = || {
            let mut water = Water::new(23, 17);
            water.set_drain(true);
            water.add_drop_sized(0.38, 0.47, 0.24, 1.3);
            water.add_drop_sized(0.65, 0.62, 0.31, 0.8);
            water
        };
        let mut whole = make();
        let mut split = make();
        for (milliseconds, mode) in [(117, 2), (469, 1), (537, 0), (1003, 2)] {
            for water in [&mut whole, &mut split] {
                match mode {
                    2 => water.set_drain(true),
                    1 => water.set_rapids(true),
                    _ => {
                        water.set_drain(false);
                        water.set_rapids(false);
                    },
                }
                water.add_drop_sized(0.2, 0.72, 0.2, 0.9);
            }
            whole.update(Duration::from_millis(milliseconds));
            for _ in 0..milliseconds / 7 {
                split.update(Duration::from_millis(7));
                split.resize(0, 0);
                split.update(Duration::ZERO);
                split.resize(23, 17);
            }
            split.update(Duration::from_millis(milliseconds % 7));
            assert_eq!(
                whole.dye, split.dye,
                "Drain coupling must ignore display cadence and empty redraws"
            );
            assert_eq!(whole.levels, split.levels);
            assert_eq!(whole.depths, split.depths);
            assert_eq!(whole.foam, split.foam);
            assert_eq!(whole.surface_velocity, split.surface_velocity);
            let levels = split.levels.clone();
            let pigment = split.dye.clone();
            let elapsed = split.elapsed();
            split.update(Duration::ZERO);
            assert_eq!(split.levels, levels);
            assert_eq!(split.dye, pigment);
            assert_eq!(split.elapsed(), elapsed);
        }
        assert!(
            whole.has_drain(),
            "the final requested mode must actually be Drain"
        );
    }

    #[test]
    fn bleaching_setting_preserves_state_until_physical_time_advances() {
        let make = || {
            let mut water = Water::new(23, 17);
            water.add_drop_sized(0.4, 0.5, 0.2, 1.2);
            water.update(Duration::from_millis(128));
            water
        };
        let mut fading = make();
        let mut conserved = make();
        let before = fading.dye.clone();
        let elapsed = fading.elapsed();
        fading.set_bleach(4.0);
        conserved.set_bleach(0.0);
        assert_eq!(fading.dye, before);
        fading.update(Duration::ZERO);
        assert_eq!(
            fading.dye, before,
            "changing bleaching must not reset or evolve pigment"
        );
        assert_eq!(fading.elapsed(), elapsed);
        fading.update(Duration::from_secs(1));
        conserved.update(Duration::from_secs(1));
        let pigment = |water: &Water| water.dye.iter().flatten().map(|c| *c as f64).sum::<f64>();
        let ratio = pigment(&fading) / pigment(&conserved);
        assert!(
            (0.70..0.72).contains(&ratio),
            "4x bleaching should retain 2^(-1/2) after one second, got {ratio}"
        );
        assert_eq!(
            fading.levels, conserved.levels,
            "bleaching cannot change wave physics"
        );
        assert_eq!(fading.surface_velocity, conserved.surface_velocity);
    }

    #[test]
    fn coupled_pigment_and_water_ignore_display_frame_partition() {
        for rapids in [false, true] {
            let make = || {
                let mut water = Water::new(31, 19);
                water.set_rapids(rapids);
                water.add_drop_sized(0.38, 0.47, 0.24, 1.3);
                water.add_drop_sized(0.65, 0.62, 0.31, 0.8);
                water
            };
            let mut single = make();
            let mut partitioned = make();
            single.update(Duration::from_millis(1003));
            for _ in 0..100 {
                partitioned.update(Duration::from_millis(10));
            }
            partitioned.update(Duration::from_millis(3));
            let difference = single
                .dye
                .iter()
                .zip(&partitioned.dye)
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0_f32, f32::max);
            assert_eq!(
                difference, 0.0,
                "coupled dye depends on frame partition, rapids={rapids}"
            );
            assert_eq!(single.levels, partitioned.levels);
            assert_eq!(single.depths, partitioned.depths);
            assert_eq!(single.foam, partitioned.foam);
            assert_eq!(single.surface_velocity, partitioned.surface_velocity);
        }
    }

    #[test]
    fn coupled_mode_changes_keep_fractional_ticks_and_resize_has_no_dynamics() {
        let mut single = Water::new(23, 17);
        let mut partitioned = Water::new(23, 17);
        for (duration_ms, enabled) in [(117, true), (469, false), (537, true)] {
            for water in [&mut single, &mut partitioned] {
                water.set_rapids(enabled);
                water.add_drop_sized(0.12, 0.98, 0.24, 1.3);
            }
            single.update(Duration::from_millis(duration_ms));
            for _ in 0..duration_ms / 7 {
                partitioned.update(Duration::from_millis(7));
                partitioned.resize(0, 0);
                partitioned.update(Duration::ZERO);
                partitioned.resize(23, 17);
            }
            partitioned.update(Duration::from_millis(duration_ms % 7));
            assert!(
                single.dye == partitioned.dye,
                "dye changed across a zero-time resize or partition"
            );
            assert_eq!(single.levels, partitioned.levels);
            assert_eq!(single.foam, partitioned.foam);
        }
    }

    #[test]
    fn heavy_colored_rain_survives_bursts_mode_changes_and_empty_resize() {
        let mut water = Water::new(31, 19);
        water.set_rapids(true);
        for step in 0..120_u32 {
            for drop in 0..if step % 15 == 0 { 15 } else { 3 } {
                let index = step * 17 + drop;
                water.add_drop_sized(
                    ((index * 37) % 997) as f32 / 997.0,
                    ((index * 71) % 991) as f32 / 991.0,
                    0.36,
                    3.0,
                );
            }
            match step {
                30 => water.resize(1, 1),
                31 => water.resize(31, 19),
                60 => water.set_rapids(false),
                80 => water.resize(0, 0),
                81 => water.resize(43, 23),
                90 => water.set_rapids(true),
                _ => {},
            }
            water.update(Duration::from_millis(50));
            assert!(water.levels.iter().all(|v| v.is_finite() && v.abs() < 1.0));
            assert!(water.depths.iter().all(|v| v.is_finite() && *v > 0.0));
            assert!(
                water
                    .foam
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
            assert!(
                water
                    .dye
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && *v >= 0.0)
            );
            assert_eq!(water.dye.len(), water.levels.len());
            assert!(water.wave_centers.len() <= MAX_WAVES);
        }
    }

    #[test]
    fn colored_drop_footprints_persist_independently_through_resize() {
        for rapids in [false, true] {
            let mut water = Water::new(31, 19);
            water.set_rapids(rapids);
            water.add_drop_sized(0.4, 0.5, 0.2, 0.5);
            water.add_drop_sized(0.6, 0.5, 0.2, 2.0);
            assert_eq!(water.wave_centers[0].size, 0.5);
            assert_eq!(water.wave_centers[1].size, 2.0);
            water.update(Duration::from_millis(520));
            assert!(water.has_dye());
            assert!(water.dye.iter().flatten().any(|v| *v > 0.001));
            assert!(
                water
                    .dye
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && *v >= 0.0)
            );
            let before = water.dye.clone();
            water.resize(1, 1);
            water.resize(31, 19);
            assert_eq!(
                water.dye, before,
                "viewport resize must not diffuse pigment"
            );
            water.update(Duration::ZERO);
            assert_eq!(water.dye, before, "pause must stop pigment transport");
            water.update(Duration::from_millis(400));
            assert_ne!(
                water.dye, before,
                "the colored source must move and disperse"
            );
        }
    }

    #[test]
    fn invalid_drop_sizes_do_not_inject_waves_or_pigment() {
        let mut water = Water::new(9, 9);
        for size in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            water.add_drop_sized(0.5, 0.5, 0.2, size);
        }
        assert!(water.wave_centers.is_empty());
        assert!(!water.has_dye());
    }

    #[test]
    fn rapids_evolution_is_independent_of_display_frame_partition() {
        let mut single = Water::new(17, 9);
        let mut partitioned = Water::new(17, 9);
        single.set_rapids(true);
        partitioned.set_rapids(true);
        single.update(Duration::from_secs(1));
        for _ in 0..100 {
            partitioned.update(Duration::from_millis(10));
        }
        assert_eq!(single.levels, partitioned.levels);
        assert_eq!(single.depths, partitioned.depths);
        assert_eq!(single.foam, partitioned.foam);
        assert_eq!(single.surface_velocity, partitioned.surface_velocity);
    }

    #[test]
    fn rapids_is_opt_in_and_keeps_its_state_through_resize() {
        let mut water = Water::new(20, 10);
        assert!(!water.has_rapids());
        let original = water.levels.clone();
        water.set_rapids(true);
        assert_eq!(
            water.levels, original,
            "enabling must not snap the displayed surface"
        );
        water.update(Duration::from_secs(1));
        assert!(water.has_rapids());
        assert_ne!(water.levels, original);
        let evolved = water.levels.clone();
        water.resize(1, 1);
        water.resize(20, 10);
        assert_eq!(water.levels, evolved);
        assert_eq!(water.foam.len(), water.levels.len());
        assert!(
            water
                .depths
                .iter()
                .all(|depth| *depth > 0.0 && depth.is_finite())
        );
        water.set_rapids(false);
        water.update(Duration::from_secs(1));
        assert!(!water.has_rapids());
        assert!(water.foam.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn weak_stokes_harmonics_sharpen_crests_and_flatten_troughs() {
        let a = 0.01;
        let k = 8.0;
        let crest = a + stokes_bound(a, a, 0.5 * k);
        let trough = -a + stokes_bound(-a, a, 0.5 * k);
        assert!(crest > a && trough > -a);
        assert!((crest + trough - k * a * a).abs() < 1e-7);
        let total_steepness: f32 = SWELL_MODES
            .iter()
            .map(|(kx, ky, amplitude)| (kx * kx + ky * ky).sqrt() * TAU / PATCH_LENGTH * amplitude)
            .sum();
        assert!(
            total_steepness < 0.3,
            "bound expansion must stay weakly nonlinear"
        );
    }

    #[test]
    fn surface_redistributes_volume_without_changing_the_mean_level() {
        let mut water = Water::new(120, 60);
        for i in 0..12 {
            water.add_drop(i as f32 / 12.0, 0.6, 0.3);
        }
        for age in [0.0, 0.1, 1.0, 3.0] {
            water.update(Duration::from_secs_f32(age));
            let mean = water
                .levels
                .iter()
                .map(|height| *height as f64)
                .sum::<f64>()
                / water.levels.len() as f64;
            assert!(mean.abs() < 1e-8, "mean height {mean}");
        }
    }

    #[test]
    fn empty_grids_accept_random_rain() {
        for (width, height) in [(0, 0), (0, 8), (8, 0)] {
            let mut water = Water::new(width, height);
            water.add_random_drop();
            water.update(Duration::from_millis(16));
            assert!(water.levels.is_empty());
        }
    }

    #[test]
    fn calm_water_is_alive_before_the_first_drop() {
        let water = Water::new(40, 20);
        let energy: f32 = water.levels.iter().map(|height| height * height).sum();
        assert!(
            energy > 0.0001,
            "background capillary waves must be visible initially"
        );
        assert!(water.levels.iter().all(|height| height.abs() < 0.06));
    }

    #[test]
    fn simulation_is_independent_of_frame_partition() {
        let mut one_frame = Water::new(40, 20);
        let mut many_frames = Water::new(40, 20);
        one_frame
            .wave_centers
            .push(WaveCenter::new(0.4, 0.6, 0.3, one_frame.time));
        many_frames
            .wave_centers
            .push(WaveCenter::new(0.4, 0.6, 0.3, many_frames.time));
        one_frame.update(Duration::from_secs(2));
        for _ in 0..125 {
            many_frames.update(Duration::from_millis(16));
        }
        for (left, right) in one_frame.levels.iter().zip(many_frames.levels.iter()) {
            assert!((left - right).abs() < 1e-6, "{left} differs from {right}");
        }
    }

    #[test]
    fn tiny_grids_stay_finite_through_resize() {
        let mut water = Water::new(1, 1);
        water.add_random_drop();
        for (width, height) in [(1, 8), (8, 1), (1, 1)] {
            water.resize(width, height);
            water.update(Duration::from_secs(1));
            assert_eq!(water.levels.len(), width as usize * height as usize);
            assert!(water.levels.iter().all(|height| height.is_finite()));
        }
    }

    #[test]
    fn storms_are_bounded_and_inputs_are_finite() {
        let mut water = Water::new(24, 12);
        water.add_drop(f32::NAN, 0.5, 0.5);
        water.add_drop(0.5, f32::INFINITY, 0.5);
        water.add_drop(0.5, 0.5, f32::NAN);
        assert!(water.wave_centers.is_empty());
        for _ in 0..300 {
            water.add_drop(-1.0, 2.0, 3.0);
        }
        assert_eq!(water.wave_centers.len(), 128);
        assert!(
            water
                .wave_centers
                .iter()
                .all(|wave| { wave.x == 0.0 && wave.y == 1.0 && wave.strength == 1.0 })
        );
        water.update(Duration::from_secs(2));
        assert!(
            water
                .levels
                .iter()
                .all(|height| height.is_finite() && height.abs() < 0.667)
        );
        water.update(SETTINGS.attenuation_time);
        assert!(water.wave_centers.is_empty());
    }

    #[test]
    fn resizing_preserves_time_and_impacts_without_accumulating_diffusion() {
        let mut water = Water::new(160, 160);
        water.add_drop(0.3, 0.7, 0.4);
        water.update(Duration::from_secs(3));
        let original = water.levels.clone();
        let instant = water.time;
        let elapsed = water.elapsed;
        water.resize(320, 320);
        assert_eq!(water.time, instant);
        assert_eq!(water.elapsed, elapsed);
        assert_eq!(water.wave_centers.len(), 1);
        assert!(water.levels.iter().all(|height| height.is_finite()));
        water.resize(160, 160);
        assert_eq!(water.levels, original);
    }
}
