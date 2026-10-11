//! Persistent, depth-averaged shallow-water rapids in a 2 m square channel.
//!
//! Conservative HLL fluxes, hydrostatic reconstruction, limited MUSCL slopes
//! and SSP-RK2 evolve water depth and both momenta. Subcritical boundaries impose
//! discharge upstream and stage downstream using the outgoing characteristic;
//! the transverse direction is periodic. Physical time uses integer fixed ticks.
//!
//! A hydraulic jump is resolved as a dissipative depth transition. Its vertical
//! overturning is not resolved by these equations. The near-surface return flow
//! used for bubbles is an explicitly empirical roller closure, not 3-D dynamics.
//! Jump-powered stochastic wave packets supply unresolved breaking detail. Their
//! damped, advected height is an optical closure, never part of conserved depth.
use std::time::Duration;

const NX: usize = 64;
const NY: usize = 48;
const CELLS: usize = NX * NY;
const LENGTH: f32 = 2.0;
const DX: f32 = LENGTH / NX as f32;
const DY: f32 = LENGTH / NY as f32;
const G: f32 = 9.81;
const REFERENCE: f32 = 0.10;
const DOWNSTREAM_STAGE: f32 = 0.09;
const MIN_DEPTH: f32 = 1e-5;
const TICK: Duration = Duration::from_millis(4);

#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    h: f32,
    mx: f32,
    my: f32,
}

impl Cell {
    fn primitive(
        self,
        bed: f32,
    ) -> [f32; 3] {
        [
            self.h + bed,
            self.mx / self.h.max(MIN_DEPTH),
            self.my / self.h.max(MIN_DEPTH),
        ]
    }

    fn valid(self) -> bool {
        self.h >= MIN_DEPTH && self.h.is_finite() && self.mx.is_finite() && self.my.is_finite()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RapidsSample {
    /// Rendered free-surface height relative to the 0.10 m datum, in metres.
    /// Includes zero-mean short-wave detail from the empirical breaking closure.
    pub elevation: f32,
    /// Conservative hydraulic depth; short-wave optical detail adds no water.
    pub depth: f32,
    pub foam: f32,
    /// Surface tracer velocity in m/s, including the empirical roller closure.
    pub velocity: [f32; 2],
}

#[derive(Debug)]
struct Workspace {
    primitive: Vec<[f32; 3]>,
    sx: Vec<[f32; 3]>,
    sy: Vec<[f32; 3]>,
}
impl Workspace {
    fn new() -> Self {
        Self {
            primitive: vec![[0.0; 3]; CELLS],
            sx: vec![[0.0; 3]; CELLS],
            sy: vec![[0.0; 3]; CELLS],
        }
    }
}

#[derive(Debug)]
pub struct Rapids {
    cells: Vec<Cell>,
    bed: Vec<f32>,
    foam: Vec<f32>,
    foam_next: Vec<f32>,
    // [displacement, vertical speed] of unresolved breaking-wave packets.
    ripples: Vec<[f32; 2]>,
    ripples_next: Vec<[f32; 2]>,
    surface: Vec<[f32; 2]>,
    source: Vec<f32>,
    roller: Vec<f32>,
    stage_a: Vec<Cell>,
    stage_b: Vec<Cell>,
    work: Workspace,
    pending: Duration,
    closed: bool,
    boundary_volume: f64,
    inlet: [f32; NY],
    upstream_noise: [f32; 8],
    noise_state: u32,
    breaking_noise: u32,
}

fn random_unit(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    // The upper endpoint is excluded, including after f32 rounding.
    (*state >> 8) as f32 / 16_777_216.0
}

fn discharge(y: f32) -> f32 {
    let phase = std::f32::consts::TAU * y / LENGTH;
    // Stationary, laterally nonuniform feed: shear is maintained by the inlet,
    // not prescribed moving vortex trajectories.
    0.045 * (1.0 + 0.15 * phase.cos() + 0.06 * (2.0 * phase).sin())
}

pub(crate) fn minmod(
    a: f32,
    b: f32,
    c: f32,
) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

fn boundary(
    interior: Cell,
    bed: f32,
    y: usize,
    upstream: bool,
    closed: bool,
    inlet: &[f32; NY],
) -> Cell {
    if closed {
        return Cell {
            mx: -interior.mx,
            ..interior
        };
    }
    let h = interior.h.max(MIN_DEPTH);
    let u = interior.mx / h;
    let c = (G * h).sqrt();
    if upstream {
        let q = inlet[y];
        let invariant = u - 2.0 * c;
        let mut depth = h;
        for _ in 0..5 {
            let wave = (G * depth).sqrt();
            let residual = q / depth - 2.0 * wave - invariant;
            let derivative = -q / (depth * depth) - wave / depth;
            depth = (depth - residual / derivative).clamp(0.005, 0.5);
        }
        Cell {
            h: depth,
            mx: q,
            my: 0.0,
        }
    } else if u >= c {
        interior
    } else {
        let depth = (DOWNSTREAM_STAGE - bed).max(MIN_DEPTH);
        let velocity = u + 2.0 * c - 2.0 * (G * depth).sqrt();
        Cell {
            h: depth,
            mx: depth * velocity,
            my: depth * interior.my / h,
        }
    }
}

fn neighbor(
    cells: &[Cell],
    bed: &[f32],
    x: isize,
    y: isize,
    closed: bool,
    inlet: &[f32; NY],
) -> (Cell, f32) {
    let row = y.rem_euclid(NY as isize) as usize;
    let col = x.clamp(0, NX as isize - 1) as usize;
    let i = row * NX + col;
    let cell = if x < 0 || x >= NX as isize {
        boundary(cells[i], bed[i], row, x < 0, closed, inlet)
    } else {
        cells[i]
    };
    (cell, bed[i])
}

/// Hydrostatic HLL interface. Return one conservative flux and the two bed
/// pressure corrections. At a lake at rest each side receives its own exact
/// hydrostatic pressure, including across unequal bottom elevations.
pub(crate) fn interface(
    left: [f32; 3],
    right: [f32; 3],
    zl: f32,
    zr: f32,
    axis: usize,
) -> ([f32; 3], f32, f32) {
    let bed = zl.max(zr);
    let hl = (left[0] - bed).max(0.0);
    let hr = (right[0] - bed).max(0.0);
    let original_l = (left[0] - zl).max(0.0);
    let original_r = (right[0] - zr).max(0.0);
    let ul = left[axis + 1];
    let ur = right[axis + 1];
    let sl = (ul - (G * hl).sqrt()).min(ur - (G * hr).sqrt()).min(0.0);
    let sr = (ul + (G * hl).sqrt()).max(ur + (G * hr).sqrt()).max(0.0);
    let ql = [hl, hl * left[1], hl * left[2]];
    let qr = [hr, hr * right[1], hr * right[2]];
    let mut fl = [hl * ul, ql[1] * ul, ql[2] * ul];
    let mut fr = [hr * ur, qr[1] * ur, qr[2] * ur];
    fl[axis + 1] += 0.5 * G * hl * hl;
    fr[axis + 1] += 0.5 * G * hr * hr;
    let mut flux = [0.0; 3];
    if sr - sl > 1e-8 {
        for k in 0..3 {
            flux[k] = (sr * fl[k] - sl * fr[k] + sl * sr * (qr[k] - ql[k])) / (sr - sl);
        }
    }
    (
        flux,
        0.5 * G * (original_l * original_l - hl * hl),
        0.5 * G * (original_r * original_r - hr * hr),
    )
}

fn add_flux(
    cell: &mut Cell,
    flux: [f32; 3],
    pressure: f32,
    axis: usize,
    scale: f32,
) {
    cell.h += scale * flux[0];
    cell.mx += scale * flux[1];
    cell.my += scale * flux[2];
    if axis == 0 {
        cell.mx += scale * pressure;
    } else {
        cell.my += scale * pressure;
    }
}

fn euler(
    cells: &[Cell],
    bed: &[f32],
    closed: bool,
    inlet: &[f32; NY],
    dt: f32,
    output: &mut [Cell],
    work: &mut Workspace,
) -> f64 {
    output.copy_from_slice(cells);
    for i in 0..CELLS {
        work.primitive[i] = cells[i].primitive(bed[i]);
    }
    for y in 0..NY {
        for x in 0..NX {
            let i = y * NX + x;
            let mid = work.primitive[i];
            for axis in 0..2 {
                let (dx, dy) = if axis == 0 { (1, 0) } else { (0, 1) };
                let (a, za) = neighbor(cells, bed, x as isize - dx, y as isize - dy, closed, inlet);
                let (b, zb) = neighbor(cells, bed, x as isize + dx, y as isize + dy, closed, inlet);
                let a = a.primitive(za);
                let b = b.primitive(zb);
                let slope = if axis == 0 {
                    &mut work.sx[i]
                } else {
                    &mut work.sy[i]
                };
                for k in 0..3 {
                    slope[k] = minmod(
                        1.3 * (mid[k] - a[k]),
                        0.5 * (b[k] - a[k]),
                        1.3 * (b[k] - mid[k]),
                    );
                }
                // Positive reconstructed depths; preserves exact flat free surfaces.
                slope[0] = slope[0].clamp(-1.8 * cells[i].h, 1.8 * cells[i].h);
            }
        }
    }
    let mut volume = 0.0_f64;
    for y in 0..NY {
        for face in 0..=NX {
            let li = y * NX + face.saturating_sub(1);
            let ri = y * NX + face.min(NX - 1);
            let mut l = work.primitive[li];
            let mut r = work.primitive[ri];
            if face == 0 {
                l = boundary(cells[ri], bed[ri], y, true, closed, inlet).primitive(bed[ri]);
            } else {
                for (k, v) in l.iter_mut().enumerate() {
                    *v += 0.5 * work.sx[li][k];
                }
            }
            if face == NX {
                r = boundary(cells[li], bed[li], y, false, closed, inlet).primitive(bed[li]);
            } else {
                for (k, v) in r.iter_mut().enumerate() {
                    *v -= 0.5 * work.sx[ri][k];
                }
            }
            // A wall mirrors the reconstructed face state. Reflecting the cell
            // centre would leave unequal face velocities whenever MUSCL has a
            // nonzero slope, introducing a spurious mass flux through the wall.
            if closed {
                if face == 0 {
                    l = [r[0], -r[1], r[2]];
                } else if face == NX {
                    r = [l[0], -l[1], l[2]];
                }
            }
            let (flux, pl, pr) = interface(l, r, bed[li], bed[ri], 0);
            if face > 0 {
                add_flux(&mut output[li], flux, pl, 0, -dt / DX);
            } else {
                volume += flux[0] as f64 * (dt * DY) as f64;
            }
            if face < NX {
                add_flux(&mut output[ri], flux, pr, 0, dt / DX);
            } else {
                volume -= flux[0] as f64 * (dt * DY) as f64;
            }
        }
    }
    for y in 0..NY {
        for x in 0..NX {
            let li = y * NX + x;
            let ri = ((y + 1) % NY) * NX + x;
            let mut l = work.primitive[li];
            let mut r = work.primitive[ri];
            for k in 0..3 {
                l[k] += 0.5 * work.sy[li][k];
                r[k] -= 0.5 * work.sy[ri][k];
            }
            let (flux, pl, pr) = interface(l, r, bed[li], bed[ri], 1);
            add_flux(&mut output[li], flux, pl, 1, -dt / DY);
            add_flux(&mut output[ri], flux, pr, 1, dt / DY);
        }
    }
    volume
}

fn sample_indices(point: [f32; 2]) -> ([usize; 4], [f32; 2]) {
    let x = (point[0].clamp(0.0, 1.0) * NX as f32 - 0.5).clamp(0.0, NX as f32 - 1.0);
    let y = point[1].rem_euclid(1.0) * NY as f32 - 0.5;
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(NX - 1);
    let yf = y.floor();
    let y0 = (yf as isize).rem_euclid(NY as isize) as usize;
    let y1 = (y0 + 1) % NY;
    ([y0 * NX + x0, y0 * NX + x1, y1 * NX + x0, y1 * NX + x1], [
        x - x0 as f32,
        y - yf,
    ])
}
fn interpolate(
    values: [f32; 4],
    blend: [f32; 2],
) -> f32 {
    let a = values[0] + (values[1] - values[0]) * blend[0];
    let b = values[2] + (values[3] - values[2]) * blend[0];
    a + (b - a) * blend[1]
}

impl Default for Rapids {
    fn default() -> Self { Self::new() }
}
impl Rapids {
    pub fn new() -> Self {
        let mut bed = vec![0.0; CELLS];
        for y in 0..NY {
            let py = (y as f32 + 0.5) * DY;
            let crest = 0.66 + 0.055 * (std::f32::consts::TAU * py / LENGTH).cos();
            for x in 0..NX {
                let px = (x as f32 + 0.5) * DX;
                let sill = 0.039 * (-((px - crest) / 0.17).powi(2)).exp();
                let boulder =
                    0.033 * (-((px - 1.02) / 0.13).powi(2) - ((py - 1.12) / 0.20).powi(2)).exp();
                let rocks = [
                    (1.14_f32, 0.36_f32, 0.075_f32, 0.15_f32, 0.21_f32),
                    (1.38, 1.28, 0.075, 0.16, 0.22),
                    (1.63, 1.75, 0.060, 0.14, 0.21),
                ]
                .into_iter()
                .map(|(cx, cy, h, rx, ry)| {
                    h * (-((px - cx) / rx).powi(2) - ((py - cy) / ry).powi(2)).exp()
                })
                .sum::<f32>();
                bed[y * NX + x] = sill + boulder * 0.3 + rocks;
            }
        }
        let mut cells = vec![Cell::default(); CELLS];
        for y in 0..NY {
            let py = (y as f32 + 0.5) * DY;
            let q = discharge(py);
            let hc = (q * q / G).cbrt();
            let peak = bed[y * NX..(y + 1) * NX]
                .iter()
                .copied()
                .fold(0.0_f32, f32::max);
            let energy = peak + 1.5 * hc;
            let crest = 0.66 + 0.055 * (std::f32::consts::TAU * py / LENGTH).cos();
            for x in 0..NX {
                let i = y * NX + x;
                let px = (x as f32 + 0.5) * DX;
                let upstream = px < crest;
                let mut lo = if upstream { hc } else { 0.004 };
                let mut hi = if upstream { 0.3 } else { hc };
                for _ in 0..22 {
                    let h = 0.5 * (lo + hi);
                    let residual = h + q * q / (2.0 * G * h * h) + bed[i] - energy;
                    if (residual > 0.0) == upstream {
                        hi = h;
                    } else {
                        lo = h;
                    }
                }
                let depth = if px > 1.20 {
                    (DOWNSTREAM_STAGE - bed[i]).max(0.02)
                } else {
                    0.5 * (lo + hi)
                };
                let seed = 0.0015 * (px * 11.3 + py * 7.7).sin() * (px * 4.1 - py * 9.3).sin();
                cells[i] = Cell {
                    h: depth,
                    mx: q,
                    my: depth * seed,
                };
            }
        }
        let mut flow = Self {
            cells,
            bed,
            foam: vec![0.0; CELLS],
            foam_next: vec![0.0; CELLS],
            ripples: vec![[0.0; 2]; CELLS],
            ripples_next: vec![[0.0; 2]; CELLS],
            surface: vec![[0.0; 2]; CELLS],
            source: vec![0.0; CELLS],
            roller: vec![0.0; CELLS],
            stage_a: vec![Cell::default(); CELLS],
            stage_b: vec![Cell::default(); CELLS],
            work: Workspace::new(),
            pending: Duration::ZERO,
            closed: false,
            boundary_volume: 0.0,
            inlet: std::array::from_fn(|y| discharge((y as f32 + 0.5) * DY)),
            upstream_noise: [0.0; 8],
            noise_state: 0x46B9_57AC,
            breaking_noise: 0x91A2_43CB,
        };
        flow.update_tracers(0.0);
        flow
    }

    pub fn advance(
        &mut self,
        elapsed: Duration,
    ) {
        self.pending = self.pending.saturating_add(elapsed);
        while self.pending >= TICK {
            self.pending -= TICK;
            self.update_inlet();
            let mut remaining = TICK.as_secs_f32();
            while remaining > 1e-8 {
                let max_rate = self
                    .cells
                    .iter()
                    .map(|c| {
                        let wave = (G * c.h).sqrt();
                        (c.mx.abs() / c.h + wave) / DX + (c.my.abs() / c.h + wave) / DY
                    })
                    .fold(0.0_f32, f32::max);
                let mut dt = remaining.min(0.38 / max_rate.max(1.0));
                loop {
                    let va = euler(
                        &self.cells,
                        &self.bed,
                        self.closed,
                        &self.inlet,
                        dt,
                        &mut self.stage_a,
                        &mut self.work,
                    );
                    if self.stage_a.iter().all(|c| c.valid()) {
                        let vb = euler(
                            &self.stage_a,
                            &self.bed,
                            self.closed,
                            &self.inlet,
                            dt,
                            &mut self.stage_b,
                            &mut self.work,
                        );
                        if self.stage_b.iter().all(|c| c.valid()) {
                            for (cell, next) in self.cells.iter_mut().zip(&self.stage_b) {
                                cell.h = 0.5 * (cell.h + next.h);
                                cell.mx = 0.5 * (cell.mx + next.mx);
                                cell.my = 0.5 * (cell.my + next.my);
                            }
                            self.boundary_volume += 0.5 * (va + vb);
                            break;
                        }
                    }
                    // Reject an unsafe stage instead of clamping depth and
                    // silently manufacturing water. CFL reduction is conservative.
                    dt *= 0.5;
                    assert!(dt > 1e-9, "rapids positivity step collapsed");
                }
                self.update_tracers(dt);
                remaining -= dt;
            }
        }
    }

    fn update_inlet(&mut self) {
        // Intermittent upstream discharge models incoming turbulent water.
        // An OU process has a physical 0.8 s correlation time; its innovations
        // are deterministic on fixed ticks, never tied to display cadence.
        let decay = (-TICK.as_secs_f32() / 0.8).exp();
        let noise = (1.0 - decay * decay).sqrt();
        for value in &mut self.upstream_noise {
            let random = random_unit(&mut self.noise_state);
            *value =
                (*value * decay + (random * 2.0 - 1.0) * 3.0_f32.sqrt() * noise).clamp(-2.5, 2.5);
        }
        for y in 0..NY {
            let position = (y as f32 + 0.5) / NY as f32 * self.upstream_noise.len() as f32;
            let i = position.floor() as usize;
            let blend = position - i as f32;
            let blend = blend * blend * (3.0 - 2.0 * blend);
            let perturbation = self.upstream_noise[i] * (1.0 - blend)
                + self.upstream_noise[(i + 1) % self.upstream_noise.len()] * blend;
            self.inlet[y] = discharge((y as f32 + 0.5) * DY) * (1.0 + 0.20 * perturbation);
        }
    }

    /// A local horizontal pressure kick. It changes momentum only; water volume
    /// and positivity are unchanged, including for an impact beside a boundary.
    pub fn disturb(
        &mut self,
        point: [f32; 2],
        strength: f32,
    ) {
        self.disturb_sized(point, strength, 1.0);
    }

    /// The footprint scales independently from impact strength. The size range
    /// matches the rain control; it changes momentum support, never water mass.
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
        {
            return;
        }
        let point = [
            point[0].clamp(0.0, 1.0) * LENGTH,
            point[1].rem_euclid(1.0) * LENGTH,
        ];
        let sigma = (0.055 + 0.035 * strength.min(1.0).sqrt()) * size.clamp(0.25, 3.0);
        let impulse = 0.08 * strength.min(1.0);
        for y in 0..NY {
            let dy =
                ((y as f32 + 0.5) * DY - point[1] + LENGTH * 0.5).rem_euclid(LENGTH) - LENGTH * 0.5;
            for x in 0..NX {
                let dx = (x as f32 + 0.5) * DX - point[0];
                let r2 = (dx * dx + dy * dy) / (sigma * sigma);
                if r2 > 20.0 {
                    continue;
                }
                let weight = impulse * (-0.5 * r2).exp() / sigma;
                let cell = &mut self.cells[y * NX + x];
                cell.mx += cell.h * weight * dx;
                cell.my += cell.h * weight * dy;
            }
        }
        self.update_tracers(0.0);
    }

    fn update_tracers(
        &mut self,
        dt: f32,
    ) {
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let (l, zl) = neighbor(
                    &self.cells,
                    &self.bed,
                    x as isize - 1,
                    y as isize,
                    self.closed,
                    &self.inlet,
                );
                let (r, zr) = neighbor(
                    &self.cells,
                    &self.bed,
                    x as isize + 1,
                    y as isize,
                    self.closed,
                    &self.inlet,
                );
                let (b, _) = neighbor(
                    &self.cells,
                    &self.bed,
                    x as isize,
                    y as isize - 1,
                    self.closed,
                    &self.inlet,
                );
                let (t, _) = neighbor(
                    &self.cells,
                    &self.bed,
                    x as isize,
                    y as isize + 1,
                    self.closed,
                    &self.inlet,
                );
                let cell = self.cells[i];
                let ux = (r.mx / r.h - l.mx / l.h) / (2.0 * DX);
                let vy = (t.my / t.h - b.my / b.h) / (2.0 * DY);
                let uy = (t.mx / t.h - b.mx / b.h) / (2.0 * DY);
                let vx = (r.my / r.h - l.my / l.h) / (2.0 * DX);
                let compression = (-(ux + vy)).max(0.0);
                let strain = ((ux - vy).powi(2) + (uy + vx).powi(2)).sqrt();
                let rise = ((r.h + zr - l.h - zl) / (2.0 * DX)).max(0.0);
                let upstream_fr = (l.mx.abs() / l.h / (G * l.h).sqrt())
                    .max(cell.mx.abs() / cell.h / (G * cell.h).sqrt());
                let transition = ((upstream_fr - 0.78) / 0.65).clamp(0.0, 1.0);
                let jump = (compression / 4.0).clamp(0.0, 1.0)
                    * (rise / 0.10).clamp(0.0, 1.0)
                    * transition;
                self.roller[i] = jump;
                self.source[i] = 3.5 * jump + 0.025 * strain.min(20.0) * transition;
            }
        }
        for y in 0..NY {
            let mut previous = 0.0_f32;
            for x in 0..NX {
                let i = y * NX + x;
                let cell = self.cells[i];
                // A jump's surface recirculation extends a few local water depths
                // downstream; the unresolved lower return branch is not inserted
                // into the depth-averaged conservative momentum equations.
                let roller = self.roller[i].max(previous * (-DX / (2.8 * cell.h)).exp());
                previous = roller;
                let fraction = (1.65 * roller).min(0.95);
                let reverse = -0.55 * (G * cell.h).sqrt();
                self.surface[i] = [
                    cell.mx / cell.h * (1.0 - fraction) + reverse * fraction,
                    cell.my / cell.h,
                ];
                self.source[i] += 1.5 * roller;
            }
        }
        if dt == 0.0 {
            return;
        }
        self.excite_breaking(dt);
        // A narrow-band gravity-wave closure around lambda=0.18 m. The local
        // c^2=g*tanh(kh)/k uses finite depth; resolved hydraulic motion remains
        // exclusively in the conservative solver. Breaking damps this detail.
        let k = std::f32::consts::TAU / 0.18;
        let ripple_decay = (-dt / 0.65).exp();
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let a = self.ripples[i];
                let left = self.ripples[y * NX + x.saturating_sub(1)][0];
                let right = self.ripples[y * NX + (x + 1).min(NX - 1)][0];
                let down = self.ripples[((y + NY - 1) % NY) * NX + x][0];
                let up = self.ripples[((y + 1) % NY) * NX + x][0];
                let laplacian =
                    (left + right - 2.0 * a[0]) / (DX * DX) + (down + up - 2.0 * a[0]) / (DY * DY);
                let c2 = G * (k * self.cells[i].h).tanh() / k;
                let speed = (a[1] + dt * c2 * laplacian) * ripple_decay;
                self.ripples_next[i] = [a[0] + dt * speed, speed];
            }
        }
        let decay = (-dt / 3.2).exp();
        for y in 0..NY {
            for x in 0..NX {
                let i = y * NX + x;
                let velocity = self.surface[i];
                let from = [
                    ((x as f32 + 0.5) * DX - velocity[0] * dt) / LENGTH,
                    ((y as f32 + 0.5) * DY - velocity[1] * dt) / LENGTH,
                ];
                let (indices, blend) = sample_indices(from);
                let prior = if from[0] < 0.0 {
                    0.0
                } else {
                    interpolate(indices.map(|j| self.foam[j]), blend)
                };
                let surviving = prior * decay;
                self.foam_next[i] = 1.0 - (1.0 - surviving) * (-0.08 * self.source[i] * dt).exp();
                self.ripples[i] = if from[0] < 0.0 {
                    [0.0; 2]
                } else {
                    std::array::from_fn(|axis| {
                        interpolate(indices.map(|j| self.ripples_next[j][axis]), blend)
                    })
                };
            }
        }
        // A visual corrugation cannot shift the patch datum or add water. This
        // projects only the subgrid closure; hydraulic depth is never adjusted.
        let mean = self
            .ripples
            .iter()
            .fold([0.0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        for ripple in &mut self.ripples {
            ripple[0] -= mean[0] / CELLS as f32;
            ripple[1] -= mean[1] / CELLS as f32;
        }
        // Depth-limited breaking dissipates excess unresolved wave energy.
        // Alternating projection keeps both the local amplitude ceiling and
        // the zero-mean optical datum; the conservative hydraulic depth stays
        // untouched. A clipped outward velocity is lost to unresolved spray.
        for _ in 0..10 {
            for (ripple, cell) in self.ripples.iter_mut().zip(&self.cells) {
                let limit = (0.20 * cell.h).min(0.015);
                let bounded = ripple[0].clamp(-limit, limit);
                if bounded != ripple[0] && ripple[0] * ripple[1] > 0.0 {
                    ripple[1] = 0.0;
                }
                ripple[0] = bounded;
            }
            let mean = self.ripples.iter().map(|r| r[0]).sum::<f32>() / CELLS as f32;
            if mean.abs() < 1e-8 {
                break;
            }
            for ripple in &mut self.ripples {
                ripple[0] -= mean;
            }
        }
        std::mem::swap(&mut self.foam, &mut self.foam_next);
    }

    fn excite_breaking(
        &mut self,
        dt: f32,
    ) {
        // Random entrainment occurs only where the resolved flow loses energy
        // to compression/strain. Packet positions are sampled in physical time,
        // then their crests and bubbles travel with the local surface current.
        // This is a stochastic breaking closure, not resolved 3-D turbulence.
        for _ in 0..2 {
            let i = (random_unit(&mut self.breaking_noise) * CELLS as f32) as usize;
            let probability = dt * CELLS as f32 * 0.012 * self.source[i] / 2.0;
            if random_unit(&mut self.breaking_noise) >= probability {
                continue;
            }
            let radius = 0.055 + 0.045 * random_unit(&mut self.breaking_noise);
            let impulse = (0.50 + 0.70 * random_unit(&mut self.breaking_noise))
                * if random_unit(&mut self.breaking_noise) < 0.5 {
                    -1.0
                } else {
                    1.0
                };
            let cx = i % NX;
            let cy = i / NX;
            let extent = (3.5 * radius / DX).ceil() as usize;
            for offset_y in -(extent as isize)..=extent as isize {
                let y = (cy as isize + offset_y).rem_euclid(NY as isize) as usize;
                for x in cx.saturating_sub(extent)..=(cx + extent).min(NX - 1) {
                    let dx = (x as f32 - cx as f32) * DX / radius;
                    let dy = offset_y as f32 * DY / radius;
                    let r2 = dx * dx + dy * dy;
                    let envelope = (-0.5 * r2).exp();
                    let j = y * NX + x;
                    self.ripples[j][1] += impulse * (1.0 - 0.5 * r2) * envelope;
                    // Patchy entrainment, transported and dissolved by the same
                    // surface tracer update as the smooth mean foam coverage.
                    self.foam[j] += (1.0 - self.foam[j]) * 0.90 * envelope;
                }
            }
        }
    }

    /// Mean velocity of the full water column, in m/s. Dissolved column mass
    /// follows this momentum flux; the empirical surface return flow is for foam.
    pub fn depth_averaged_velocity(
        &self,
        point: [f32; 2],
    ) -> [f32; 2] {
        if point.iter().any(|p| !p.is_finite()) {
            return [0.0; 2];
        }
        let (indices, blend) = sample_indices(point);
        let depth = interpolate(indices.map(|i| self.cells[i].h), blend).max(MIN_DEPTH);
        [
            interpolate(indices.map(|i| self.cells[i].mx), blend) / depth,
            interpolate(indices.map(|i| self.cells[i].my), blend) / depth,
        ]
    }

    pub fn sample(
        &self,
        point: [f32; 2],
    ) -> RapidsSample {
        if !point.iter().all(|p| p.is_finite()) {
            return RapidsSample::default();
        }
        let (indices, blend) = sample_indices(point);
        RapidsSample {
            elevation: interpolate(
                indices.map(|i| self.cells[i].h + self.bed[i] - REFERENCE + self.ripples[i][0]),
                blend,
            ),
            depth: interpolate(indices.map(|i| self.cells[i].h), blend),
            foam: interpolate(indices.map(|i| self.foam[i]), blend).clamp(0.0, 1.0),
            velocity: std::array::from_fn(|axis| {
                interpolate(indices.map(|i| self.surface[i][axis]), blend)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dissolved_column_velocity_uses_water_momentum_not_surface_return_flow() {
        let mut flow = Rapids::new();
        for (i, cell) in flow.cells.iter_mut().enumerate() {
            let h = 0.04 + 0.08 * (i % NX) as f32 / NX as f32;
            *cell = Cell {
                h,
                mx: 0.45 * h,
                my: -0.08 * h,
            };
        }
        flow.surface.fill([-0.3, 0.7]);
        for point in [[0.0, 0.0], [0.371, 0.513], [1.0, 1.0]] {
            let velocity = flow.depth_averaged_velocity(point);
            assert!((velocity[0] - 0.45).abs() < 1e-6);
            assert!((velocity[1] + 0.08).abs() < 1e-6);
            assert!((flow.sample(point).velocity[0] + 0.3).abs() < 1e-6);
        }
        assert_eq!(flow.depth_averaged_velocity([f32::NAN, 0.5]), [0.0; 2]);
    }

    fn volume(flow: &Rapids) -> f64 {
        flow.cells.iter().map(|c| c.h as f64).sum::<f64>() * (DX * DY) as f64
    }
    fn resting() -> Rapids {
        let mut flow = Rapids::new();
        flow.closed = true;
        for (i, c) in flow.cells.iter_mut().enumerate() {
            *c = Cell {
                h: 0.16 - flow.bed[i],
                mx: 0.0,
                my: 0.0,
            };
        }
        flow.surface.fill([0.0; 2]);
        flow
    }
    #[test]
    fn closed_walls_have_zero_mass_flux_with_nonzero_muscl_velocity_slopes() {
        for upstream in [true, false] {
            let mut cells = vec![Cell::default(); CELLS];
            for y in 0..NY {
                for x in 0..NX {
                    let distance = if upstream { x } else { NX - 1 - x };
                    let speed = 0.2 + 0.2 * distance.min(2) as f32;
                    cells[y * NX + x] = Cell {
                        h: 0.1,
                        mx: 0.1 * speed * if upstream { 1.0 } else { -1.0 },
                        my: 0.013,
                    };
                }
            }
            let mut output = vec![Cell::default(); CELLS];
            let mut work = Workspace::new();
            let volume = euler(
                &cells,
                &vec![0.0; CELLS],
                true,
                &[0.045; NY],
                0.001,
                &mut output,
                &mut work,
            );
            let edge = if upstream { 0 } else { NX - 1 };
            assert!(
                work.sx[edge][1].abs() > 0.1,
                "regression must exercise nonzero face reconstruction"
            );
            assert_eq!(volume, 0.0, "a closed wall cannot pass water");
        }
    }

    #[test]
    fn hydrostatic_lake_stays_at_rest_over_a_nonflat_bed() {
        let mut flow = resting();
        let start = flow.cells.clone();
        flow.advance(Duration::from_millis(400));
        for (a, b) in start.iter().zip(&flow.cells) {
            assert!((a.h - b.h).abs() < 2e-6);
            assert!(b.mx.abs() < 2e-6 && b.my.abs() < 2e-6);
        }
        assert!(flow.bed.iter().copied().fold(0.0_f32, f32::max) > 0.03);
        assert!(flow.ripples.iter().all(|v| v[0] == 0.0 && v[1] == 0.0));
    }
    #[test]
    fn finite_volume_mass_matches_the_open_boundary_flux() {
        let mut flow = Rapids::new();
        let initial = volume(&flow);
        flow.advance(Duration::from_secs(1));
        let residual = volume(&flow) - initial - flow.boundary_volume;
        assert!(residual.abs() < 3e-6, "mass residual: {residual}");
        assert!(
            flow.boundary_volume.abs() > 1e-6,
            "boundary flow must develop"
        );
        assert!(
            flow.cells
                .iter()
                .all(|c| c.h > 0.0001 && c.h.is_finite() && c.mx.is_finite() && c.my.is_finite())
        );
    }
    #[test]
    fn flow_advances_independently_of_display_partition() {
        let mut a = Rapids::new();
        let mut b = Rapids::new();
        a.advance(Duration::from_secs(1));
        for _ in 0..100 {
            b.advance(Duration::from_millis(10));
        }
        assert_eq!(a.pending, b.pending);
        for (x, y) in a.cells.iter().zip(&b.cells) {
            assert_eq!([x.h, x.mx, x.my], [y.h, y.mx, y.my]);
        }
        assert_eq!(a.foam, b.foam);
        assert_eq!(a.ripples, b.ripples);
    }
    #[test]
    fn foam_decays_at_rest_without_new_bubbles() {
        let mut flow = resting();
        flow.foam.fill(0.8);
        flow.advance(Duration::from_millis(400));
        let expected = 0.8 * (-0.4_f32 / 3.2).exp();
        assert!(flow.foam.iter().all(|v| (v - expected).abs() < 3e-5));
    }
    #[test]
    fn developed_rapids_have_bounded_foam_and_surface_reverse_flow() {
        let mut flow = Rapids::new();
        flow.advance(Duration::from_secs(4));
        assert!(
            flow.foam
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(flow.foam.iter().any(|v| *v > 0.04));
        assert!(
            flow.surface.iter().any(|v| v[0] < -0.02),
            "a hydraulic roller must return some surface water upstream"
        );
        let sample = flow.sample([0.55, 0.5]);
        assert!(sample.depth > 0.001 && sample.depth < 0.3);
        assert!(sample.elevation.is_finite());
        assert!(sample.velocity.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn upstream_variability_keeps_the_developed_current_alive() {
        let mut flow = Rapids::new();
        let mass = volume(&flow);
        flow.advance(Duration::from_secs(35));
        let before = flow.cells.clone();
        flow.advance(Duration::from_millis(800));
        let rms = (flow
            .cells
            .iter()
            .zip(&before)
            .map(|(a, b)| (a.h - b.h).powi(2))
            .sum::<f32>()
            / CELLS as f32)
            .sqrt();
        assert!(rms > 0.00015, "late-time depth motion vanished: {rms}");
        let residual = volume(&flow) - mass - flow.boundary_volume;
        assert!(residual.abs() < 2e-5, "long-run mass residual: {residual}");
        assert!(flow.cells.iter().all(|c| c.valid()));
        assert!(
            flow.ripples
                .iter()
                .all(|v| v[0].is_finite() && v[0].abs() < 0.02)
        );
    }

    #[test]
    fn breaking_flow_has_moving_short_surface_waves_without_added_water() {
        let mut flow = Rapids::new();
        flow.advance(Duration::from_secs(4));
        let detail = |flow: &Rapids| -> Vec<f32> {
            (0..CELLS)
                .map(|i| {
                    let p = [
                        (i % NX) as f32 / NX as f32 + 0.5 / NX as f32,
                        (i / NX) as f32 / NY as f32 + 0.5 / NY as f32,
                    ];
                    flow.sample(p).elevation - (flow.cells[i].h + flow.bed[i] - REFERENCE)
                })
                .collect()
        };
        let a = detail(&flow);
        flow.advance(Duration::from_millis(300));
        let b = detail(&flow);
        let rms = (b.iter().map(|v| v * v).sum::<f32>() / CELLS as f32).sqrt();
        let motion =
            (a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / CELLS as f32).sqrt();
        assert!(
            rms > 0.0002,
            "a hydraulic jump needs resolved short surface detail: {rms}"
        );
        assert!(
            motion > 0.0002,
            "breaking detail must move in physical time: {motion}"
        );
        assert!(b.iter().sum::<f32>().abs() / (CELLS as f32) < 1e-6);
        assert!(b.iter().all(|v| v.is_finite() && v.abs() <= 0.015 + 1e-7));
    }
    #[test]
    fn impacts_change_momentum_without_changing_water_volume() {
        let mut a = Rapids::new();
        let mut b = Rapids::new();
        let initial = volume(&a);
        a.disturb([0.0, 0.99], 0.6);
        a.disturb([0.50, 0.50], 0.6);
        a.disturb([f32::NAN, 0.2], 0.5);
        assert_eq!(volume(&a), initial);
        assert!(a.cells.iter().all(|c| c.valid()));
        let kick = a
            .cells
            .iter()
            .zip(&b.cells)
            .map(|(a, b)| (a.mx - b.mx).abs() + (a.my - b.my).abs())
            .sum::<f32>();
        assert!(kick > 0.0001);
        a.advance(Duration::from_millis(80));
        b.advance(Duration::from_millis(80));
        assert!(
            a.cells
                .iter()
                .zip(&b.cells)
                .any(|(a, b)| (a.h - b.h).abs() > 1e-5)
        );
    }

    #[test]
    fn impact_size_changes_footprint_without_adding_water() {
        let original = Rapids::new();
        let footprint = |size: f32| {
            let mut flow = Rapids::new();
            flow.disturb_sized([0.5, 0.5], 0.7, size);
            assert_eq!(volume(&flow), volume(&original));
            let mut weight = 0.0;
            let mut moment = 0.0;
            for (i, (a, b)) in flow.cells.iter().zip(&original.cells).enumerate() {
                let energy = (a.mx - b.mx).powi(2) + (a.my - b.my).powi(2);
                let x = ((i % NX) as f32 + 0.5) * DX - 1.0;
                let y = ((i / NX) as f32 + 0.5) * DY - 1.0;
                weight += energy;
                moment += energy * (x * x + y * y);
            }
            moment / weight
        };
        let small = footprint(0.5);
        let normal = footprint(1.0);
        let large = footprint(2.0);
        assert!(
            normal > 2.5 * small,
            "a smaller drop must have a narrower impulse"
        );
        assert!(
            large > 2.5 * normal,
            "a larger drop must have a wider impulse"
        );
    }

    #[test]
    fn developed_rollers_keep_visible_but_depth_bounded_breaking_detail() {
        let mut flow = Rapids::new();
        flow.advance(Duration::from_secs(10));
        let active: Vec<_> = (0..CELLS).filter(|&i| flow.roller[i] > 0.2).collect();
        let rms = (active
            .iter()
            .map(|&i| flow.ripples[i][0].powi(2))
            .sum::<f32>()
            / active.len() as f32)
            .sqrt();
        assert!(
            rms > 0.004,
            "active rollers need visible short-wave detail: {rms}"
        );
        for (ripple, cell) in flow.ripples.iter().zip(&flow.cells) {
            assert!(ripple[0].abs() <= (0.20 * cell.h).min(0.015) + 1e-7);
        }
        let mean = flow.ripples.iter().map(|r| r[0]).sum::<f32>() / CELLS as f32;
        assert!(mean.abs() < 1e-7);
    }
}
