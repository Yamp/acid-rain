use crate::colors::{sky_color, water_body_color};
use anyhow::Result;
use crossterm::style::Color;
use crossterm::{cursor, queue, style, terminal};
use ndarray::Array2;
use std::f32::consts::PI;
use std::io::{Stdout, Write};
use std::time::Instant;
use crate::water::Water;


const LIGHT_ELEVATION: f32 = 0.8;
const LIGHT_RADIUS: f32 = 0.3;
const LIGHT_PERIOD: f32 = 60.0;

const CAMERA_ROLL_PERIOD: f32 = 23.0;
const CAMERA_ELEV_DEG: f32 = 70.0;   // 70° above the surface
const CAMERA_DIST: f32 = 1.0;
const CAMERA_FOV_DEG: f32 = 60.0;    // horizontal FOV
const CHAR_ASPECT: f32 = 2.0;        // terminal char height / width
const WATER_SCALE: f32 = 2.0;        // water plane size multiplier

const DRIFT_SPEED: f32 = 0.15;       // world units per second
const DRIFT_TURN_PERIOD: f32 = 180.0; // heading rotation period (lazy circle)

const FOG_DENSITY: f32 = 0.08;

const TAA_KEEP: f32 = 0.20;
const TAA_NEW: f32 = 1.0 - TAA_KEEP;

const CA_STRENGTH: f32 = 0.004; // chromatic aberration — max UV shift at corners

const GOD_RAY_SAMPLES: usize = 16;
const GOD_RAY_DECAY: f32 = 0.96;
const GOD_RAY_DENSITY: f32 = 0.5;
const GOD_RAY_EXPOSURE: f32 = 0.12;

const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_INTENSITY: f32 = 0.15;
const BLOOM_RADIUS: i32 = 3;
// Gaussian σ≈2, 7-tap separable kernel
const BLOOM_KERNEL: [f32; 7] = [0.0702, 0.1311, 0.1907, 0.2161, 0.1907, 0.1311, 0.0702];

const NORMAL_STRENGTH: f32 = 7.0;
const NDF_FILTER_K: f32 = 0.5;

const CAUSTIC_STRENGTH: f32 = 12.0;
const CAUSTIC_DISPERSION: f32 = 0.012; // R/B chromatic offset in UV
const AMBIENT: f32 = 0.55;
const DIFFUSE_K: f32 = 0.4;
const GGX_ROUGHNESS: f32 = 0.08;

// Artistic Fresnel R0 per channel (chromatic dispersion: blue reflects more)
const R0_R: f32 = 0.25;
const R0_G: f32 = 0.30;
const R0_B: f32 = 0.35;

// Subsurface scattering
const SSS_STRENGTH: f32 = 0.15;
const SSS_DISTORTION: f32 = 0.3;
const SSS_POWER: f32 = 3.0;

// ── vector helpers ──────────────────────────────────────────────────

#[inline(always)]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-10 { return [0.0, 0.0, 1.0]; }
    [v[0] / l, v[1] / l, v[2] / l]
}

#[inline(always)]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ── Schlick Fresnel per-channel (chromatic, artistic R0) ────────────

#[inline(always)]
fn fresnel_rgb(cos_theta: f32) -> [f32; 3] {
    let ct = cos_theta.max(0.0);
    let omc = (1.0 - ct).powi(5);
    [
        R0_R + (1.0 - R0_R) * omc,
        R0_G + (1.0 - R0_G) * omc,
        R0_B + (1.0 - R0_B) * omc,
    ]
}

#[inline(always)]
fn fresnel_scalar(cos_theta: f32) -> f32 {
    let r0 = (R0_R + R0_G + R0_B) / 3.0;
    let omc = (1.0 - cos_theta.max(0.0)).powi(5);
    r0 + (1.0 - r0) * omc
}

// ── ACES filmic tone mapping ────────────────────────────────────────

#[inline(always)]
fn aces_tonemap(x: f32) -> f32 {
    let a = x * (x * 2.51 + 0.03);
    let b = x * (x * 2.43 + 0.59) + 0.14;
    (a / b).clamp(0.0, 1.0)
}

// ── linear → sRGB gamma ────────────────────────────────────────────

#[inline(always)]
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

// ── interleaved gradient noise (Jimenez 2014) ──────────────────────

#[inline(always)]
fn ign(x: f32, y: f32, frame: f32) -> f32 {
    let t = 0.06711056 * x + 0.00583715 * y + 0.00519673 * frame;
    ((52.9829189 * t.fract()).fract()) - 0.5
}

// ── sky envmap (octahedral, precomputed per frame) ──────────────────

const SKY_ENV_SIZE: usize = 64;

/// Octahedral mapping: direction → UV [0,1]² (no transcendentals).
#[inline(always)]
fn dir_to_oct(d: [f32; 3]) -> (f32, f32) {
    let inv = 1.0 / (d[0].abs() + d[1].abs() + d[2].abs());
    let mut u = d[0] * inv;
    let mut v = d[1] * inv;
    if d[2] < 0.0 {
        let su = u;
        let sv = v;
        u = (1.0 - sv.abs()) * if su >= 0.0 { 1.0 } else { -1.0 };
        v = (1.0 - su.abs()) * if sv >= 0.0 { 1.0 } else { -1.0 };
    }
    (u * 0.5 + 0.5, v * 0.5 + 0.5)
}

/// Inverse: UV [0,1]² → direction.
#[inline(always)]
fn oct_to_dir(uu: f32, vv: f32) -> [f32; 3] {
    let u = uu * 2.0 - 1.0;
    let v = vv * 2.0 - 1.0;
    let z = 1.0 - u.abs() - v.abs();
    let (u, v) = if z < 0.0 {
        (
            (1.0 - v.abs()) * if u >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - u.abs()) * if v >= 0.0 { 1.0 } else { -1.0 },
        )
    } else {
        (u, v)
    };
    norm3([u, v, z])
}

fn precompute_sky(elapsed: f32, out: &mut Vec<(f32, f32, f32)>) {
    let n = SKY_ENV_SIZE;
    out.resize(n * n, (0.0, 0.0, 0.0));
    let inv = 1.0 / n as f32;
    for y in 0..n {
        for x in 0..n {
            let dir = oct_to_dir((x as f32 + 0.5) * inv, (y as f32 + 0.5) * inv);
            out[y * n + x] = sky_color(elapsed, dir);
        }
    }
}

#[inline(always)]
fn sample_sky(env: &[(f32, f32, f32)], dir: [f32; 3]) -> (f32, f32, f32) {
    let n = SKY_ENV_SIZE as f32;
    let (u, v) = dir_to_oct(dir);
    let fx = (u * n).clamp(0.0, n - 1.001);
    let fy = (v * n).clamp(0.0, n - 1.001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let ns = SKY_ENV_SIZE;
    let p00 = env[y0 * ns + x0];
    let p10 = env[y0 * ns + x0 + 1];
    let p01 = env[(y0 + 1) * ns + x0];
    let p11 = env[(y0 + 1) * ns + x0 + 1];
    (
        (p00.0 + (p10.0 - p00.0) * tx) * (1.0 - ty) + (p01.0 + (p11.0 - p01.0) * tx) * ty,
        (p00.1 + (p10.1 - p00.1) * tx) * (1.0 - ty) + (p01.1 + (p11.1 - p01.1) * tx) * ty,
        (p00.2 + (p10.2 - p00.2) * tx) * (1.0 - ty) + (p01.2 + (p11.2 - p01.2) * tx) * ty,
    )
}

// ── camera ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Camera {
    pos: [f32; 3],
    ground: [f32; 2],   // look-at point on z=0 (water plane center)
    fwd: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    half_w: f32,
    half_h: f32,
}

impl Camera {
    fn new(elapsed: f32, sw: u16, sh: u16) -> Self {
        let el = CAMERA_ELEV_DEG.to_radians();

        // Drift: heading slowly rotates → lazy circle
        let heading = elapsed / DRIFT_TURN_PERIOD * 2.0 * PI;
        let drift_r = DRIFT_SPEED * DRIFT_TURN_PERIOD / (2.0 * PI);
        let ground = [
            drift_r * heading.sin(),
            -drift_r * heading.cos(),
        ];

        // Camera sits behind and above the ground point
        let pos = [
            ground[0] - CAMERA_DIST * el.cos() * heading.cos(),
            ground[1] - CAMERA_DIST * el.cos() * heading.sin(),
            CAMERA_DIST * el.sin(),
        ];

        let fwd = norm3([ground[0] - pos[0], ground[1] - pos[1], -pos[2]]);
        // right = fwd × world_up(0,0,1)  →  (fwd.y, -fwd.x, 0)
        let r0 = norm3([fwd[1], -fwd[0], 0.0]);
        let u0 = [
            r0[1] * fwd[2] - r0[2] * fwd[1],
            r0[2] * fwd[0] - r0[0] * fwd[2],
            r0[0] * fwd[1] - r0[1] * fwd[0],
        ];

        // Roll around forward axis
        let roll = elapsed / CAMERA_ROLL_PERIOD * 2.0 * PI;
        let (sr, cr) = (roll.sin(), roll.cos());
        let right = [r0[0] * cr + u0[0] * sr, r0[1] * cr + u0[1] * sr, r0[2] * cr + u0[2] * sr];
        let up = [-r0[0] * sr + u0[0] * cr, -r0[1] * sr + u0[1] * cr, -r0[2] * sr + u0[2] * cr];

        let aspect = sw as f32 / (sh as f32 * CHAR_ASPECT);
        let half_w = (CAMERA_FOV_DEG.to_radians() * 0.5).tan();
        let half_h = half_w / aspect;

        Camera { pos, ground, fwd, right, up, half_w, half_h }
    }

    /// Perspective ray direction for screen-space (u,v) in [0,1]².
    #[inline(always)]
    fn ray(&self, u: f32, v: f32) -> [f32; 3] {
        let su = (u - 0.5) * 2.0 * self.half_w;
        let sv = (0.5 - v) * 2.0 * self.half_h; // screen top = up
        norm3([
            self.fwd[0] + su * self.right[0] + sv * self.up[0],
            self.fwd[1] + su * self.right[1] + sv * self.up[1],
            self.fwd[2] + su * self.right[2] + sv * self.up[2],
        ])
    }

    /// Inverse of `ray`: project a direction back to screen-space (u,v).
    #[inline(always)]
    fn project_dir(&self, d: [f32; 3]) -> Option<(f32, f32)> {
        let sf = dot3(d, self.fwd);
        if sf <= 0.0 { return None; }
        let u = 0.5 + dot3(d, self.right) / (sf * 2.0 * self.half_w);
        let v = 0.5 - dot3(d, self.up)    / (sf * 2.0 * self.half_h);
        if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 { return None; }
        Some((u, v))
    }

    /// Intersect ray with the z=0 water plane; returns normalised (wx,wy) or None.
    #[inline(always)]
    fn hit_water(&self, dir: [f32; 3]) -> Option<[f32; 2]> {
        if dir[2] >= -1e-6 { return None; }
        let t = -self.pos[2] / dir[2];
        let wx = (self.pos[0] + t * dir[0] - self.ground[0]) / WATER_SCALE + 0.5;
        let wy = (self.pos[1] + t * dir[1] - self.ground[1]) / WATER_SCALE + 0.5;
        if wx < 0.0 || wx >= 1.0 || wy < 0.0 || wy >= 1.0 { return None; }
        Some([wx, wy])
    }
}

// ── bilinear sampling ───────────────────────────────────────────────

#[inline(always)]
fn sample(levels: &Array2<f32>, wx: f32, wy: f32) -> f32 {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    let fx = (wx * gw as f32).clamp(0.0, gw as f32 - 1.001);
    let fy = (wy * gh as f32).clamp(0.0, gh as f32 - 1.001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(gw - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let a = levels[(x0, y0)] + (levels[(x1, y0)] - levels[(x0, y0)]) * tx;
    let b = levels[(x0, y1)] + (levels[(x1, y1)] - levels[(x0, y1)]) * tx;
    a + (b - a) * ty
}

// ── precomputed grids (Scharr normals + Laplacian caustics) ─────────
// Computed once per frame from the level grid using direct neighbor access
// (no bilinear sampling). Shading then bilinearly samples these grids,
// replacing ~40 bilinear level lookups per shade_water call with ~5.

/// Bilinear-sample a flat [f32; 4] grid (layout: x * gh + y).
#[inline(always)]
fn sample_grid4(grid: &[[f32; 4]], gw: usize, gh: usize, wx: f32, wy: f32) -> [f32; 4] {
    let fx = (wx * gw as f32).clamp(0.0, gw as f32 - 1.001);
    let fy = (wy * gh as f32).clamp(0.0, gh as f32 - 1.001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let i00 = x0 * gh + y0;
    let i10 = x1 * gh + y0;
    let i01 = x0 * gh + y1;
    let i11 = x1 * gh + y1;
    // Safety: fx/fy clamped to [0, dim-1.001) so x1/y1 < dim
    unsafe {
        let g00 = grid.get_unchecked(i00);
        let g10 = grid.get_unchecked(i10);
        let g01 = grid.get_unchecked(i01);
        let g11 = grid.get_unchecked(i11);
        let mut r = [0.0_f32; 4];
        for c in 0..4 {
            let a = g00[c] + (g10[c] - g00[c]) * tx;
            let b = g01[c] + (g11[c] - g01[c]) * tx;
            r[c] = a + (b - a) * ty;
        }
        r
    }
}

/// Bilinear-sample a flat f32 grid (layout: x * gh + y).
#[inline(always)]
fn sample_grid1(grid: &[f32], gw: usize, gh: usize, wx: f32, wy: f32) -> f32 {
    let fx = (wx * gw as f32).clamp(0.0, gw as f32 - 1.001);
    let fy = (wy * gh as f32).clamp(0.0, gh as f32 - 1.001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    unsafe {
        let a = grid.get_unchecked(x0 * gh + y0)
              + (grid.get_unchecked(x1 * gh + y0) - grid.get_unchecked(x0 * gh + y0)) * tx;
        let b = grid.get_unchecked(x0 * gh + y1)
              + (grid.get_unchecked(x1 * gh + y1) - grid.get_unchecked(x0 * gh + y1)) * tx;
        a + (b - a) * ty
    }
}

/// Precompute Scharr normals + slope variance for the entire level grid.
/// Output layout: flat Vec, index = x * gh + y. Each element = [nx, ny, nz, slope_var].
fn precompute_normals(levels: &Array2<f32>, out: &mut Vec<[f32; 4]>) {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    out.resize(gw * gh, [0.0; 4]);
    for x in 0..gw {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(gw - 1);
        for y in 0..gh {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(gh - 1);
            // Scharr 3×3: direct neighbor access (no bilinear needed)
            unsafe {
                let tl = *levels.uget((xm, ym));
                let ml = *levels.uget((xm, y));
                let bl = *levels.uget((xm, yp));
                let tm = *levels.uget((x, ym));
                let bm = *levels.uget((x, yp));
                let tr = *levels.uget((xp, ym));
                let mr = *levels.uget((xp, y));
                let br = *levels.uget((xp, yp));

                let ddx = (tr * 3.0 + mr * 10.0 + br * 3.0
                         - tl * 3.0 - ml * 10.0 - bl * 3.0) * (1.0 / 16.0) * NORMAL_STRENGTH;
                let ddy = (bl * 3.0 + bm * 10.0 + br * 3.0
                         - tl * 3.0 - tm * 10.0 - tr * 3.0) * (1.0 / 16.0) * NORMAL_STRENGTH;
                let sv = ddx * ddx + ddy * ddy;
                let il = 1.0 / (sv + 1.0).sqrt();
                *out.get_unchecked_mut(x * gh + y) = [-ddx * il, -ddy * il, il, sv];
            }
        }
    }
}

/// Precompute Laplacian of the level grid (for caustics).
/// Negative Laplacian = concave = light convergence.
fn precompute_laplacian(levels: &Array2<f32>, out: &mut Vec<f32>) {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    out.resize(gw * gh, 0.0);
    for x in 0..gw {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(gw - 1);
        for y in 0..gh {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(gh - 1);
            unsafe {
                let c = *levels.uget((x, y));
                let lap = *levels.uget((xp, y)) + *levels.uget((xm, y))
                        + *levels.uget((x, yp)) + *levels.uget((x, ym)) - 4.0 * c;
                *out.get_unchecked_mut(x * gh + y) = lap;
            }
        }
    }
}

/// Sample caustic intensity from precomputed Laplacian grid with chromatic dispersion.
#[inline(always)]
fn caustic_from_grid(
    lap_grid: &[f32], gw: usize, gh: usize,
    wx: f32, wy: f32, n: [f32; 3],
) -> [f32; 3] {
    let d = CAUSTIC_DISPERSION;
    let lap_g = sample_grid1(lap_grid, gw, gh, wx, wy);
    let lap_r = sample_grid1(lap_grid, gw, gh,
        (wx + n[0] * d).clamp(0.001, 0.999),
        (wy + n[1] * d).clamp(0.001, 0.999));
    let lap_b = sample_grid1(lap_grid, gw, gh,
        (wx - n[0] * d).clamp(0.001, 0.999),
        (wy - n[1] * d).clamp(0.001, 0.999));
    let fr = (-lap_r * CAUSTIC_STRENGTH).max(0.0);
    let fg = (-lap_g * CAUSTIC_STRENGTH).max(0.0);
    let fb = (-lap_b * CAUSTIC_STRENGTH).max(0.0);
    [fr * fr, fg * fg, fb * fb]
}

// ── water shading with secondary bounce ─────────────────────────────

/// Shade a water hit point — energy-conserving microfacet BRDF.
///
/// Rendering equation decomposition:
///   L_o = env × F(θ_v)                                — environment reflection
///       + D·G·F(θ_h) / (4·n·v)                        — Cook-Torrance direct specular
///       + (1-F(θ_l))·(1-F(θ_v))·c·n·l·DIFFUSE_K      — direct diffuse
///       + (1-F(θ_l))·(1-F(θ_v))·c·caustic·n·l        — caustics
///       + (1-F(θ_v))·c·AMBIENT                        — indirect diffuse
///       + (1-F(θ_v))·c·sss                            — subsurface scattering
fn shade_water(
    levels: &Array2<f32>,
    normal_grid: &[[f32; 4]],
    lap_grid: &[f32],
    sky_env: &[(f32, f32, f32)],
    gw: usize, gh: usize,
    light: [f32; 3],
    cam_ground: [f32; 2],
    dir: [f32; 3],
    wp: [f32; 2],
) -> [f32; 3] {
    let level = sample(levels, wp[0], wp[1]);
    let ng = sample_grid4(normal_grid, gw, gh, wp[0], wp[1]);
    let n = [ng[0], ng[1], ng[2]];
    let slope_var = ng[3];

    let view = [-dir[0], -dir[1], -dir[2]];
    let n_dot_v = dot3(n, view).max(0.001);

    // Fresnel at view angle (env reflection / transmission split)
    let fr_v = fresnel_rgb(n_dot_v);

    let refl_dir = [
        dir[0] + 2.0 * n_dot_v * n[0],
        dir[1] + 2.0 * n_dot_v * n[1],
        dir[2] + 2.0 * n_dot_v * n[2],
    ];

    // ── Environment reflection (sky or secondary water bounce) ──
    let env = if refl_dir[2] < -1e-6 {
        let world_x = (wp[0] - 0.5) * WATER_SCALE + cam_ground[0];
        let world_y = (wp[1] - 0.5) * WATER_SCALE + cam_ground[1];
        let hit_z: f32 = 0.02;
        let t2 = hit_z / (-refl_dir[2]);
        let wx2 = (world_x + refl_dir[0] * t2 - cam_ground[0]) / WATER_SCALE + 0.5;
        let wy2 = (world_y + refl_dir[1] * t2 - cam_ground[1]) / WATER_SCALE + 0.5;

        if wx2 >= 0.0 && wx2 < 1.0 && wy2 >= 0.0 && wy2 < 1.0 {
            let level2 = sample(levels, wx2, wy2);
            let ng2 = sample_grid4(normal_grid, gw, gh, wx2, wy2);
            let n2 = [ng2[0], ng2[1], ng2[2]];
            let n_dot_l2 = dot3(n2, light).max(0.0);
            let wb2 = water_body_color(level2);

            let view2 = norm3([-refl_dir[0], -refl_dir[1], -refl_dir[2]]);
            let cos_t2 = dot3(n2, view2).max(0.0);
            let fr_v2 = fresnel_scalar(cos_t2);
            let fr_l2 = fresnel_scalar(n_dot_l2);

            let refl2 = [
                refl_dir[0] + 2.0 * cos_t2 * n2[0],
                refl_dir[1] + 2.0 * cos_t2 * n2[1],
                refl_dir[2] + 2.0 * cos_t2 * n2[2],
            ];
            let sky2 = sample_sky(sky_env, refl2);

            // Energy-conserving secondary bounce
            let tx2 = (1.0 - fr_v2) * ((1.0 - fr_l2) * n_dot_l2 * DIFFUSE_K + AMBIENT);
            (
                wb2.0 * tx2 + sky2.0 * fr_v2,
                wb2.1 * tx2 + sky2.1 * fr_v2,
                wb2.2 * tx2 + sky2.2 * fr_v2,
            )
        } else {
            sample_sky(sky_env, refl_dir)
        }
    } else {
        sample_sky(sky_env, refl_dir)
    };

    // ── Direct lighting ──
    let n_dot_l = dot3(n, light).max(0.0);
    let h = norm3([light[0] + view[0], light[1] + view[1], light[2] + view[2]]);
    let n_dot_h = dot3(n, h).max(0.0);
    let v_dot_h = dot3(view, h).max(0.0);

    // GGX NDF with slope-variance filtering
    let a2 = (GGX_ROUGHNESS * GGX_ROUGHNESS + slope_var * NDF_FILTER_K).min(1.0);
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (PI * denom * denom);

    // Smith-GGX height-correlated geometry
    let k = a2 * 0.5;
    let g1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let geom = g1v * g1l;

    // Fresnel at half-vector angle (correct for microfacet specular)
    let fr_h = fresnel_rgb(v_dot_h);

    // Cook-Torrance: n·l cancels with rendering equation denominator
    let spec_base = (d_ggx * geom / (4.0 * n_dot_v)).min(1.0);
    let direct_spec = [
        fr_h[0] * spec_base,
        fr_h[1] * spec_base,
        fr_h[2] * spec_base,
    ];

    // Fresnel at light angle (diffuse energy conservation)
    let fr_l = fresnel_rgb(n_dot_l);

    // ── Transmitted component ──
    let wb = water_body_color(level);
    let caustic = caustic_from_grid(lap_grid, gw, gh, wp[0], wp[1], n);

    // Direct diffuse: (1-F_l)(1-F_v) — light enters, scatters, exits
    let direct_diff = [
        (1.0 - fr_l[0]) * (1.0 - fr_v[0]) * wb.0 * n_dot_l * DIFFUSE_K,
        (1.0 - fr_l[1]) * (1.0 - fr_v[1]) * wb.1 * n_dot_l * DIFFUSE_K,
        (1.0 - fr_l[2]) * (1.0 - fr_v[2]) * wb.2 * n_dot_l * DIFFUSE_K,
    ];

    // Caustics: refracted sunlight focused by surface curvature
    let caustic_term = [
        (1.0 - fr_l[0]) * (1.0 - fr_v[0]) * wb.0 * caustic[0] * n_dot_l,
        (1.0 - fr_l[1]) * (1.0 - fr_v[1]) * wb.1 * caustic[1] * n_dot_l,
        (1.0 - fr_l[2]) * (1.0 - fr_v[2]) * wb.2 * caustic[2] * n_dot_l,
    ];

    // Ambient (indirect hemisphere irradiance)
    let ambient_tx = [
        (1.0 - fr_v[0]) * wb.0 * AMBIENT,
        (1.0 - fr_v[1]) * wb.1 * AMBIENT,
        (1.0 - fr_v[2]) * wb.2 * AMBIENT,
    ];

    // SSS: forward-scattered transmitted light
    let sss_dir = norm3([
        -light[0] + n[0] * SSS_DISTORTION,
        -light[1] + n[1] * SSS_DISTORTION,
        -light[2] + n[2] * SSS_DISTORTION,
    ]);
    let sss_dot = dot3(view, sss_dir).max(0.0).powf(SSS_POWER);
    let sss_val = sss_dot * SSS_STRENGTH;
    let sss_term = [
        (1.0 - fr_v[0]) * wb.0 * sss_val,
        (1.0 - fr_v[1]) * wb.1 * sss_val,
        (1.0 - fr_v[2]) * wb.2 * sss_val,
    ];

    // ── Combine ──
    // env × F_v + specular (F_h inside) + diffuse + caustics + ambient + SSS
    [
        env.0 * fr_v[0] + direct_spec[0] + direct_diff[0] + caustic_term[0] + ambient_tx[0] + sss_term[0],
        env.1 * fr_v[1] + direct_spec[1] + direct_diff[1] + caustic_term[1] + ambient_tx[1] + sss_term[1],
        env.2 * fr_v[2] + direct_spec[2] + direct_diff[2] + caustic_term[2] + ambient_tx[2] + sss_term[2],
    ]
}

/// Cast a single ray at (u, v), shade water/sky, apply aerial fog.
#[inline(always)]
fn shade_ray(
    cam: &Camera,
    levels: &Array2<f32>,
    normal_grid: &[[f32; 4]],
    lap_grid: &[f32],
    sky_env: &[(f32, f32, f32)],
    gw: usize, gh: usize,
    light: [f32; 3],
    u: f32,
    v: f32,
) -> [f32; 3] {
    let dir = cam.ray(u, v);

    if let Some(wp) = cam.hit_water(dir) {
        let water = shade_water(levels, normal_grid, lap_grid, sky_env, gw, gh,
                                light, cam.ground, dir, wp);
        if dir[2] < -1e-6 {
            let t = -cam.pos[2] / dir[2];
            let fog = 1.0 - (-t * FOG_DENSITY).exp();
            let hz = sample_sky(sky_env, norm3([dir[0], dir[1], 0.001]));
            [
                water[0] + (hz.0 - water[0]) * fog,
                water[1] + (hz.1 - water[1]) * fog,
                water[2] + (hz.2 - water[2]) * fog,
            ]
        } else {
            water
        }
    } else {
        let s = sample_sky(sky_env, dir);
        [s.0, s.1, s.2]
    }
}

// ── bilinear HDR buffer sampling (for TAA reprojection) ─────────────

#[inline(always)]
fn sample_hdr(buf: &[[f32; 3]], w: usize, vh: usize, u: f32, v: f32) -> [f32; 3] {
    let fx = (u * w as f32).clamp(0.0, w as f32 - 1.001);
    let fy = (v * vh as f32).clamp(0.0, vh as f32 - 1.001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(vh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut r = [0.0_f32; 3];
    for c in 0..3 {
        let a = buf[y0 * w + x0][c] * (1.0 - tx) + buf[y0 * w + x1][c] * tx;
        let b = buf[y1 * w + x0][c] * (1.0 - tx) + buf[y1 * w + x1][c] * tx;
        r[c] = a * (1.0 - ty) + b * ty;
    }
    r
}

// ── color grading ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct ColorGrade {
    name: &'static str,
    exposure: f32,
    contrast: f32,
    saturation: f32,
    gain: [f32; 3],
}

const PRESETS: &[ColorGrade] = &[
    ColorGrade { name: "default",   exposure: 1.0,  contrast: 1.0,  saturation: 1.0,  gain: [1.0, 1.0, 1.0] },
    ColorGrade { name: "storm",     exposure: 0.85, contrast: 1.25, saturation: 0.55, gain: [0.82, 0.9, 1.08] },
    ColorGrade { name: "neon",      exposure: 1.05, contrast: 1.3,  saturation: 1.5,  gain: [1.05, 0.85, 1.15] },
    ColorGrade { name: "moonlight", exposure: 0.55, contrast: 0.85, saturation: 0.25, gain: [0.7, 0.82, 1.3] },
];

/// Tonemap + color grade → sRGB [0,1] (ready for dither + quantize).
#[inline(always)]
fn apply_grade(hdr: [f32; 3], g: &ColorGrade) -> [f32; 3] {
    let mut r = aces_tonemap(hdr[0] * g.exposure);
    let mut gv = aces_tonemap(hdr[1] * g.exposure);
    let mut b = aces_tonemap(hdr[2] * g.exposure);

    // contrast (pivot 0.5)
    r = 0.5 + (r - 0.5) * g.contrast;
    gv = 0.5 + (gv - 0.5) * g.contrast;
    b = 0.5 + (b - 0.5) * g.contrast;

    // saturation
    let luma = r * 0.2126 + gv * 0.7152 + b * 0.0722;
    r = luma + (r - luma) * g.saturation;
    gv = luma + (gv - luma) * g.saturation;
    b = luma + (b - luma) * g.saturation;

    // tint
    [
        linear_to_srgb((r * g.gain[0]).clamp(0.0, 1.0)),
        linear_to_srgb((gv * g.gain[1]).clamp(0.0, 1.0)),
        linear_to_srgb((b * g.gain[2]).clamp(0.0, 1.0)),
    ]
}

// ── renderer ────────────────────────────────────────────────────────

pub struct Renderer {
    prev_colors: Vec<((u8, u8, u8), (u8, u8, u8))>,
    prev_hdr_buf: Vec<[f32; 3]>,  // previous frame post-bloom HDR (row-major, w × vh)
    prev_cam: Option<Camera>,
    hdr_buf: Vec<[f32; 3]>,       // w × vh virtual pixels (row-major)
    bloom_a: Vec<[f32; 3]>,       // bloom scratch
    bloom_b: Vec<[f32; 3]>,       // bloom scratch
    normal_grid: Vec<[f32; 4]>,   // precomputed Scharr normals [nx,ny,nz,slope_var]
    lap_grid: Vec<f32>,           // precomputed Laplacian (for caustics)
    sky_envmap: Vec<(f32, f32, f32)>, // octahedral sky envmap (SKY_ENV_SIZE²)
    preset_idx: usize,
    frame: u32,
    width: u16,
    height: u16,
    buf: Vec<u8>,
    start: Instant,
}

impl Renderer {
    pub fn new(width: u16, height: u16) -> Self {
        let n = width as usize * height as usize;
        let vn = width as usize * height as usize * 2;
        Self {
            prev_colors: vec![((0, 0, 0), (0, 0, 0)); n],
            prev_hdr_buf: vec![[0.0; 3]; vn],
            prev_cam: None,
            hdr_buf: vec![[0.0; 3]; vn],
            bloom_a: vec![[0.0; 3]; vn],
            bloom_b: vec![[0.0; 3]; vn],
            normal_grid: vec![[0.0; 4]; n],
            lap_grid: vec![0.0; n],
            sky_envmap: vec![(0.0, 0.0, 0.0); SKY_ENV_SIZE * SKY_ENV_SIZE],
            preset_idx: 0,
            frame: 0,
            width,
            height,
            buf: Vec::with_capacity(65536),
            start: Instant::now(),
        }
    }

    pub fn cycle_preset(&mut self) {
        self.preset_idx = (self.preset_idx + 1) % PRESETS.len();
    }

    pub fn preset_name(&self) -> &'static str {
        PRESETS[self.preset_idx].name
    }

    /// Blend current virtual pixel with reprojected previous frame.
    fn reproject_blend(
        &self, cur: [f32; 3], cam: &Camera,
        sx: f32, sy_v: f32, w: f32, vh: f32, w_us: usize, vh_us: usize,
    ) -> [f32; 3] {
        let prev = if let Some(ref pc) = self.prev_cam {
            let uc = (sx + 0.5) / w;
            let vc = (sy_v + 0.5) / vh;
            let dir = cam.ray(uc, vc);

            // Water hit → reproject world point; sky → reproject direction
            let repr_dir = if dir[2] < -1e-6 {
                let t = -cam.pos[2] / dir[2];
                [cam.pos[0] + t * dir[0] - pc.pos[0],
                 cam.pos[1] + t * dir[1] - pc.pos[1],
                 -pc.pos[2]]
            } else {
                dir
            };

            match pc.project_dir(repr_dir) {
                Some((pu, pv)) => sample_hdr(&self.prev_hdr_buf, w_us, vh_us, pu, pv),
                None => cur,
            }
        } else {
            cur
        };
        [
            prev[0] * TAA_KEEP + cur[0] * TAA_NEW,
            prev[1] * TAA_KEEP + cur[1] * TAA_NEW,
            prev[2] * TAA_KEEP + cur[2] * TAA_NEW,
        ]
    }

    pub fn draw(&mut self, stdout: &mut Stdout, water: &Water) -> Result<()> {
        let w = water.width();
        let h = water.height();
        let vh = h * 2; // virtual height: two virtual pixel rows per terminal row

        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            let n = w as usize * h as usize;
            let vn = w as usize * h as usize * 2;
            self.prev_colors = vec![((0, 0, 0), (0, 0, 0)); n];
            self.prev_hdr_buf = vec![[0.0; 3]; vn];
            self.prev_cam = None;
            self.hdr_buf = vec![[0.0; 3]; vn];
            self.bloom_a = vec![[0.0; 3]; vn];
            self.bloom_b = vec![[0.0; 3]; vn];
            queue!(stdout, terminal::Clear(terminal::ClearType::All))?;
            stdout.flush()?;
        }

        let elapsed = self.start.elapsed().as_secs_f32();
        let cam = Camera::new(elapsed, w, h);

        // Rotating directional light
        let la = elapsed * 2.0 * PI / LIGHT_PERIOD;
        let light = {
            let lx = la.cos() * LIGHT_RADIUS;
            let ly = la.sin() * LIGHT_RADIUS;
            let lz = LIGHT_ELEVATION;
            let l = (lx * lx + ly * ly + lz * lz).sqrt();
            [lx / l, ly / l, lz / l]
        };

        self.buf.clear();
        let levels = &water.levels;
        let w_f = w as f32;
        let vh_f = vh as f32;
        let w_us = w as usize;
        let vh_us = vh as usize;

        // ── precompute grids (once per frame) ──
        let gw = levels.shape()[0];
        let gh = levels.shape()[1];
        precompute_normals(levels, &mut self.normal_grid);
        precompute_laplacian(levels, &mut self.lap_grid);
        precompute_sky(elapsed, &mut self.sky_envmap);

        // ── pass 1: shade into HDR buffer (4× RGSS, no CA yet) ──
        // RGSS offsets (rotated grid for best coverage)
        const RGSS: [(f32, f32); 4] = [
            (-3.0/8.0,  1.0/8.0),
            ( 1.0/8.0,  3.0/8.0),
            ( 3.0/8.0, -1.0/8.0),
            (-1.0/8.0, -3.0/8.0),
        ];
        for sy in 0..h {
            for sx in 0..w {
                let sy2 = sy as usize * 2;
                let sx_us = sx as usize;

                for (vi, vrow) in [(sy2, 0.0_f32), (sy2 + 1, 1.0_f32)] {
                    let vc_center = (sy as f32 * 2.0 + vrow + 0.5) / vh_f;
                    let uc_center = (sx as f32 + 0.5) / w_f;

                    let mut accum = [0.0_f32; 3];

                    for &(ox, oy) in &RGSS {
                        let uc = uc_center + ox / w_f;
                        let vc = vc_center + oy / vh_f;
                        let c = shade_ray(&cam, levels,
                            &self.normal_grid, &self.lap_grid, &self.sky_envmap,
                            gw, gh, light, uc, vc);
                        accum[0] += c[0];
                        accum[1] += c[1];
                        accum[2] += c[2];
                    }

                    self.hdr_buf[vi * w_us + sx_us] = [
                        accum[0] * 0.25,
                        accum[1] * 0.25,
                        accum[2] * 0.25,
                    ];
                }
            }
        }

        // ── pass 1b: chromatic aberration (post-pass on HDR buffer) ──
        // Use bloom_b as temp buffer to avoid read/write conflict
        for y in 0..vh_us {
            for x in 0..w_us {
                let uc = (x as f32 + 0.5) / w_f;
                let vc = (y as f32 + 0.5) / vh_f;
                let du = uc - 0.5;
                let dv = vc - 0.5;
                let r2 = du * du + dv * dv;
                let shift = r2 * CA_STRENGTH;

                let center = self.hdr_buf[y * w_us + x];

                // R: sample outward-shifted, B: sample inward-shifted
                let ru = (uc + du * shift).clamp(0.0, 0.999);
                let rv = (vc + dv * shift).clamp(0.0, 0.999);
                let bu = (uc - du * shift).clamp(0.0, 0.999);
                let bv = (vc - dv * shift).clamp(0.0, 0.999);

                let r_sample = sample_hdr(&self.hdr_buf, w_us, vh_us, ru, rv);
                let b_sample = sample_hdr(&self.hdr_buf, w_us, vh_us, bu, bv);

                self.bloom_b[y * w_us + x] = [r_sample[0], center[1], b_sample[2]];
            }
        }
        // Copy CA result back to hdr_buf
        let total = w_us * vh_us;
        self.hdr_buf[..total].copy_from_slice(&self.bloom_b[..total]);

        // ── pass 2: bloom — extract bright, separable gaussian blur, add back ──
        for i in 0..total {
            let h = self.hdr_buf[i];
            self.bloom_a[i] = [
                (h[0] - BLOOM_THRESHOLD).max(0.0),
                (h[1] - BLOOM_THRESHOLD).max(0.0),
                (h[2] - BLOOM_THRESHOLD).max(0.0),
            ];
        }
        // ── pass 2b: god rays — radial blur of bright extraction toward sun ──
        if let Some((sun_u, sun_v)) = cam.project_dir(light) {
            for y in 0..vh_us {
                for x in 0..w_us {
                    let px = (x as f32 + 0.5) / w_f;
                    let py = (y as f32 + 0.5) / vh_f;
                    let dx = (px - sun_u) * GOD_RAY_DENSITY / GOD_RAY_SAMPLES as f32;
                    let dy = (py - sun_v) * GOD_RAY_DENSITY / GOD_RAY_SAMPLES as f32;
                    let mut su = px;
                    let mut sv = py;
                    let mut acc = [0.0_f32; 3];
                    let mut wt = 1.0_f32;
                    for _ in 0..GOD_RAY_SAMPLES {
                        su -= dx;
                        sv -= dy;
                        if su >= 0.0 && su < 1.0 && sv >= 0.0 && sv < 1.0 {
                            let si = (sv * vh_f) as usize * w_us
                                   + (su * w_f) as usize;
                            let p = self.bloom_a[si.min(total - 1)];
                            acc[0] += p[0] * wt;
                            acc[1] += p[1] * wt;
                            acc[2] += p[2] * wt;
                        }
                        wt *= GOD_RAY_DECAY;
                    }
                    let inv = GOD_RAY_EXPOSURE / GOD_RAY_SAMPLES as f32;
                    self.bloom_b[y * w_us + x] = [
                        acc[0] * inv, acc[1] * inv, acc[2] * inv,
                    ];
                }
            }
            for i in 0..total {
                let g = self.bloom_b[i];
                self.hdr_buf[i][0] += g[0];
                self.hdr_buf[i][1] += g[1];
                self.hdr_buf[i][2] += g[2];
            }
        }

        // horizontal: bloom_a → bloom_b
        for y in 0..vh_us {
            let row = y * w_us;
            for x in 0..w_us {
                let mut s = [0.0_f32; 3];
                for (ki, &kw) in BLOOM_KERNEL.iter().enumerate() {
                    let nx = (x as i32 + ki as i32 - BLOOM_RADIUS)
                        .max(0).min(w_us as i32 - 1) as usize;
                    let p = self.bloom_a[row + nx];
                    s[0] += p[0] * kw; s[1] += p[1] * kw; s[2] += p[2] * kw;
                }
                self.bloom_b[row + x] = s;
            }
        }
        // vertical: bloom_b → bloom_a
        for y in 0..vh_us {
            for x in 0..w_us {
                let mut s = [0.0_f32; 3];
                for (ki, &kw) in BLOOM_KERNEL.iter().enumerate() {
                    let ny = (y as i32 + ki as i32 - BLOOM_RADIUS)
                        .max(0).min(vh_us as i32 - 1) as usize;
                    let p = self.bloom_b[ny * w_us + x];
                    s[0] += p[0] * kw; s[1] += p[1] * kw; s[2] += p[2] * kw;
                }
                self.bloom_a[y * w_us + x] = s;
            }
        }
        // add bloom back
        for i in 0..total {
            let b = self.bloom_a[i];
            self.hdr_buf[i][0] += b[0] * BLOOM_INTENSITY;
            self.hdr_buf[i][1] += b[1] * BLOOM_INTENSITY;
            self.hdr_buf[i][2] += b[2] * BLOOM_INTENSITY;
        }

        // ── pass 3: reprojected TAA + grade + dither + output ──
        let grade = &PRESETS[self.preset_idx];
        let frame_f = self.frame as f32;
        self.frame = self.frame.wrapping_add(1);
        for sx in 0..w {
            for sy in 0..h {
                let sy2 = sy as usize * 2;
                let sx_us = sx as usize;
                let cur_top = self.hdr_buf[sy2 * w_us + sx_us];
                let cur_bot = self.hdr_buf[(sy2 + 1) * w_us + sx_us];

                // Reprojected TAA for each virtual half
                let top_hdr = self.reproject_blend(
                    cur_top, &cam, sx as f32, (sy * 2) as f32, w_f, vh_f, w_us, vh_us);
                let bot_hdr = self.reproject_blend(
                    cur_bot, &cam, sx as f32, (sy * 2 + 1) as f32, w_f, vh_f, w_us, vh_us);

                let ts = apply_grade(top_hdr, grade);
                let bs = apply_grade(bot_hdr, grade);
                let dt = ign(sx as f32, (sy * 2) as f32, frame_f);
                let db = ign(sx as f32, (sy * 2 + 1) as f32, frame_f);

                let top = (
                    (ts[0] * 255.0 + dt).clamp(0.0, 255.0) as u8,
                    (ts[1] * 255.0 + dt).clamp(0.0, 255.0) as u8,
                    (ts[2] * 255.0 + dt).clamp(0.0, 255.0) as u8,
                );
                let bottom = (
                    (bs[0] * 255.0 + db).clamp(0.0, 255.0) as u8,
                    (bs[1] * 255.0 + db).clamp(0.0, 255.0) as u8,
                    (bs[2] * 255.0 + db).clamp(0.0, 255.0) as u8,
                );

                let idx = sx_us * h as usize + sy as usize;
                let pair = (top, bottom);
                if self.prev_colors[idx] != pair {
                    self.prev_colors[idx] = pair;
                    queue!(self.buf, cursor::MoveTo(sx, sy))?;
                    queue!(self.buf, style::SetForegroundColor(Color::from(top)))?;
                    queue!(self.buf, style::SetBackgroundColor(Color::from(bottom)))?;
                    queue!(self.buf, style::Print("▀"))?;
                }
            }
        }

        // Save current frame for next TAA pass
        self.prev_cam = Some(cam);
        std::mem::swap(&mut self.hdr_buf, &mut self.prev_hdr_buf);

        if !self.buf.is_empty() {
            // Synchronized output: terminal buffers everything between these markers
            stdout.write_all(b"\x1b[?2026h")?;
            stdout.write_all(&self.buf)?;
            stdout.write_all(b"\x1b[?2026l")?;
            stdout.flush()?;
        }

        Ok(())
    }
}
