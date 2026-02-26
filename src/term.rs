use crate::colors::{fresnel, sky_color, water_body_color};
use anyhow::Result;
use crossterm::style::Color;
use crossterm::{cursor, queue, style, terminal};
use ndarray::Array2;
use std::f32::consts::PI;
use std::io::{Stdout, Write};
use std::time::Instant;
use crate::water::Water;

pub fn clear(w: &mut Stdout) -> Result<()> {
    queue!(w, terminal::Clear(terminal::ClearType::All))?;
    Ok(())
}

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

const NORMAL_STRENGTH: f32 = 15.0;
const AMBIENT: f32 = 0.3;
const DIFFUSE_K: f32 = 0.6;
const SHININESS: i32 = 32;

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

// ── camera ──────────────────────────────────────────────────────────

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

#[inline(always)]
fn sample_normal(levels: &Array2<f32>, wx: f32, wy: f32) -> [f32; 3] {
    let ex = 1.0 / levels.shape()[0] as f32;
    let ey = 1.0 / levels.shape()[1] as f32;
    let ddx = (sample(levels, (wx + ex).min(0.999), wy)
             - sample(levels, (wx - ex).max(0.0), wy)) * NORMAL_STRENGTH;
    let ddy = (sample(levels, wx, (wy + ey).min(0.999))
             - sample(levels, wx, (wy - ey).max(0.0))) * NORMAL_STRENGTH;
    let il = 1.0 / (ddx * ddx + ddy * ddy + 1.0).sqrt();
    [-ddx * il, -ddy * il, il]
}

// ── renderer ────────────────────────────────────────────────────────

pub struct Renderer {
    prev_colors: Vec<(u8, u8, u8)>,
    width: u16,
    height: u16,
    buf: Vec<u8>,
    start: Instant,
}

impl Renderer {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            prev_colors: vec![(0, 0, 0); width as usize * height as usize],
            width,
            height,
            buf: Vec::with_capacity(65536),
            start: Instant::now(),
        }
    }

    pub fn draw(&mut self, stdout: &mut Stdout, water: &Water) -> Result<()> {
        let w = water.width();
        let h = water.height();

        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.prev_colors = vec![(0, 0, 0); w as usize * h as usize];
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

        for sx in 0..w {
            for sy in 0..h {
                let u = (sx as f32 + 0.5) / w as f32;
                let v = (sy as f32 + 0.5) / h as f32;
                let dir = cam.ray(u, v);

                let color = if let Some(wp) = cam.hit_water(dir) {
                    let level = sample(levels, wp[0], wp[1]);
                    let n = sample_normal(levels, wp[0], wp[1]);

                    // view = surface → eye = −ray
                    let view = [-dir[0], -dir[1], -dir[2]];
                    let cos_t = dot3(n, view).max(0.0);

                    // Fresnel
                    let refl = fresnel(cos_t);

                    // Reflected direction for sky lookup: R = dir + 2·cosθ·n
                    let rz = dir[2] + 2.0 * cos_t * n[2];
                    let sky = sky_color(elapsed, rz.max(0.0));

                    // Diffuse
                    let diff = dot3(n, light).max(0.0);
                    let lit = AMBIENT + diff * DIFFUSE_K;

                    // Transmitted (water body seen through surface)
                    let wb = water_body_color(level);
                    let t = 1.0 - refl;
                    let tx = (wb.0 * lit * t, wb.1 * lit * t, wb.2 * lit * t);

                    // Reflected sky
                    let rf = (sky.0 * refl, sky.1 * refl, sky.2 * refl);

                    // Specular (Blinn-Phong, per-pixel half-vector)
                    let hv = norm3([light[0] + view[0], light[1] + view[1], light[2] + view[2]]);
                    let spec = dot3(n, hv).max(0.0).powi(SHININESS);

                    let r = ((tx.0 + rf.0 + spec * 0.55) * 255.0).min(255.0) as u8;
                    let g = ((tx.1 + rf.1 + spec * 0.75) * 255.0).min(255.0) as u8;
                    let b = ((tx.2 + rf.2 + spec * 1.00) * 255.0).min(255.0) as u8;
                    (r, g, b)
                } else {
                    // Sky pixel (above horizon or outside water bounds)
                    let sky = sky_color(elapsed, dir[2].max(0.0));
                    (
                        (sky.0 * 255.0) as u8,
                        (sky.1 * 255.0) as u8,
                        (sky.2 * 255.0) as u8,
                    )
                };

                let idx = sx as usize * h as usize + sy as usize;
                if self.prev_colors[idx] != color {
                    self.prev_colors[idx] = color;
                    queue!(self.buf, cursor::MoveTo(sx, sy))?;
                    queue!(self.buf, style::SetForegroundColor(Color::from(color)))?;
                    queue!(self.buf, style::Print("█"))?;
                }
            }
        }

        if !self.buf.is_empty() {
            stdout.write_all(&self.buf)?;
            stdout.flush()?;
        }

        Ok(())
    }
}
