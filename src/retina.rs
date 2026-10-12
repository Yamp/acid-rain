//! Three scales of artistic ocular persistence, in linear light. Reservoirs
//! capture brief light quickly and release it slowly, following unwarped input;
//! only their displayed echoes drift
//! within a bounded pixel radius. This avoids feedback blur and leaves water dynamics alone.
//! Current RGB channels sample independent moving coordinates before display
//! mapping. A normalized finite-aperture PSF redistributes linear HDR light.
//! Chroma adaptation preserves luminance; neutral histories cannot bleach dyes.

const LAYERS: usize = 3;
const LAYER_WEIGHTS: [f32; LAYERS] = [0.12, 0.11, 0.09];
const ATTACK_HALF_LIVES: [f32; LAYERS] = [0.025, 0.05, 0.09];
const RELEASE_HALF_LIVES: [f32; LAYERS] = [0.24, 0.72, 1.65];
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
pub const DEFAULT_RGB_SHIFT: f32 = 2.0;
pub const MAX_RGB_SHIFT: f32 = 12.0;
pub struct Retina {
    history: [Vec<[f32; 3]>; LAYERS],
    width: usize,
    height: usize,
    enabled: bool,
    valid: bool,
    rgb_shift: f32,
    subjective_intensity: f32,
    elapsed: f32,
    column_wave: Vec<f32>,
    row_wave: Vec<f32>,
    chroma_columns: Vec<[f32; 4]>,
    chroma_rows: Vec<[f32; 5]>,
    contrast: Vec<f32>,
}

#[inline]
fn smoothstep(
    low: f32,
    high: f32,
    value: f32,
) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn oscillate(
    elapsed: f32,
    rate: f64,
    phase: f64,
) -> f32 {
    (f64::from(elapsed) * rate + phase)
        .rem_euclid(std::f64::consts::TAU)
        .sin() as f32
}

#[inline]
fn history_affinity(
    now: [f32; 3],
    old: [f32; 3],
) -> f32 {
    let peak = now[0].max(now[1]).max(now[2]);
    let old_peak = old[0].max(old[1]).max(old[2]);
    let saturation = (peak - now[0].min(now[1]).min(now[2])) / peak.max(1e-6);
    let old_saturation = (old_peak - old[0].min(old[1]).min(old[2])) / old_peak.max(1e-6);
    let protection = smoothstep(0.04, 0.55, saturation)
        * smoothstep(0.005, 0.045, peak)
        * smoothstep(0.015, 0.06, old_peak / peak.max(0.001));
    let required = saturation.max(0.001);
    let retention =
        1.0 + (smoothstep(0.25 * required, 0.8 * required, old_saturation) - 1.0) * protection;
    let now_sum: f32 = now.iter().sum();
    let old_sum: f32 = old.iter().sum();
    if now_sum <= 0.0 || old_sum <= 0.0 {
        return retention;
    }
    let distance = (0..3)
        .map(|c| (now[c] / now_sum - old[c] / old_sum).abs())
        .sum::<f32>()
        * 0.5;
    let reliability = (4.0 * now_sum.min(old_sum) / (now_sum + old_sum)).min(1.0);
    let agreement = 1.0 - (distance * reliability).clamp(0.0, 1.0);
    retention * (0.05 + 0.95 * agreement * agreement)
}

#[inline]
fn channel<const N: usize>(
    pixels: &[[f32; N]],
    w: usize,
    h: usize,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let x = (u * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let y = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let upper = pixels[y0 * w + x0][c] + (pixels[y0 * w + x1][c] - pixels[y0 * w + x0][c]) * tx;
    let lower = pixels[y1 * w + x0][c] + (pixels[y1 * w + x1][c] - pixels[y1 * w + x0][c]) * tx;
    upper + (lower - upper) * ty
}

/// Reflect about pixel-cell boundaries: ...1,0 | 0,1,...,n-1 | n-1,n-2...
/// Symmetric taps then conserve finite-image channel energy even at corners.
#[inline]
fn reflected_index(
    index: isize,
    length: usize,
) -> usize {
    if index >= 0 && (index as usize) < length {
        return index as usize;
    }
    let period = (length * 2) as isize;
    let folded = index.rem_euclid(period) as usize;
    if folded < length {
        folded
    } else {
        length * 2 - 1 - folded
    }
}

#[inline]
fn reflected_axis(
    pixels: &[[f32; 3]],
    (w, h): (usize, usize),
    (x, y): (usize, usize),
    c: usize,
    offset: f32,
    vertical: bool,
) -> f32 {
    let (position, length, stride, base) = if vertical {
        (y, h, w, x)
    } else {
        (x, w, 1, y * w)
    };
    let sample = position as f32 + offset;
    let lower = sample.floor() as isize;
    let fraction = sample - lower as f32;
    let a = pixels[base + reflected_index(lower, length) * stride][c];
    let b = pixels[base + reflected_index(lower + 1, length) * stride][c];
    a + (b - a) * fraction
}

impl Retina {
    pub fn set_subjective_intensity(
        &mut self,
        intensity: f32,
    ) {
        if intensity.is_finite() {
            let intensity = intensity.clamp(0.0, 2.0);
            if intensity == 0.0 || self.subjective_intensity == 0.0 {
                self.valid = false;
            }
            self.subjective_intensity = intensity;
        }
    }

    pub fn reset_history(&mut self) { self.valid = false; }

    fn dose(
        &self,
        elapsed: f32,
        trip: f32,
    ) -> f32 {
        if !self.enabled || !trip.is_finite() || trip <= 0.0 || self.subjective_intensity <= 0.0 {
            return 0.0;
        }
        let time = if elapsed.is_finite() { elapsed } else { 0.0 };
        let envelope =
            0.72 + 0.17 * oscillate(time, 0.19, 0.7) + 0.11 * oscillate(time, 0.071, -1.2);
        self.subjective_intensity * (trip * 0.25).clamp(0.0, 2.0) * envelope
    }

    pub fn set_rgb_shift(
        &mut self,
        radius: f32,
    ) {
        if radius.is_finite() {
            self.rgb_shift = radius.clamp(0.0, MAX_RGB_SHIFT);
        }
    }

    /// Assemble RGB directly from three smoothly drifting source positions.
    /// Input is scene-linear HDR, before tone/gamut mapping. No luminance
    /// mask or color enhancement is involved, and nothing feeds HDR history.
    pub fn apply_channels(
        &self,
        pixels: &mut Vec<[f32; 3]>,
        scratch: &mut Vec<[f32; 3]>,
        size: (usize, usize),
        display: (usize, usize),
        elapsed: f32,
        intensity: f32,
    ) {
        let (w, h) = size;
        let dose = self.dose(elapsed, intensity);
        if !self.enabled
            || !intensity.is_finite()
            || intensity <= 0.0
            || self.rgb_shift == 0.0
            || dose <= 0.0
            || w == 0
            || h == 0
            || w.checked_mul(h) != Some(pixels.len())
        {
            return;
        }
        let elapsed = if elapsed.is_finite() {
            f64::from(elapsed)
        } else {
            0.0
        };
        let shifts: [(f32, f32); 3] = std::array::from_fn(|c| {
            let offset = c as f64 * std::f64::consts::TAU / 3.0;
            let angle =
                (elapsed * (0.13 + 0.009 * c as f64) + offset).rem_euclid(std::f64::consts::TAU);
            let breathing = (elapsed * 0.12 + offset)
                .rem_euclid(std::f64::consts::TAU)
                .sin();
            let radius =
                self.rgb_shift * (0.675 + 0.325 * breathing as f32) * smoothstep(0.0, 1.0, dose);
            let (sine, cosine) = angle.sin_cos();
            (
                cosine as f32 * radius / display.0.max(1) as f32,
                sine as f32 * radius / display.1.max(1) as f32,
            )
        });
        scratch.resize(pixels.len(), [0.0; 3]);
        for y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                scratch[y * w + x] = std::array::from_fn(|c| {
                    let (du, dv) = shifts[c];
                    channel(pixels, w, h, u + du, v + dv, c)
                });
            }
        }
        std::mem::swap(pixels, scratch);
    }

    /// Sparse normalized finite-aperture PSF in HDR linear light. Four symmetric
    /// aperture lobes approximate wavelength-dependent diffraction; this is not
    /// a coherent-wave solver. Their weights sum with the central core to one.
    pub fn apply_diffraction(
        &self,
        pixels: &mut Vec<[f32; 3]>,
        scratch: &mut Vec<[f32; 3]>,
        size: (usize, usize),
        display: (usize, usize),
        intensity: f32,
    ) {
        let (w, h) = size;
        if !self.enabled
            || !intensity.is_finite()
            || intensity <= 0.0
            || w == 0
            || h == 0
            || w.checked_mul(h) != Some(pixels.len())
        {
            return;
        }
        let dose = self.dose(self.elapsed, intensity);
        if dose <= 0.0 {
            return;
        }
        let lobe_weight = 0.13 * dose.min(1.7) * 0.25;
        let radii = [650.0, 550.0, 460.0].map(|wavelength| {
            let radius = 6.0 * wavelength / 550.0;
            (
                radius * w as f32 / display.0.max(1) as f32,
                radius * h as f32 / display.1.max(1) as f32,
            )
        });
        scratch.resize(pixels.len(), [0.0; 3]);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                scratch[i] = std::array::from_fn(|c| {
                    let current = pixels[i][c];
                    let (rx, ry) = radii[c];
                    let mut result = current;
                    for (offset, vertical) in [(rx, false), (-rx, false), (ry, true), (-ry, true)] {
                        let sampled = reflected_axis(pixels, size, (x, y), c, offset, vertical);
                        result += (sampled - current) * lobe_weight;
                    }
                    result
                });
            }
        }
        std::mem::swap(pixels, scratch);
    }

    pub fn new() -> Self {
        Self {
            history: std::array::from_fn(|_| Vec::new()),
            width: 0,
            height: 0,
            enabled: true,
            valid: false,
            rgb_shift: DEFAULT_RGB_SHIFT,
            subjective_intensity: 1.0,
            elapsed: 0.0,
            column_wave: Vec::new(),
            row_wave: Vec::new(),
            chroma_columns: Vec::new(),
            chroma_rows: Vec::new(),
            contrast: Vec::new(),
        }
    }

    pub fn set_enabled(
        &mut self,
        enabled: bool,
    ) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.valid = false;
        }
    }

    /// Contrast-driven artistic adaptation in scene-linear HDR. Oscillators are
    /// cached by row/column; the inner loop has no trigonometry or allocation.
    pub fn apply_chroma(
        &mut self,
        pixels: &mut [[f32; 3]],
        size: (usize, usize),
        elapsed: f32,
        intensity: f32,
    ) {
        let (w, h) = size;
        let dose = self.dose(elapsed, intensity);
        if dose <= 0.0 || w == 0 || h == 0 || w.checked_mul(h) != Some(pixels.len()) {
            return;
        }
        let t = if elapsed.is_finite() { elapsed } else { 0.0 };
        self.chroma_columns.resize(w, [0.0; 4]);
        self.chroma_rows.resize(h, [0.0; 5]);
        self.contrast.resize(pixels.len(), 0.0);
        for (x, column) in self.chroma_columns.iter_mut().enumerate() {
            let p = ((x as f32 + 0.5) / w as f32 - 0.5) * 8.0;
            let (a, b) = p.sin_cos();
            let (c, d) = (p * 2.3).sin_cos();
            *column = [a, b, c, d];
        }
        for (y, row) in self.chroma_rows.iter_mut().enumerate() {
            let p = ((y as f32 + 0.5) / h as f32 - 0.5) * 8.0;
            let (inner_sin, inner_cos) = oscillate(t, 0.11, f64::from(p * 1.37)).sin_cos();
            *row = [
                inner_sin,
                inner_cos,
                oscillate(t, -0.087, f64::from(p * 0.91) + std::f64::consts::FRAC_PI_2),
                oscillate(t, 0.061, f64::from(-p * 1.7)),
                oscillate(t, 0.061, f64::from(-p * 1.7) + std::f64::consts::FRAC_PI_2),
            ];
        }
        // Read the unmodified image first, so row order cannot accumulate blur.
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let neighbors = [
                    y * w + x.saturating_sub(1),
                    y * w + (x + 1).min(w - 1),
                    y.saturating_sub(1) * w + x,
                    (y + 1).min(h - 1) * w + x,
                ];
                let mut difference = 0.0;
                let mut magnitude = 0.0;
                for (c, &current) in pixels[i].iter().enumerate() {
                    let local = neighbors.iter().map(|n| pixels[*n][c]).sum::<f32>() * 0.25;
                    difference += (current - local).abs();
                    magnitude += current + local;
                }
                self.contrast[i] = (4.0 * difference / magnitude.max(0.05)).clamp(0.0, 1.0);
            }
        }
        let norm = LUMA.iter().map(|c| c * c).sum::<f32>().sqrt();
        let axis = LUMA.map(|c| c / norm);
        let saturation_wave = oscillate(t, 0.07, 0.0);
        for (i, color) in pixels.iter_mut().enumerate() {
            let amount = dose.min(2.0) * self.contrast[i];
            if amount <= 1e-7 {
                continue;
            }
            let luminance: f32 = (0..3).map(|c| color[c] * LUMA[c]).sum();
            if luminance <= 1e-8 {
                continue;
            }
            let x = self.chroma_columns[i % w];
            let y = self.chroma_rows[i / w];
            let field = (x[0] * y[1] + x[1] * y[0]) * y[2] + 0.35 * (x[2] * y[4] + x[3] * y[3]);
            let chroma = color.map(|c| c - luminance);
            let cross = [
                axis[1] * chroma[2] - axis[2] * chroma[1],
                axis[2] * chroma[0] - axis[0] * chroma[2],
                axis[0] * chroma[1] - axis[1] * chroma[0],
            ];
            // Cayley rotation: exact sine/cosine pair without per-pixel sin().
            let half_angle = 0.125 * amount * field;
            let inverse = 1.0 / (1.0 + half_angle * half_angle);
            let sine = 2.0 * half_angle * inverse;
            let cosine = (1.0 - half_angle * half_angle) * inverse;
            let saturation = 1.0 + 0.06 * amount * field / 1.35 * saturation_wave;
            let rotated: [f32; 3] =
                std::array::from_fn(|c| (chroma[c] * cosine + cross[c] * sine) * saturation);
            let minimum = rotated[0].min(rotated[1]).min(rotated[2]);
            let scale = if minimum < -luminance {
                luminance / (-minimum).max(1e-8)
            } else {
                1.0
            };
            *color = rotated.map(|c| (luminance + c * scale).max(0.0));
        }
    }

    pub fn apply(
        &mut self,
        pixels: &mut Vec<[f32; 3]>,
        scratch: &mut Vec<[f32; 3]>,
        size: (usize, usize),
        display: (usize, usize),
        time: (f32, f32),
        intensity: f32,
    ) {
        let (w, h) = size;
        let (elapsed, dt) = time;
        self.elapsed = if elapsed.is_finite() { elapsed } else { 0.0 };
        let dose = self.dose(self.elapsed, intensity);
        if !self.enabled
            || !intensity.is_finite()
            || intensity <= 0.0
            || dose <= 0.0
            || w == 0
            || h == 0
            || w.checked_mul(h) != Some(pixels.len())
        {
            self.valid = false;
            return;
        }
        if (self.width, self.height) != size {
            for history in &mut self.history {
                if self.valid && self.width > 0 && self.height > 0 {
                    let mut resized = vec![[0.0; 3]; pixels.len()];
                    for y in 0..h {
                        for x in 0..w {
                            resized[y * w + x] = std::array::from_fn(|c| {
                                channel(
                                    history,
                                    self.width,
                                    self.height,
                                    (x as f32 + 0.5) / w as f32,
                                    (y as f32 + 0.5) / h as f32,
                                    c,
                                )
                            });
                        }
                    }
                    *history = resized;
                } else {
                    history.resize(pixels.len(), [0.0; 3]);
                }
            }
            self.width = w;
            self.height = h;
        }
        if !self.valid {
            for history in &mut self.history {
                history.copy_from_slice(pixels);
            }
            self.valid = true;
        }
        scratch.resize(pixels.len(), [0.0; 3]);
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        // Integrate only raw, unwarped exposed HDR. A shared luminance-driven
        // attack/release clock captures brief glints without changing their hue.
        // Half-lives use wall time and remain independent of dose and cadence.
        let response = |half_life: f32| -(-std::f32::consts::LN_2 * dt / half_life).exp_m1();
        let attack_alpha = ATTACK_HALF_LIVES.map(response);
        let release_alpha = RELEASE_HALF_LIVES.map(response);
        for (layer, history) in self.history.iter_mut().enumerate() {
            for (stored, input) in history.iter_mut().zip(pixels.iter()) {
                let input_luma: f32 = input.iter().zip(LUMA).map(|(c, weight)| c * weight).sum();
                let stored_luma: f32 = stored.iter().zip(LUMA).map(|(c, weight)| c * weight).sum();
                let alpha = if input_luma > stored_luma {
                    attack_alpha[layer]
                } else {
                    release_alpha[layer]
                };
                for c in 0..3 {
                    stored[c] = if stored[c].is_finite() {
                        stored[c] + (input[c] - stored[c]) * alpha
                    } else {
                        input[c]
                    };
                }
            }
        }
        let display_w = display.0.max(1) as f32;
        let display_h = display.1.max(1) as f32;
        let amplitude = (0.5 + 0.45 * self.rgb_shift) * dose.min(2.0);
        let t = self.elapsed;
        self.column_wave.resize(w, 0.0);
        self.row_wave.resize(h, 0.0);
        for (x, value) in self.column_wave.iter_mut().enumerate() {
            *value = oscillate(
                t,
                -0.13,
                f64::from((x as f32 + 0.5) / w as f32 * 6.0) + std::f64::consts::FRAC_PI_2,
            );
        }
        for (y, value) in self.row_wave.iter_mut().enumerate() {
            *value = oscillate(t, 0.17, f64::from((y as f32 + 0.5) / h as f32 * 7.0));
        }
        // Three slowly drifting layers, with a bounded local bend. No global
        // zoom or rotation: phase calculations are O(width+height), not pixels.
        let offsets = [
            [
                0.35,
                0.35 * oscillate(t, 0.31, 0.0),
                0.35 * oscillate(t, 0.23, std::f64::consts::FRAC_PI_2),
            ],
            [
                0.60,
                0.60 * oscillate(t, 0.19, 0.9 + std::f64::consts::FRAC_PI_2),
                0.60 * oscillate(t, 0.27, 0.0),
            ],
            [
                -0.8,
                0.75 * oscillate(t, 0.11, 2.0),
                0.75 * oscillate(t, 0.17, std::f64::consts::FRAC_PI_2),
            ],
        ];
        let weights = LAYER_WEIGHTS.map(|weight| weight * dose.min(1.7));
        for y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                let i = y * w + x;
                let current = pixels[i];
                let mut result = current;
                for layer in 0..LAYERS {
                    let [bend, dx, dy] = offsets[layer];
                    let su = u + amplitude * (bend * self.row_wave[y] + dx) / display_w;
                    let sv = v + amplitude * (bend * self.column_wave[x] + dy) / display_h;
                    let echo =
                        std::array::from_fn(|c| channel(&self.history[layer], w, h, su, sv, c));
                    let weight = weights[layer] * history_affinity(current, echo);
                    for c in 0..3 {
                        result[c] += (echo[c] - current[c]) * weight;
                    }
                }
                scratch[i] = result;
            }
        }
        std::mem::swap(pixels, scratch);
    }
}

#[cfg(test)]
mod native_port_regressions {
    use super::*;

    #[test]
    fn configured_rgb_radius_is_a_strict_vector_bound() {
        let eye = Retina::new();
        let size = (65, 65);
        for time in [0.0, 12.0, 50.67, 500.0] {
            let mut pixels = vec![[0.0; 3]; size.0 * size.1];
            pixels[32 * size.0 + 32] = [1.0; 3];
            eye.apply_channels(&mut pixels, &mut Vec::new(), size, size, time, 4.0);
            for c in 0..3 {
                let energy: f32 = pixels.iter().map(|p| p[c]).sum();
                let mut center = [0.0; 2];
                for (i, pixel) in pixels.iter().enumerate() {
                    center[0] += pixel[c] * ((i % size.0) as f32 - 32.0) / energy;
                    center[1] += pixel[c] * ((i / size.0) as f32 - 32.0) / energy;
                }
                assert!(
                    center[0].hypot(center[1]) <= 2.0001,
                    "time={time}, channel={c}, shift={center:?}"
                );
            }
        }
    }

    #[test]
    fn bright_neutral_reservoirs_cannot_bleach_saturated_current() {
        let mut eye = Retina::new();
        let mut pixels = vec![[4.0; 3]; 9];
        let mut scratch = Vec::new();
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (173.0, 0.0), 4.0);
        pixels.fill([1.0, 0.0, 1.0]);
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (173.0, 0.0), 4.0);
        assert!(
            pixels.iter().all(|p| p[1] < 0.02 * p[0].max(p[2])),
            "neutral history produced milk: {:?}",
            pixels[0]
        );
    }

    #[test]
    fn raw_reservoirs_have_fixed_achromatic_wall_time_release_half_lives() {
        let mut eye = Retina::new();
        let mut pixels = vec![[1.0; 3]];
        let mut scratch = Vec::new();
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
        pixels[0] = [0.0; 3];
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.35, 0.35), 4.0);
        for (layer, half_life) in [0.24_f32, 0.72, 1.65].into_iter().enumerate() {
            let expected = 2.0_f32.powf(-0.35 / half_life);
            for actual in eye.history[layer][0] {
                assert!(
                    (actual - expected).abs() < 0.000001,
                    "layer={layer}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn subjective_control_clamps_and_zero_bypasses_every_stage() {
        let mut eye = Retina::new();
        assert_eq!(eye.subjective_intensity, 1.0);
        eye.set_subjective_intensity(8.0);
        assert_eq!(eye.subjective_intensity, 2.0);
        eye.set_subjective_intensity(f32::NAN);
        assert_eq!(eye.subjective_intensity, 2.0);
        eye.set_subjective_intensity(-1.0);
        assert_eq!(eye.subjective_intensity, 0.0);
        let source: Vec<_> = (0..35)
            .map(|i| [i as f32 * 0.03, 0.2, (i % 4) as f32 * 0.2])
            .collect();
        let mut pixels = source.clone();
        let mut scratch = Vec::new();
        eye.apply(&mut pixels, &mut scratch, (7, 5), (7, 5), (173.0, 0.1), 4.0);
        eye.apply_diffraction(&mut pixels, &mut scratch, (7, 5), (7, 5), 4.0);
        eye.apply_channels(&mut pixels, &mut scratch, (7, 5), (7, 5), 173.0, 4.0);
        eye.apply_chroma(&mut pixels, (7, 5), 173.0, 4.0);
        assert_eq!(pixels, source);
        assert!(!eye.valid);
    }

    #[test]
    fn chroma_fields_preserve_luminance_positive_gamut_neutrals_and_frozen_state() {
        let mut eye = Retina::new();
        eye.set_subjective_intensity(2.0);
        let size = (53, 31);
        let original: Vec<_> = (0..size.0 * size.1)
            .map(|i| {
                [
                    0.2 + (i % 7) as f32 * 0.2,
                    0.04,
                    0.6 + (i % 3) as f32 * 0.15,
                ]
            })
            .collect();
        let mut first = original.clone();
        eye.apply_chroma(&mut first, size, 17.0, 4.0);
        let mut changed = 0;
        for (a, b) in first.iter().zip(&original) {
            let y = |p: &[f32; 3]| (0..3).map(|c| p[c] * LUMA[c]).sum::<f32>();
            assert!((y(a) - y(b)).abs() < 0.000001);
            assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0));
            changed += usize::from((0..3).any(|c| (a[c] - b[c]).abs() > 0.001));
        }
        assert!(
            changed > first.len() / 3,
            "spatial chroma adaptation is inactive"
        );
        let mut repeated = original;
        eye.apply_chroma(&mut repeated, size, 17.0, 4.0);
        assert_eq!(first, repeated);
        let mut neutral: Vec<_> = (0..size.0 * size.1)
            .map(|i| [(i % 13) as f32 * 0.3; 3])
            .collect();
        let expected = neutral.clone();
        eye.apply_chroma(&mut neutral, size, 17.0, 4.0);
        assert!(
            neutral
                .iter()
                .zip(expected)
                .all(|(a, b)| a.iter().zip(b).all(|(a, b)| (*a - b).abs() < 0.000001))
        );
        let mut uniform = vec![[0.9, 0.04, 1.2]; size.0 * size.1];
        let expected = uniform.clone();
        eye.apply_chroma(&mut uniform, size, 17.0, 4.0);
        assert_eq!(uniform, expected);
    }

    #[test]
    fn envelope_varies_coherently_and_reset_preserves_phase_without_old_light() {
        let mut eye = Retina::new();
        let mut minimum = f32::INFINITY;
        let mut maximum = 0.0_f32;
        for step in 0..1000 {
            let time = step as f32 * 0.173;
            let dose = eye.dose(time, 4.0);
            minimum = minimum.min(dose);
            maximum = maximum.max(dose);
            assert!((dose - eye.dose(time + 0.001, 4.0)).abs() < 0.0001);
        }
        assert!(maximum - minimum > 0.2);
        let mut pixels = vec![[4.0; 3]; 9];
        let mut scratch = Vec::new();
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (173.0, 0.1), 4.0);
        eye.reset_history();
        pixels.fill([0.0; 3]);
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (173.0, 0.0), 4.0);
        assert_eq!(eye.elapsed, 173.0);
        assert!(pixels.iter().all(|p| *p == [0.0; 3]));
    }
}

#[cfg(test)]
mod pulse_regressions {
    use super::*;

    #[test]
    fn a_short_glint_is_captured_before_its_slow_tail_fades() {
        let mut eye = Retina::new();
        let mut scratch = Vec::new();
        let mut pixels = vec![[0.0; 3]];
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
        let glint = [4.0, 0.2, 1.0];
        pixels[0] = glint;
        eye.apply(
            &mut pixels,
            &mut scratch,
            (1, 1),
            (1, 1),
            (1.0 / 60.0, 1.0 / 60.0),
            4.0,
        );
        for (layer, history) in eye.history.iter().enumerate() {
            let captured = history[0][0] / glint[0];
            assert!(
                captured > 0.10,
                "layer {layer} lost the brief glint: {captured}"
            );
        }
        pixels[0] = [0.0; 3];
        eye.apply(
            &mut pixels,
            &mut scratch,
            (1, 1),
            (1, 1),
            (1.0 + 1.0 / 60.0, 1.0),
            4.0,
        );
        assert!(
            pixels[0][0] > 0.02,
            "brief glint left no visible one-second tail: {:?}",
            pixels[0]
        );
        assert!((pixels[0][0] / pixels[0][2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn attack_and_release_steps_keep_one_shared_rgb_clock() {
        for (layer, (attack, release)) in [(0.025, 0.24), (0.05, 0.72), (0.09, 1.65)]
            .into_iter()
            .enumerate()
        {
            for source in [[1.0; 3], [0.8, 0.2, 0.4]] {
                let mut eye = Retina::new();
                let mut scratch = Vec::new();
                let mut pixels = vec![[0.0; 3]];
                eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
                pixels[0] = source;
                eye.apply(
                    &mut pixels,
                    &mut scratch,
                    (1, 1),
                    (1, 1),
                    (attack, attack),
                    4.0,
                );
                for (actual, source) in eye.history[layer][0].into_iter().zip(source) {
                    assert!(
                        (actual - source * 0.5).abs() < 1e-6,
                        "attack layer {layer}: {actual}"
                    );
                }
                pixels[0] = [0.0; 3];
                eye.apply(
                    &mut pixels,
                    &mut scratch,
                    (1, 1),
                    (1, 1),
                    (attack + release, release),
                    4.0,
                );
                for (actual, source) in eye.history[layer][0].into_iter().zip(source) {
                    assert!(
                        (actual - source * 0.25).abs() < 1e-6,
                        "release layer {layer}: {actual}"
                    );
                }
            }
        }
    }

    fn pulse_at_cadence(fps: usize) -> (Retina, [f32; 3]) {
        let mut eye = Retina::new();
        eye.set_subjective_intensity(2.0);
        let mut scratch = Vec::new();
        let mut pixels = vec![[0.0; 3]];
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
        // The same 100 ms exposure, followed by 900 ms of darkness at every FPS.
        // A one-frame pulse would have a different physical duration at each cadence.
        for frame in 1..=fps {
            pixels[0] = if frame <= fps / 10 {
                [2.0, 0.5, 1.0]
            } else {
                [0.0; 3]
            };
            eye.apply(
                &mut pixels,
                &mut scratch,
                (1, 1),
                (1, 1),
                (frame as f32 / fps as f32, 1.0 / fps as f32),
                4.0,
            );
        }
        (eye, pixels[0])
    }

    #[test]
    fn the_same_wall_time_pulse_has_the_same_tail_at_30_60_and_120_hz() {
        let (reference, reference_output) = pulse_at_cadence(120);
        assert!(
            reference_output[0] > 0.06,
            "pulse was barely retained: {reference_output:?}"
        );
        for fps in [30, 60] {
            let (eye, output) = pulse_at_cadence(fps);
            for (actual, expected) in eye.history.iter().zip(&reference.history) {
                for (actual, expected) in actual[0].iter().zip(expected[0]) {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "{fps} Hz history: {actual} != {expected}"
                    );
                }
            }
            for (actual, expected) in output.into_iter().zip(reference_output) {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "{fps} Hz output: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn a_dimming_color_change_uses_release_for_every_channel() {
        let mut eye = Retina::new();
        let mut scratch = Vec::new();
        let mut pixels = vec![[0.0, 0.5, 0.0]];
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
        // Red rises, but total luminance falls: choosing attack separately for red
        // would change the hue and leave the segment between the two source colors.
        pixels[0] = [1.0, 0.0, 0.0];
        eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.24, 0.24), 4.0);
        assert!((eye.history[0][0][0] - 0.5).abs() < 1e-6);
        assert!((eye.history[0][0][1] - 0.25).abs() < 1e-6);
        for history in &eye.history {
            assert!((history[0][0] + history[0][1] * 2.0 - 1.0).abs() < 1e-6);
            assert_eq!(history[0][2], 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complementary_history_preserves_current_color_and_a_faint_old_trace() {
        let mut eye = Retina::new();
        let mut pixels = vec![[1.0, 0.0, 0.0]; 9];
        let mut scratch = Vec::new();
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (0.0, 0.0), 4.0);
        pixels.fill([0.0, 1.0, 1.0]);
        eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (0.0, 0.0), 4.0);
        println!("red history -> cyan current, intensity 4: {:?}", pixels[0]);
        for pixel in pixels {
            assert!(
                (0.001..0.03).contains(&pixel[0]),
                "old red trace: {pixel:?}"
            );
            assert!(
                pixel[1] > 0.97 && pixel[2] > 0.97,
                "cyan became grey: {pixel:?}"
            );
        }
        assert!(eye.history.iter().flatten().all(|p| *p == [1.0, 0.0, 0.0]));
    }

    #[test]
    fn matching_history_chromaticity_and_black_tails_remain_convex_and_colored() {
        let previous = [0.4, 0.2, 0.1];
        for current in [[0.8, 0.4, 0.2], [0.0; 3]] {
            let mut eye = Retina::new();
            let mut pixels = vec![previous; 9];
            let mut scratch = Vec::new();
            eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (0.0, 0.0), 4.0);
            pixels.fill(current);
            eye.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (0.0, 0.0), 4.0);
            for pixel in pixels {
                assert!(pixel[0] > 0.0 && pixel[0] < 0.8);
                assert!((pixel[0] - 2.0 * pixel[1]).abs() < 1e-6);
                assert!((pixel[1] - 2.0 * pixel[2]).abs() < 1e-6);
                for c in 0..3 {
                    assert!(
                        pixel[c] >= current[c].min(previous[c])
                            && pixel[c] <= current[c].max(previous[c])
                    );
                }
            }
        }
    }

    #[test]
    fn complementary_history_blending_is_continuous_as_either_color_reaches_black() {
        fn output(
            previous: [f32; 3],
            current: [f32; 3],
        ) -> [f32; 3] {
            let mut eye = Retina::new();
            let mut pixels = vec![previous];
            let mut scratch = Vec::new();
            eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
            pixels[0] = current;
            eye.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 4.0);
            pixels[0]
        }
        for reverse in [false, true] {
            let pair = |small| {
                if reverse {
                    ([1.0, 0.0, 0.0], [0.0, small, 0.0])
                } else {
                    ([0.0, small, 0.0], [1.0, 0.0, 0.0])
                }
            };
            let (a, b) = pair(0.0);
            let black_limit = output(a, b);
            for small in [0.999e-6, 1.001e-6, 1e-8] {
                let (a, b) = pair(small);
                let near = output(a, b);
                assert!(
                    near.iter()
                        .zip(black_limit)
                        .all(|(a, b)| (*a - b).abs() < 1e-5),
                    "discontinuous near black: {near:?} versus {black_limit:?}"
                );
            }
        }
    }

    #[test]
    fn rgb_shift_strength_rises_smoothly_with_dose() {
        let eye = Retina::new();
        let source: Vec<_> = (0..64 * 24).map(|i| [(i % 64) as f32 / 64.0; 3]).collect();
        let mut previous_change = 0.0;
        for intensity in [0.1, 1.0, 4.0] {
            let mut actual = source.clone();
            eye.apply_channels(
                &mut actual,
                &mut Vec::new(),
                (64, 24),
                (64, 24),
                4.0,
                intensity,
            );
            let change = actual
                .iter()
                .zip(&source)
                .map(|(a, b)| (a[0] - b[0]).abs())
                .sum::<f32>();
            assert!(change > previous_change);
            previous_change = change;
        }
    }

    #[test]
    fn rgb_shift_defaults_small_clamps_and_zero_is_exact_identity() {
        let mut eye = Retina::new();
        assert_eq!(eye.rgb_shift, 2.0);
        let source: Vec<_> = (0..64 * 24).map(|i| [(i % 64) as f32 / 64.0; 3]).collect();
        let mut default = source.clone();
        eye.apply_channels(&mut default, &mut Vec::new(), (64, 24), (64, 24), 0.0, 2.0);
        // Two pixels is a ceiling; the coherent envelope starts below it.
        let displacement = (default[32][0] - source[32][0]) * 64.0;
        assert!(displacement > 0.0 && displacement <= 2.0);
        for value in [0.0, -1.0] {
            eye.set_rgb_shift(value);
            let mut pixels = source.clone();
            eye.apply_channels(&mut pixels, &mut Vec::new(), (64, 24), (64, 24), 0.0, 4.0);
            assert_eq!(pixels, source);
        }
        eye.set_rgb_shift(99.0);
        assert_eq!(eye.rgb_shift, 12.0);
        for invalid in [f32::NAN, f32::INFINITY] {
            eye.set_rgb_shift(invalid);
            assert_eq!(eye.rgb_shift, 12.0);
        }
    }

    #[test]
    fn direct_channels_split_a_dim_broad_image_without_a_glint_mask() {
        let (w, h) = (65, 33);
        let original: Vec<_> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 - 32.0;
                [0.04 + 0.12 * (-x * x / 32.0).exp(); 3]
            })
            .collect();
        let mut eye = Retina::new();
        eye.set_rgb_shift(8.0);
        let mut pixels = original.clone();
        eye.apply_channels(&mut pixels, &mut Vec::new(), (w, h), (w, h), 0.0, 2.0);
        let centroids: [f32; 3] = std::array::from_fn(|c| {
            let mut mass = 0.0;
            let mut moment = 0.0;
            for x in 0..w {
                let v = (pixels[16 * w + x][c] - 0.04).max(0.0);
                mass += v;
                moment += v * x as f32;
            }
            moment / mass
        });
        assert!(
            centroids.iter().copied().fold(0.0_f32, f32::max)
                - centroids.iter().copied().fold(100.0_f32, f32::min)
                > 2.0,
            "channels not spatially separated: {centroids:?}"
        );
        assert!(pixels.iter().zip(original).any(|(a, b)| a != &b));
    }


    #[test]
    fn direct_channels_preserve_uniform_light_bypass_and_paused_determinism() {
        let (w, h) = (65, 33);
        let mut eye = Retina::new();
        let mut scratch = Vec::new();
        for light in [[0.0; 3], [0.001, 0.2, 0.95], [1.0; 3]] {
            let mut pixels = vec![light; w * h];
            let before = pixels.clone();
            eye.apply_channels(&mut pixels, &mut scratch, (w, h), (w, h), 30.0, 4.0);
            assert_eq!(pixels, before);
        }
        let source: Vec<_> = (0..w * h)
            .map(|i| {
                [
                    (i % 11) as f32 / 10.0,
                    (i % 7) as f32 / 6.0,
                    (i % 3) as f32 / 2.0,
                ]
            })
            .collect();
        let mut first = source.clone();
        eye.apply_channels(&mut first, &mut scratch, (w, h), (w, h), 7.0, 2.0);
        assert_ne!(first, source);
        assert!(
            first
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(eye.history.iter().all(Vec::is_empty));
        for _ in 0..3 {
            let mut next = source.clone();
            eye.apply_channels(&mut next, &mut scratch, (w, h), (w, h), 7.0, 2.0);
            assert_eq!(next, first);
        }
        for intensity in [0.0, -1.0, f32::NAN] {
            let mut p = source.clone();
            eye.apply_channels(&mut p, &mut scratch, (w, h), (w, h), 7.0, intensity);
            assert_eq!(p, source);
        }
        eye.set_enabled(false);
        let mut p = source.clone();
        eye.apply_channels(&mut p, &mut scratch, (w, h), (w, h), 7.0, 4.0);
        assert_eq!(p, source);
    }

    #[test]
    fn direct_channel_drift_is_continuous_and_uses_display_pixel_units() {
        fn centroid(scale: usize) -> [f32; 3] {
            let (w, h) = (65 * scale, 65 * scale);
            let eye = Retina::new();
            let mut p: Vec<_> = (0..w * h)
                .map(|i| {
                    let x = (i % w) as f32 / scale as f32 + 0.5 / scale as f32 - 32.5;
                    let y = (i / w) as f32 / scale as f32 + 0.5 / scale as f32 - 32.5;
                    [(-(x * x + y * y) / 8.0).exp(); 3]
                })
                .collect();
            eye.apply_channels(&mut p, &mut Vec::new(), (w, h), (65, 65), 10.0, 2.0);
            std::array::from_fn(|c| {
                let sum: f32 = p.iter().map(|p| p[c]).sum();
                p.iter()
                    .enumerate()
                    .map(|(i, p)| p[c] * ((i % w) as f32 + 0.5) / scale as f32)
                    .sum::<f32>()
                    / sum
            })
        }
        for (a, b) in centroid(1).into_iter().zip(centroid(2)) {
            assert!(
                (a - b).abs() < 0.02,
                "display shift changed with quality: {a} {b}"
            );
        }
        let eye = Retina::new();
        let (w, h) = (65, 33);
        let mut scratch = Vec::new();
        let source: Vec<_> = (0..w * h).map(|i| [(i % w) as f32 / w as f32; 3]).collect();
        for cycle in [std::f32::consts::TAU / 0.075, std::f32::consts::TAU / 0.031] {
            let mut a = source.clone();
            let mut b = source.clone();
            eye.apply_channels(&mut a, &mut scratch, (w, h), (w, h), cycle - 0.001, 4.0);
            let capacity = scratch.capacity();
            eye.apply_channels(&mut b, &mut scratch, (w, h), (w, h), cycle + 0.001, 4.0);
            assert_eq!(scratch.capacity(), capacity);
            assert!(
                a.iter()
                    .zip(b)
                    .all(|(a, b)| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 0.001))
            );
        }
    }

    fn fading_uniform_light(
        intensity: f32,
        elapsed: f32,
        steps: usize,
    ) -> [f32; 3] {
        let mut retina = Retina::new();
        let mut pixels = vec![[1.0; 3]; 9];
        let mut scratch = Vec::new();
        retina.apply(
            &mut pixels,
            &mut scratch,
            (3, 3),
            (3, 3),
            (0.0, 0.0),
            intensity,
        );
        for step in 1..=steps {
            pixels.fill([0.0; 3]);
            retina.apply(
                &mut pixels,
                &mut scratch,
                (3, 3),
                (3, 3),
                (elapsed * step as f32 / steps as f32, elapsed / steps as f32),
                intensity,
            );
        }
        pixels[4]
    }

    #[test]
    fn layered_echo_remains_visible_beyond_one_second() {
        let tail = fading_uniform_light(2.0, 1.6, 96);
        assert!(tail.iter().all(|c| *c > 0.005 && *c < 0.04), "{tail:?}");
    }

    #[test]
    fn higher_trip_strengthens_echo_without_changing_its_age() {
        let normal = fading_uniform_light(2.0, 1.6, 96);
        let overdrive = fading_uniform_light(4.0, 1.6, 96);
        for c in 0..3 {
            assert!(
                overdrive[c] > normal[c] * 1.3,
                "{normal:?} -> {overdrive:?}"
            );
            assert!(overdrive[c] <= 1.0);
        }
    }

    #[test]
    fn current_edges_retain_a_minimum_unblurred_contribution() {
        for (intensity, minimum) in [(2.0, 0.78), (4.0, 0.45)] {
            let mut retina = Retina::new();
            let mut pixels = vec![[0.0; 3]; 9];
            let mut scratch = Vec::new();
            retina.apply(
                &mut pixels,
                &mut scratch,
                (3, 3),
                (3, 3),
                (0.0, 0.0),
                intensity,
            );
            pixels[4] = [1.0; 3];
            retina.apply(
                &mut pixels,
                &mut scratch,
                (3, 3),
                (3, 3),
                (0.0, 0.0),
                intensity,
            );
            assert!(
                pixels[4].iter().all(|c| *c >= minimum - 1e-6),
                "dose {intensity}: {:?}",
                pixels[4]
            );
        }
    }

    #[test]
    fn old_light_drifts_into_spatially_separated_chromatic_echoes() {
        let (w, h) = (129, 65);
        let mut retina = Retina::new();
        let mut pixels = vec![[0.0; 3]; w * h];
        let mut scratch = Vec::new();
        for y in 27..32 {
            for x in 83..88 {
                pixels[y * w + x] = [1.0; 3];
            }
        }
        retina.apply(&mut pixels, &mut scratch, (w, h), (w, h), (7.0, 0.0), 4.0);
        pixels.fill([0.0; 3]);
        retina.apply(&mut pixels, &mut scratch, (w, h), (w, h), (7.1, 0.1), 4.0);
        retina.apply_channels(&mut pixels, &mut scratch, (w, h), (w, h), 7.1, 4.0);
        let mut outside = 0;
        let mut separation = 0.0_f32;
        for y in 0..h {
            for x in 0..w {
                let p = pixels[y * w + x];
                if !(83..88).contains(&x) || !(27..32).contains(&y) {
                    outside += usize::from(p.iter().any(|c| *c > 0.02));
                    separation = separation.max((p[0] - p[2]).abs());
                }
            }
        }
        assert!(outside >= 8, "only {outside} pixels beyond the source");
        assert!(
            separation > 0.02,
            "RGB echoes must separate, got {separation}"
        );
    }

    #[test]
    fn spatial_drift_is_continuous_across_phase_cycles() {
        let (w, h) = (65, 33);
        let source: Vec<_> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                [
                    if x % 11 < 4 { 1.0 } else { 0.0 },
                    y as f32 / h as f32,
                    if y % 7 < 3 { 0.8 } else { 0.1 },
                ]
            })
            .collect();
        let mut retina = Retina::new();
        let mut scratch = Vec::new();
        let cycle = std::f32::consts::TAU / 0.12;
        let mut before = source.clone();
        retina.apply(
            &mut before,
            &mut scratch,
            (w, h),
            (w, h),
            (cycle - 0.001, 0.0),
            4.0,
        );
        let mut after = source;
        retina.apply(
            &mut after,
            &mut scratch,
            (w, h),
            (w, h),
            (cycle + 0.001, 0.0),
            4.0,
        );
        let maximum = before
            .iter()
            .zip(after)
            .flat_map(|(a, b)| std::array::from_fn::<_, 3, _>(|c| (a[c] - b[c]).abs()))
            .fold(0.0_f32, f32::max);
        assert!(
            maximum < 0.003,
            "cycle reset caused an echo jump of {maximum}"
        );
    }

    #[test]
    fn zero_dose_invalidates_history_instead_of_resurrecting_old_light() {
        let mut retina = Retina::new();
        let mut pixels = vec![[1.0; 3]; 4];
        let mut scratch = Vec::new();
        retina.apply(&mut pixels, &mut scratch, (2, 2), (2, 2), (0.0, 0.0), 2.0);
        pixels.fill([0.0; 3]);
        retina.apply(&mut pixels, &mut scratch, (2, 2), (2, 2), (1.0, 0.2), 0.0);
        assert!(pixels.iter().all(|p| *p == [0.0; 3]));
        retina.apply(&mut pixels, &mut scratch, (2, 2), (2, 2), (1.0, 0.0), 2.0);
        assert!(pixels.iter().all(|p| *p == [0.0; 3]));
    }

    #[test]
    fn empty_or_malformed_images_invalidate_without_indexing() {
        let mut retina = Retina::new();
        let mut pixels = vec![[0.4; 3]; 5];
        let original = pixels.clone();
        let mut scratch = Vec::new();
        retina.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (0.0, 0.1), 2.0);
        assert_eq!(pixels, original);
        pixels.clear();
        retina.apply(&mut pixels, &mut scratch, (0, 0), (0, 0), (0.0, 0.1), 2.0);
        pixels.resize(1, [0.0; 3]);
        retina.apply(&mut pixels, &mut scratch, (1, 1), (1, 1), (0.0, 0.0), 2.0);
        assert_eq!(pixels, vec![[0.0; 3]]);
    }

    #[test]
    fn paused_redraws_do_not_advance_any_reservoir() {
        let (w, h) = (7, 5);
        let mut retina = Retina::new();
        let mut pixels: Vec<_> = (0..w * h).map(|i| [i as f32 / 35.0; 3]).collect();
        let mut scratch = Vec::new();
        retina.apply(&mut pixels, &mut scratch, (w, h), (w, h), (0.0, 0.0), 4.0);
        pixels.fill([0.0; 3]);
        retina.apply(&mut pixels, &mut scratch, (w, h), (w, h), (0.1, 0.1), 4.0);
        let frozen_image = pixels.clone();
        let frozen_history = retina.history.clone();
        for _ in 0..32 {
            pixels.fill([0.0; 3]);
            retina.apply(&mut pixels, &mut scratch, (w, h), (w, h), (0.1, 0.0), 4.0);
            assert_eq!(pixels, frozen_image);
            assert_eq!(retina.history, frozen_history);
        }
    }

    #[test]
    fn changing_quality_preserves_the_age_of_every_echo_layer() {
        let mut retina = Retina::new();
        let mut pixels = vec![[1.0; 3]; 13 * 7];
        let mut scratch = Vec::new();
        retina.apply(&mut pixels, &mut scratch, (13, 7), (13, 7), (0.0, 0.0), 2.0);
        for step in 1..=40 {
            let size = [(7, 4), (13, 7), (1, 1)][step % 3];
            pixels.resize(size.0 * size.1, [0.0; 3]);
            pixels.fill([0.0; 3]);
            retina.apply(
                &mut pixels,
                &mut scratch,
                size,
                (13, 7),
                (step as f32 * 0.04, 0.04),
                2.0,
            );
        }
        let expected = fading_uniform_light(2.0, 1.6, 40);
        assert!(
            pixels
                .iter()
                .all(|p| { p.iter().zip(expected).all(|(a, b)| (*a - b).abs() < 1e-6) })
        );
    }

    #[test]
    fn layered_mixing_cannot_exceed_the_channel_radiance_bounds() {
        let (w, h) = (23, 13);
        let maximum = [0.31, 0.83, 1.73];
        let mut retina = Retina::new();
        retina.set_subjective_intensity(2.0);
        let mut pixels = vec![[0.0; 3]; w * h];
        let mut scratch = Vec::new();
        for step in 0..40 {
            for (i, pixel) in pixels.iter_mut().enumerate() {
                *pixel = std::array::from_fn(|c| {
                    maximum[c] * ((i * 37 + c * 17 + step * 71) % 11) as f32 / 10.0
                });
            }
            retina.apply(
                &mut pixels,
                &mut scratch,
                (w, h),
                (w, h),
                (step as f32 * 0.1, [0.0, 0.005, 0.03, 0.4][step % 4]),
                [0.25, 2.0, 4.0, 20.0][step % 4],
            );
            for pixel in &pixels {
                for c in 0..3 {
                    assert!(
                        pixel[c].is_finite() && pixel[c] >= 0.0 && pixel[c] <= maximum[c] + 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn uniform_light_is_preserved_through_drift_history_and_resize() {
        let mut retina = Retina::new();
        let color = [0.3, 0.6, 1.1];
        let mut scratch = Vec::new();
        for (w, h) in [(13, 7), (7, 4), (1, 1), (13, 7)] {
            for frame in 0..20 {
                let mut pixels = vec![color; w * h];
                retina.apply(
                    &mut pixels,
                    &mut scratch,
                    (w, h),
                    (13, 14),
                    (frame as f32 * 0.1, 0.1),
                    [0.25, 1.0, 2.0, 4.0][frame % 4],
                );
                assert!(
                    pixels
                        .iter()
                        .all(|p| p.iter().zip(color).all(|(a, b)| (*a - b).abs() < 1e-6))
                );
            }
        }
    }

    #[test]
    fn temporal_persistence_is_independent_of_frame_partition() {
        for intensity in [0.25, 1.0, 2.0, 4.0] {
            let one = fading_uniform_light(intensity, 1.6, 1);
            let many = fading_uniform_light(intensity, 1.6, 96);
            for c in 0..3 {
                assert!((one[c] - many[c]).abs() < 1e-6);
            }
            assert_eq!(one[0], one[1]);
            assert_eq!(one[1], one[2]);
        }
    }

    #[test]
    fn disabling_echo_is_identity_and_does_not_keep_a_stale_frame() {
        let mut retina = Retina::new();
        let original: Vec<_> = (0..9).map(|i| [i as f32 * 0.1; 3]).collect();
        let mut pixels = original.clone();
        let mut scratch = Vec::new();
        retina.set_enabled(false);
        retina.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (10.0, 0.1), 1.0);
        assert_eq!(pixels, original);
        retina.set_enabled(true);
        pixels.fill([0.0; 3]);
        retina.apply(&mut pixels, &mut scratch, (3, 3), (3, 3), (10.0, 0.1), 1.0);
        assert!(pixels.iter().all(|p| *p == [0.0; 3]));
    }
}

#[cfg(test)]
mod diffraction_contract_tests {
    use super::Retina;

    #[test]
    fn diffraction_preserves_uniform_hdr_exactly_even_in_degenerate_viewports() {
        for size in [(1, 1), (1, 19), (23, 1), (23, 17)] {
            for value in [[0.0; 3], [0.137, 2.5, 16.0]] {
                for intensity in [2.0, 4.0] {
                    let eye = Retina::new();
                    let original = vec![value; size.0 * size.1];
                    let mut pixels = original.clone();
                    let mut scratch = Vec::new();
                    eye.apply_diffraction(&mut pixels, &mut scratch, size, size, intensity);
                    assert_eq!(pixels, original, "size={size:?}, intensity={intensity}");
                }
            }
        }
    }

    #[test]
    fn diffraction_preserves_impulse_channel_energy_at_interior_edges_and_corners() {
        // Includes 1-D and 1-pixel viewports, whose lobe coordinates reflect
        // repeatedly. HDR amplitudes also catch accidental SDR clipping.
        let amplitude = [1.0_f32, 2.0, 8.0];
        for (w, h) in [(1, 1), (1, 19), (23, 1), (65, 49)] {
            for (x, y) in [
                (w / 2, h / 2),
                (0, h / 2),
                (w / 2, 0),
                (0, 0),
                (w - 1, h - 1),
            ] {
                for intensity in [2.0, 4.0] {
                    let eye = Retina::new();
                    let mut pixels = vec![[0.0; 3]; w * h];
                    pixels[y * w + x] = amplitude;
                    let mut scratch = Vec::new();
                    eye.apply_diffraction(&mut pixels, &mut scratch, (w, h), (w, h), intensity);
                    for (c, expected_energy) in amplitude.into_iter().enumerate() {
                        let energy: f64 = pixels.iter().map(|p| f64::from(p[c])).sum();
                        let error = (energy / f64::from(expected_energy) - 1.0).abs();
                        assert!(
                            error < 0.00002,
                            "size={w}x{h}, impulse=({x},{y}), dose={intensity}, c={c}, \
                             energy={energy}"
                        );
                        assert!(pixels.iter().all(|p| {
                            p[c].is_finite() && p[c] >= 0.0 && p[c] <= expected_energy + 0.000001
                        }));
                    }
                }
            }
        }
    }

    #[test]
    fn diffraction_wavelength_spread_uses_display_pixels_and_preserves_the_interior_centroid() {
        fn moment(scale: usize) -> [f64; 3] {
            let display = (129, 97);
            let (w, h) = (display.0 * scale, display.1 * scale);
            let (cx, cy) = (w / 2, h / 2);
            let eye = Retina::new();
            let mut pixels = vec![[0.0; 3]; w * h];
            pixels[cy * w + cx] = [1.0; 3];
            let mut scratch = Vec::new();
            eye.apply_diffraction(&mut pixels, &mut scratch, (w, h), display, 2.0);
            let mut radial_moment = [0.0_f64; 3];
            let mut centroid = [[0.0_f64; 2]; 3];
            for (i, pixel) in pixels.iter().enumerate() {
                let dx = ((i % w) as f64 - cx as f64) / scale as f64;
                let dy = ((i / w) as f64 - cy as f64) / scale as f64;
                for c in 0..3 {
                    let mass = f64::from(pixel[c]);
                    radial_moment[c] += mass * (dx * dx + dy * dy);
                    centroid[c][0] += mass * dx;
                    centroid[c][1] += mass * dy;
                }
            }
            assert!(
                radial_moment[0] > radial_moment[1]
                    && radial_moment[1] > radial_moment[2]
                    && radial_moment[2] > 0.0,
                "wavelength order 650/550/460nm: {radial_moment:?}"
            );
            assert!(centroid.iter().flatten().all(|v| v.abs() < 0.0001));
            assert!(pixels[cy * w + cx].iter().all(|v| *v >= 0.8999 && *v < 1.0));
            radial_moment
        }
        let ordinary = moment(1);
        let supersampled = moment(2);
        for c in 0..3 {
            // Bilinear sampling adds <= 0.5 pixel^2 times the 0.10
            // side-lobe mass; it need not scale exactly with wavelength^2.
            assert!(
                (ordinary[c] - supersampled[c]).abs() < 0.06,
                "channel {c}: spread changed with grid scale: {ordinary:?} vs {supersampled:?}"
            );
        }
    }

    #[test]
    fn diffraction_bypasses_exactly_and_repeated_redraws_do_not_change_history() {
        let (w, h) = (31, 23);
        let original: Vec<[f32; 3]> = (0..w * h)
            .map(|i| [(i % 7) as f32 * 0.2, 0.1, (i % 11) as f32 * 0.09])
            .collect();
        let mut eye = Retina::new();
        let mut scratch = Vec::new();
        let mut history_input = original.clone();
        eye.apply(
            &mut history_input,
            &mut scratch,
            (w, h),
            (w, h),
            (7.0, 0.1),
            2.0,
        );
        let histories = eye.history.clone();
        for intensity in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut pixels = original.clone();
            eye.apply_diffraction(&mut pixels, &mut scratch, (w, h), (w, h), intensity);
            assert_eq!(pixels, original);
        }
        let mut expected = original.clone();
        eye.apply_diffraction(&mut expected, &mut scratch, (w, h), (w, h), 2.0);
        assert_ne!(expected, original, "nonuniform HDR did not scatter");
        for _ in 0..4 {
            let mut pixels = original.clone();
            eye.apply_diffraction(&mut pixels, &mut scratch, (w, h), (w, h), 2.0);
            assert_eq!(pixels, expected, "a fixed input accumulated PSF feedback");
            assert_eq!(eye.history, histories, "PSF modified retinal reservoirs");
        }
        eye.set_enabled(false);
        let mut pixels = original.clone();
        eye.apply_diffraction(&mut pixels, &mut scratch, (w, h), (w, h), 4.0);
        assert_eq!(pixels, original);
    }
}
