#[inline(always)]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let (r, g, b) = ((r1 + m) * 255.0, (g1 + m) * 255.0, (b1 + m) * 255.0);
    (r as u8, g as u8, b as u8)
}

#[inline(always)]
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r1 = r as f32 / 255.0;
    let g1 = g as f32 / 255.0;
    let b1 = b as f32 / 255.0;

    let cmax = r1.max(g1).max(b1);
    let cmin = r1.min(g1).min(b1);
    let delta = cmax - cmin;

    let h = if delta == 0.0 {
        0.0
    } else if cmax == r1 {
        60.0 * (((g1 - b1) / delta) % 6.0)
    } else if cmax == g1 {
        60.0 * (((b1 - r1) / delta) + 2.0)
    } else {
        60.0 * (((r1 - g1) / delta) + 4.0)
    };

    let s = if cmax == 0.0 { 0.0 } else { delta / cmax };

    (h, s, cmax)
}

/// Water body color (intrinsic, what you see looking through the surface).
#[inline(always)]
pub fn water_body_color(value: f32) -> (f32, f32, f32) {
    let amplitude = value.abs();
    let t_lin = (amplitude * 5.0).min(1.0);
    let t = t_lin * t_lin.sqrt();

    let (hue, sat, val) = if value >= 0.0 {
        (200.0 - t * 5.0, 0.5 + t * 0.5, 0.15 + t * 0.75)
    } else {
        (215.0 - t * 15.0, 0.2 + t * 0.7, 0.12 + t * 0.58)
    };

    let (r, g, b) = hsv_to_rgb(hue, sat, val);
    (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// Sky color at a given elevation (0 = horizon, 1 = zenith), cycling over time.
#[inline(always)]
pub fn sky_color(elapsed: f32, elevation: f32) -> (f32, f32, f32) {
    let phase = elapsed / 150.0 * 2.0 * std::f32::consts::PI;

    // Base hue rotates through the full spectrum over 2.5 minutes
    let base_hue = (elapsed / 150.0).fract() * 360.0;

    // Horizon is warmer (+20°) and brighter; zenith is deeper and more saturated
    let hue = base_hue + (1.0 - elevation) * 20.0;
    let hue = ((hue % 360.0) + 360.0) % 360.0;
    let sat = 0.2 + elevation * 0.2 + 0.05 * (phase * 1.5).sin();
    let val = 0.75 - elevation * 0.2 + 0.1 * phase.cos();

    let (r, g, b) = hsv_to_rgb(hue, sat.max(0.0), val.clamp(0.0, 1.0));
    (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// Schlick's Fresnel approximation for water (boosted R0 for visual effect).
#[inline(always)]
pub fn fresnel(cos_theta: f32) -> f32 {
    const R0: f32 = 0.12;
    R0 + (1.0 - R0) * (1.0 - cos_theta.max(0.0)).powi(5)
}
