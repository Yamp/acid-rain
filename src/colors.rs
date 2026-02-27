/// sRGB → linear conversion for physically correct lighting
#[inline(always)]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

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

/// Water body color (intrinsic). Returns **linear** RGB.
#[inline(always)]
pub fn water_body_color(value: f32) -> (f32, f32, f32) {
    let amplitude = value.abs();
    let t_lin = (amplitude * 5.0).min(1.0);
    let t = t_lin * t_lin.sqrt();

    let (hue, sat, val) = if value >= 0.0 {
        (200.0 - t * 5.0, 0.6 + t * 0.3, 0.65 + t * 0.3)
    } else {
        (215.0 - t * 15.0, 0.4 + t * 0.5, 0.55 + t * 0.3)
    };

    let (r, g, b) = hsv_to_rgb(hue, sat, val);
    (
        srgb_to_linear(r as f32 / 255.0),
        srgb_to_linear(g as f32 / 255.0),
        srgb_to_linear(b as f32 / 255.0),
    )
}

/// Rainbow sky color based on full 3D reflected direction, cycling over time.
/// Returns **linear** RGB.
#[inline(always)]
pub fn sky_color(elapsed: f32, dir: [f32; 3]) -> (f32, f32, f32) {
    use std::f32::consts::PI;
    let period = 37.5; // 150/4 = 37.5s full cycle
    let phase = elapsed / period * 2.0 * PI;
    let elevation = dir[2].max(0.0);

    // 2 linear rainbow cycles over 360° azimuth.
    // Exactly 2× means the atan2 seam at ±π falls on a hue period boundary → invisible.
    let base_hue = (elapsed / period).fract() * 360.0;
    let angle = dir[1].atan2(dir[0]); // [-π, π]
    let hue = base_hue
        + angle * (360.0 / PI)               // 2 full hue cycles per revolution
        + elevation * 60.0;
    let hue = ((hue % 360.0) + 360.0) % 360.0;

    let sat = 0.6 + elevation * 0.15 + 0.04 * (phase * 1.5).sin();
    let val = 0.88 + 0.07 * phase.cos();

    let (r, g, b) = hsv_to_rgb(hue, sat.max(0.0), val.clamp(0.0, 1.0));
    (
        srgb_to_linear(r as f32 / 255.0),
        srgb_to_linear(g as f32 / 255.0),
        srgb_to_linear(b as f32 / 255.0),
    )
}

/// Schlick's Fresnel approximation for water (boosted R0 for visual effect).
#[inline(always)]
pub fn fresnel(cos_theta: f32) -> f32 {
    const R0: f32 = 0.25;
    R0 + (1.0 - R0) * (1.0 - cos_theta.max(0.0)).powi(5)
}
