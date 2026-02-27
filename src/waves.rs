use std::f32::consts::PI;

use crate::settings::SETTINGS;

/// Fast exp approximation via IEEE 754 bit manipulation.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // 2^23 / ln(2) ≈ 12102203.0; 127 * 2^23 = 1065353216.0
    let v = 12102203.0f32 * x + 1065353216.0;
    f32::from_bits(v.max(0.0) as u32)
}

pub fn calculate_wave_level(
    initial_strength: f32,
    secs_since_disturbance: f32,
    distance_from_disturbance: f32,
    full_attenuation_time: f32,
) -> f32 {
    let wavefront = secs_since_disturbance * SETTINGS.wave_speed;

    if distance_from_disturbance > wavefront {
        return 0.0;
    }

    let time_left = full_attenuation_time - secs_since_disturbance;
    if time_left <= 0.0 {
        return 0.0;
    }

    let radius = distance_from_disturbance;

    // Quadratic time decay
    let time_attenuation = (time_left / full_attenuation_time).powi(2);

    // Cylindrical spreading: 1/sqrt(r) — more gradual than 1/r²
    let dist_attenuation = 1.0 / (radius * 8.0 + 1.0).sqrt();

    // Suppress amplitude near the drop center — smooth Gaussian fade, no hard edge
    let r15 = radius * 15.0;
    let center_fade = 1.0 - fast_exp(-r15 * r15);

    // Wavefront envelope: x²·exp(1−x²) — zero at front, smooth peak, natural decay
    let behind_front = wavefront - radius;
    let wavelength = 1.0 / SETTINGS.waves_in_screen;
    let peak_dist = wavelength * 1.5;
    let bw = behind_front / peak_dist;
    let bw2 = bw * bw;
    let envelope = bw2 * fast_exp(1.0 - bw2);

    let amplitude = initial_strength * time_attenuation * dist_attenuation * envelope * center_fade;

    let time_phase = secs_since_disturbance * PI * 2.0 * SETTINGS.waves_per_second;
    let distance_phase = radius * PI * 2.0 * SETTINGS.waves_in_screen;
    let phase = time_phase - distance_phase;

    // Stokes-like profile: sin + sin² gives sharper crests, broader troughs
    let raw = phase.sin();
    let wave = raw + 0.3 * raw * raw + 0.12 * (2.0 * phase).sin();

    wave * amplitude
}
