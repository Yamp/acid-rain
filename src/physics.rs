//! SI units shared by dynamics and geometry; time scaling belongs to the UI.
pub const PATCH_LENGTH: f32 = 2.0;
pub const DEPTH: f32 = 0.70;
pub const GRAVITY: f32 = 9.81;
pub const DENSITY: f32 = 998.2;
pub const SURFACE_TENSION: f32 = 0.0728;
pub const KINEMATIC_VISCOSITY: f32 = 1.004e-6;

pub fn angular_frequency(
    k: f32,
    depth: f32,
) -> f32 {
    (GRAVITY * k + SURFACE_TENSION / DENSITY * k * k * k)
        .mul_add((k * depth).tanh(), 0.0)
        .sqrt()
}

#[cfg(test)]
pub fn group_velocity(
    k: f32,
    depth: f32,
) -> f32 {
    let omega = angular_frequency(k, depth);
    if omega == 0.0 {
        return (GRAVITY * depth).sqrt();
    }
    let tension = SURFACE_TENSION / DENSITY;
    let t = (k * depth).tanh();
    ((GRAVITY + 3.0 * tension * k * k) * t
        + (GRAVITY * k + tension * k * k * k) * depth * (1.0 - t * t))
        / (2.0 * omega)
}

/// Leading-order clean-water viscous amplitude decay (weak damping).
pub fn damping_rate(k: f32) -> f32 { 2.0 * KINEMATIC_VISCOSITY * k * k }

/// Exact underdamped impulse oscillator: displacement and velocity per unit kick.
pub fn impulse_response(
    k: f32,
    age: f32,
) -> (f32, f32) {
    if age < 0.0 {
        return (0.0, 0.0);
    }
    let omega = angular_frequency(k, DEPTH);
    let gamma = damping_rate(k);
    let frequency = (omega * omega - gamma * gamma).max(0.0).sqrt();
    if frequency <= f32::EPSILON {
        return (0.0, 0.0);
    }
    let (s, c) = (frequency * age).sin_cos();
    let decay = (-gamma * age).exp();
    let displacement = decay * s / frequency;
    (displacement, decay * c - gamma * displacement)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_dispersion_has_shallow_and_deep_limits() {
        let k = 0.001;
        let shallow = angular_frequency(k, DEPTH) / k;
        assert!((shallow / (GRAVITY * DEPTH).sqrt() - 1.0).abs() < 1e-5);
        let k = 5.0;
        let expected = (GRAVITY * k + SURFACE_TENSION / DENSITY * k.powi(3)).sqrt();
        assert!((angular_frequency(k, 10.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn group_speed_is_the_derivative_of_dispersion() {
        for k in [0.02, 1.0, 8.0, 50.0, 500.0] {
            let epsilon = k * 0.002;
            let numeric = (angular_frequency(k + epsilon, DEPTH)
                - angular_frequency(k - epsilon, DEPTH))
                / (2.0 * epsilon);
            assert!((group_velocity(k, DEPTH) - numeric).abs() < 0.002);
        }
    }

    #[test]
    fn capillary_waves_recover_increasing_short_wave_speed() {
        assert!(
            angular_frequency(2000.0, DEPTH) / 2000.0 > angular_frequency(500.0, DEPTH) / 500.0
        );
    }

    #[test]
    fn damped_modes_start_with_momentum_and_lose_energy_monotonically() {
        for k in [2.0, 20.0, 180.0] {
            assert_eq!(impulse_response(k, 0.0), (0.0, 1.0));
            let omega = angular_frequency(k, DEPTH);
            let mut previous = 1.0;
            for step in 1..=1000 {
                let (height, velocity) = impulse_response(k, step as f32 * 0.01);
                let energy = velocity * velocity + omega * omega * height * height;
                assert!(
                    energy <= previous + 1e-5,
                    "mode k={k}: {energy} > {previous}"
                );
                previous = energy;
            }
        }
    }
}
