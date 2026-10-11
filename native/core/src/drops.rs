//! Bounded ballistic sources for the native three-dimensional pigment field.
//! This module owns source identities, not dissolved pigment. Impact records
//! survive skipped display frames; the GPU consumes each ID exactly once.
use crate::{
    Float4,
    physics::GRAVITY,
};

pub const MAX_DROPS: usize = 512;
pub const IMPACT_RETENTION: f64 = 4.0;
const FALL_HEIGHT: f32 = 0.42;
const INITIAL_DOWN_SPEED: f32 = 0.45;
// Native rain uses a concentrated three-dimensional source rather than the
// terminal's broad column footprint. Calibrate its pigment loading separately;
// the source still has an explicit conserved integral and diameter-cubed mass.
const NATIVE_PIGMENT_LOADING: f32 = 0.20;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ARDrop {
    pub position: Float4,
    pub pigment: Float4,
    pub dynamics: Float4,
    pub identity: [u32; 4],
}

#[derive(Clone, Debug)]
pub struct Drop {
    pub id: u64,
    pub world: [f64; 2],
    pub born: f64,
    pub start_z: f32,
    pub z: f32,
    pub radius: f32,
    pub strength: f32,
    pub size: f32,
    pub species: [f32; 3],
    pub mass: f32,
    pub impact: Option<f64>,
    pub impact_speed: f32,
    previous_time: f64,
    previous_surface: f32,
}

impl Drop {
    pub fn new(
        id: u64,
        world: [f64; 2],
        time: f64,
        surface: f32,
        strength: f32,
        size: f32,
    ) -> Self {
        let size = size.clamp(0.25, 3.0);
        let strength = strength.clamp(0.0, 1.0);
        let radius = 0.0055 * size;
        let start_z = surface + FALL_HEIGHT + radius;
        // Same independent species as terminal rain, with native source loading.
        let phase = id.wrapping_sub(1).wrapping_mul(37) % 101;
        let weights = match phase % 6 {
            0 => [0.0, 1.0, 0.05],
            1 => [0.7, 0.3, 0.0],
            2 => [1.0, 0.0, 0.05],
            3 => [0.45, 0.0, 0.55],
            4 => [0.0, 0.1, 0.9],
            _ => [0.15, 0.75, 0.1],
        };
        let sum: f32 = weights.iter().sum();
        Self {
            id,
            world,
            born: time,
            start_z,
            z: start_z,
            radius,
            strength,
            size,
            species: weights.map(|v| v / sum),
            mass: NATIVE_PIGMENT_LOADING * 0.006 * (strength / 0.2) * size.powi(3),
            impact: None,
            impact_speed: 0.0,
            previous_time: time,
            previous_surface: surface,
        }
    }

    fn altitude(
        &self,
        time: f64,
    ) -> f32 {
        let age = (time - self.born).max(0.0) as f32;
        self.start_z - INITIAL_DOWN_SPEED * age - 0.5 * GRAVITY * age * age
    }

    /// Detect contact against the resolved moving surface and locate it within
    /// the accepted physical step. The source stops at the surface on impact;
    /// position.z in the impact journal is the surface height, not sphere center.
    pub fn advance(
        &mut self,
        time: f64,
        surface: f32,
    ) -> bool {
        if self.impact.is_some() || time <= self.previous_time {
            return false;
        }
        self.z = self.altitude(time);
        if self.z - self.radius > surface {
            self.previous_time = time;
            self.previous_surface = surface;
            return false;
        }
        // Linear interpolation of the two physical surface samples avoids
        // backdating contact when a crest rises into a falling sphere.
        let height = |t: f64| {
            let fraction = ((t - self.previous_time) / (time - self.previous_time)) as f32;
            self.previous_surface + (surface - self.previous_surface) * fraction
        };
        let mut near = self.previous_time;
        let mut far = time;
        for _ in 0..20 {
            let middle = (near + far) * 0.5;
            if self.altitude(middle) - self.radius > height(middle) {
                near = middle;
            } else {
                far = middle;
            }
        }
        let impact = (near + far) * 0.5;
        let contact_surface = height(impact);
        let surface_speed = (surface - self.previous_surface) / (time - self.previous_time) as f32;
        self.impact_speed =
            (INITIAL_DOWN_SPEED + GRAVITY * (impact - self.born) as f32 + surface_speed).max(0.0);
        self.impact = Some(impact);
        self.z = contact_surface;
        true
    }

    pub fn retained(
        &self,
        time: f64,
    ) -> bool {
        self.impact
            .is_none_or(|impact| time - impact <= IMPACT_RETENTION)
    }

    pub fn export(
        &self,
        origin: [f64; 2],
    ) -> ARDrop {
        ARDrop {
            position: Float4([
                (self.world[0] - origin[0]) as f32,
                (self.world[1] - origin[1]) as f32,
                self.z,
                self.radius,
            ]),
            pigment: Float4([self.species[0], self.species[1], self.species[2], self.mass]),
            dynamics: Float4([
                (self.radius * 1.5).max(0.012),
                (self.radius * 1.3).max(0.012),
                self.impact.unwrap_or(0.0) as f32,
                0.12 + 0.16 * self.strength.sqrt(),
            ]),
            identity: [
                self.id as u32,
                (self.id >> 32) as u32,
                u32::from(self.impact.is_some()),
                0,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ballistic_contact_is_once_and_integral_does_not_depend_on_source_width() {
        let mut drop = Drop::new(1, [10.0, -7.0], 2.0, 0.02, 0.2, 1.5);
        let mass = drop.mass;
        assert!(!drop.advance(2.1, 0.02));
        assert!(drop.z > 0.02 + drop.radius);
        assert!(drop.advance(2.4, 0.02));
        assert!(!drop.advance(2.5, 0.02));
        assert_eq!(drop.z, 0.02);
        assert_eq!(drop.mass, mass);
        let expected_speed = (INITIAL_DOWN_SPEED.powi(2) + 2.0 * GRAVITY * FALL_HEIGHT).sqrt();
        assert!((drop.impact_speed - expected_speed).abs() < 1e-5);
        assert!((mass - 0.0012 * 1.5_f32.powi(3)).abs() < 1e-7);
        // Three independently rounded f32 divisions may sum one ULP above 1.
        assert!((drop.species.iter().sum::<f32>() - 1.0).abs() <= 2.0 * f32::EPSILON);
    }

    #[test]
    fn rebasing_changes_local_coordinates_without_moving_the_source() {
        let drop = Drop::new(0x1_0000_0002, [10.0, -7.0], 0.0, 0.0, 0.2, 1.0);
        let a = drop.export([9.0, -8.0]);
        let b = drop.export([9.5, -8.25]);
        assert_eq!(a.position.0[..2], [1.0, 1.0]);
        assert_eq!(b.position.0[..2], [0.5, 1.25]);
        assert_eq!(a.identity, [2, 1, 0, 0]);
        assert_eq!(drop.world, [10.0, -7.0]);
    }

    #[test]
    fn rising_surface_contact_stays_inside_the_latest_step() {
        let mut drop = Drop::new(1, [0.0; 2], 0.0, 0.0, 0.2, 1.0);
        assert!(!drop.advance(0.2, 0.0));
        assert!(drop.advance(0.21, 0.5));
        assert!((0.2..=0.21).contains(&drop.impact.unwrap()));
        assert!(drop.z > 0.0 && drop.z < 0.5);
    }

    #[test]
    fn native_source_loading_retains_the_volume_law_at_all_control_sizes() {
        for size in [0.25_f32, 1.0, 1.5, 3.0] {
            let drop = Drop::new(1, [0.0; 2], 0.0, 0.0, 0.2, size);
            assert!((drop.mass / size.powi(3) - 0.0012).abs() < 3e-10);
            assert_eq!(drop.export([0.0; 2]).pigment.0[3], drop.mass);
        }
    }
}
