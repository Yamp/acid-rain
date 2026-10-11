//! Slowly evolving worlds shared by the sky, optics, and camera.

use clap::ValueEnum;

pub type Vec3 = [f32; 3];

pub const WORLD_DURATION: f32 = 90.0;
pub const TRANSITION_DURATION: f32 = 30.0;
pub const DEFAULT_INTENSITY: f32 = 4.0;
pub const MAX_INTENSITY: f32 = 4.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum SceneKind {
    #[default]
    Journey,
    Acid,
    Aurora,
    Storm,
    Abyss,
    Lava,
    Prism,
}

impl SceneKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Journey => "journey",
            Self::Acid => "acid",
            Self::Aurora => "aurora",
            Self::Storm => "storm",
            Self::Abyss => "abyss",
            Self::Lava => "lava",
            Self::Prism => "prism",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Journey => Self::Acid,
            Self::Acid => Self::Aurora,
            Self::Aurora => Self::Storm,
            Self::Storm => Self::Abyss,
            Self::Abyss => Self::Lava,
            Self::Lava => Self::Prism,
            Self::Prism => Self::Journey,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SceneFrame {
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gain: Vec3,
    pub hue_angle: f32,
    pub chromatic_aberration: f32,
    pub camera_roll: f32,
    pub camera_elevation: f32,
    pub fov: f32,
    pub light_dir: Vec3,
    pub fog_density: f32,
    pub interference_strength: f32,
    pub interference_phase: f32,
    pub absorption_scale: f32,
    pub(crate) sky_a: SceneKind,
    pub(crate) sky_b: SceneKind,
    pub(crate) sky_mix: f32,
    pub(crate) sky_rotation: [f32; 2],
    pub(crate) sky_phase: [f32; 4],
    pub(crate) sky_energy: f32,
    /// Artistic spatial light complexity; independent of radiance and chroma.
    pub(crate) light_complexity: f32,
    pub(crate) storm_glow: f32,
}

impl SceneFrame {
    pub fn new(
        kind: SceneKind,
        elapsed: f32,
        intensity: f32,
    ) -> Self {
        let elapsed = if elapsed.is_finite() { elapsed } else { 0.0 };
        let intensity = if intensity.is_finite() {
            intensity.clamp(0.0, MAX_INTENSITY)
        } else {
            DEFAULT_INTENSITY
        };
        // Extra dose builds optical structure and layered perception. Display
        // chroma and emitted energy stop growing at dose two; the gamut shoulder
        // keeps the stronger pigment grade soft without changing optical mass.
        let color_dose = intensity.min(2.0);
        let material_time = elapsed * 0.15;
        let (first, second, blend) = if kind == SceneKind::Journey {
            const WORLDS: [SceneKind; 6] = [
                SceneKind::Acid,
                SceneKind::Aurora,
                SceneKind::Storm,
                SceneKind::Abyss,
                SceneKind::Lava,
                SceneKind::Prism,
            ];
            let phase = elapsed.rem_euclid(WORLD_DURATION * WORLDS.len() as f32) / WORLD_DURATION;
            let index = phase.floor() as usize % WORLDS.len();
            let transition = ((phase.fract() * WORLD_DURATION
                - (WORLD_DURATION - TRANSITION_DURATION))
                / TRANSITION_DURATION)
                .clamp(0.0, 1.0);
            // Zero velocity at both ends prevents a visible gear change.
            let blend = transition * transition * (3.0 - 2.0 * transition);
            (WORLDS[index], WORLDS[(index + 1) % WORLDS.len()], blend)
        } else {
            (kind, kind, 0.0)
        };
        let a = Self::preset(first);
        let b = Self::preset(second);
        let lerp = |a: f32, b: f32| a + (b - a) * blend;
        let rgb = |a: Vec3, b: Vec3| std::array::from_fn(|i| lerp(a[i], b[i]));
        let (sway, _) = (elapsed * 0.024).sin_cos();
        let (breath, _) = (elapsed * 0.021).sin_cos();
        let chroma = (material_time * 0.061).sin();
        let (light_y, light_x) = (material_time * 0.031).sin_cos();
        let light = [light_x * 0.35, light_y * 0.35, 0.86];
        let inv_length =
            1.0 / (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt();
        let (rotation_sin, rotation_cos) = (material_time * 0.027).sin_cos();
        let lightning =
            ((material_time * 0.39).sin() * (material_time * 0.117 + 1.2).sin()).max(0.0);
        Self {
            interference_strength: lerp(a.interference_strength, b.interference_strength)
                * intensity,
            interference_phase: material_time * 0.31,
            absorption_scale: lerp(a.absorption_scale, b.absorption_scale),
            exposure: lerp(a.exposure, b.exposure),
            contrast: lerp(a.contrast, b.contrast),
            saturation: lerp(a.saturation, b.saturation) * (0.80 + color_dose * 0.62),
            gain: rgb(a.gain, b.gain),
            hue_angle: lerp(a.hue_angle, b.hue_angle) * chroma * intensity,
            chromatic_aberration: lerp(a.chromatic_aberration, b.chromatic_aberration) * intensity,
            // An unwrapped angle maintains a full, steady rotation through every world.
            camera_roll: elapsed * 4.0_f32.to_radians()
                + lerp(a.camera_roll, b.camera_roll) * sway * 0.10,
            camera_elevation: lerp(a.camera_elevation, b.camera_elevation) + breath * 0.070,
            fov: lerp(a.fov, b.fov) + breath * 0.045,
            light_dir: light.map(|v| v * inv_length),
            fog_density: lerp(a.fog_density, b.fog_density),
            sky_a: first,
            sky_b: second,
            sky_mix: blend,
            sky_rotation: [rotation_cos, rotation_sin],
            sky_phase: [
                material_time * 0.13,
                material_time * -0.09,
                material_time * 0.057,
                material_time * 0.031,
            ],
            sky_energy: 0.75 + color_dose * 0.25,
            light_complexity: intensity / MAX_INTENSITY,
            // A slow, rounded sheet-lightning glow rather than a strobe.
            storm_glow: lightning * lightning * lightning * 0.25,
        }
    }

    fn preset(kind: SceneKind) -> Self {
        let mut frame = Self {
            interference_strength: 0.55,
            interference_phase: 0.0,
            absorption_scale: 0.20,
            exposure: 0.95,
            contrast: 1.0,
            saturation: 0.86,
            gain: [1.0; 3],
            hue_angle: 0.025,
            chromatic_aberration: 0.0009,
            camera_roll: 0.70,
            camera_elevation: 70.0_f32.to_radians(),
            fov: 61.0_f32.to_radians(),
            light_dir: [0.0, 0.0, 1.0],
            // Clear-air extinction per metre; water extinction is integrated separately.
            fog_density: 0.00065,
            sky_a: kind,
            sky_b: kind,
            sky_mix: 0.0,
            sky_rotation: [1.0, 0.0],
            sky_phase: [0.0; 4],
            sky_energy: 1.0,
            light_complexity: 0.0,
            storm_glow: 0.0,
        };
        match kind {
            SceneKind::Journey | SceneKind::Acid => {},
            SceneKind::Aurora => {
                frame.absorption_scale = 0.18;
                frame.interference_strength = 0.40;
                frame.saturation = 0.80;
                frame.chromatic_aberration = 0.0005;
                frame.camera_roll = 0.08;
                frame.camera_elevation = 64.0_f32.to_radians();
                frame.fog_density = 0.0006;
            },
            SceneKind::Storm => {
                frame.absorption_scale = 0.26;
                frame.interference_strength = 0.20;
                frame.exposure = 0.90;
                frame.saturation = 0.74;
                frame.hue_angle = 0.0;
                frame.chromatic_aberration = 0.0002;
                frame.camera_roll = 0.045;
                frame.camera_elevation = 65.0_f32.to_radians();
                frame.fov = 58.0_f32.to_radians();
                frame.fog_density = 0.0010;
            },
            SceneKind::Abyss => {
                frame.absorption_scale = 0.30;
                frame.interference_strength = 0.50;
                frame.exposure = 0.92;
                frame.contrast = 1.025;
                frame.saturation = 0.84;
                frame.chromatic_aberration = 0.0007;
                frame.camera_roll = 0.35;
                frame.camera_elevation = 75.0_f32.to_radians();
                frame.fov = 64.0_f32.to_radians();
                frame.fog_density = 0.00085;
            },
            SceneKind::Lava => {
                frame.absorption_scale = 0.24;
                frame.interference_strength = 0.55;
                frame.exposure = 0.94;
                frame.saturation = 0.88;
                frame.chromatic_aberration = 0.0007;
                frame.camera_roll = 0.45;
                frame.camera_elevation = 72.0_f32.to_radians();
                frame.fog_density = 0.0007;
            },
            SceneKind::Prism => {
                frame.absorption_scale = 0.16;
                frame.interference_strength = 0.80;
                frame.exposure = 0.95;
                frame.saturation = 0.88;
                frame.contrast = 1.02;
                frame.hue_angle = 0.04;
                frame.chromatic_aberration = 0.0012;
                frame.camera_roll = 1.15;
                frame.camera_elevation = 59.0_f32.to_radians();
                frame.fov = 66.0_f32.to_radians();
                frame.fog_density = 0.00055;
            },
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_journey_and_manual_cycle_cover_every_world() {
        assert_eq!(SceneKind::default(), SceneKind::Journey);
        let names = [
            "journey", "acid", "aurora", "storm", "abyss", "lava", "prism",
        ];
        let mut scene = SceneKind::default();
        for name in names {
            assert_eq!(scene.name(), name);
            scene = scene.next();
        }
        assert_eq!(scene, SceneKind::default());
    }

    #[test]
    fn every_world_keeps_rotating_through_full_turns_without_stopping() {
        for scene in SceneKind::value_variants() {
            let mut previous = SceneFrame::new(*scene, 0.0, 1.0).camera_roll;
            for second in 1..=(WORLD_DURATION * 6.0) as u32 {
                let current = SceneFrame::new(*scene, second as f32, 1.0).camera_roll;
                let advance = (current - previous).to_degrees();
                assert!(
                    (3.2..=4.8).contains(&advance),
                    "{scene:?} rotates {advance} degrees at {second}s"
                );
                previous = current;
            }
            assert!(previous > std::f32::consts::TAU * 4.0);
        }
    }

    #[test]
    fn optical_materials_evolve_more_slowly_than_camera_motion() {
        let before = SceneFrame::new(SceneKind::Acid, 10.0, 1.0);
        let after = SceneFrame::new(SceneKind::Acid, 11.0, 1.0);
        assert!((after.sky_phase[0] - before.sky_phase[0]).abs() < 0.05);
        assert!((after.interference_phase - before.interference_phase).abs() < 0.10);
        assert!((after.camera_roll - before.camera_roll).abs() > 0.05);
    }

    #[test]
    fn scene_materials_keep_reflection_and_transmission_finite() {
        for scene in SceneKind::value_variants() {
            for time in [
                0.0,
                WORLD_DURATION - TRANSITION_DURATION,
                WORLD_DURATION,
                WORLD_DURATION * 2.0,
                WORLD_DURATION * 6.0 - 0.001,
                WORLD_DURATION * 6.0,
                100_000.0,
            ] {
                let frame = SceneFrame::new(*scene, time, 2.0);
                assert!(frame.absorption_scale.is_finite() && frame.absorption_scale > 0.0);
            }
        }
        let film = SceneFrame::new(SceneKind::Prism, 0.0, 1.0);
        let water = SceneFrame::new(SceneKind::Storm, 0.0, 1.0);
        assert!(film.interference_strength > water.interference_strength);
    }

    #[test]
    fn thin_film_interference_scales_without_changing_its_phase() {
        let subdued = SceneFrame::new(SceneKind::Prism, 17.0, 0.5);
        let vivid = SceneFrame::new(SceneKind::Prism, 17.0, 2.0);
        assert!(vivid.interference_strength > subdued.interference_strength);
        assert!(vivid.interference_strength <= 2.0);
        assert_eq!(subdued.interference_phase, vivid.interference_phase);
        let storm = SceneFrame::new(SceneKind::Storm, 17.0, 1.0);
        assert!(storm.interference_strength < subdued.interference_strength);
    }

    #[test]
    fn storm_and_acid_have_different_optics() {
        let storm = SceneFrame::new(SceneKind::Storm, 10.0, 1.0);
        let acid = SceneFrame::new(SceneKind::Acid, 10.0, 1.0);
        assert!(storm.saturation < acid.saturation);
        assert!(storm.chromatic_aberration < acid.chromatic_aberration);
    }

    #[test]
    fn journey_visits_all_worlds_without_parameter_jumps() {
        let worlds = [
            SceneKind::Acid,
            SceneKind::Aurora,
            SceneKind::Storm,
            SceneKind::Abyss,
            SceneKind::Lava,
            SceneKind::Prism,
        ];
        for (index, world) in worlds.into_iter().enumerate() {
            let time = index as f32 * WORLD_DURATION + TRANSITION_DURATION * 0.4;
            let journey = SceneFrame::new(SceneKind::Journey, time, 1.0);
            let selected = SceneFrame::new(world, time, 1.0);
            assert_eq!(journey.absorption_scale, selected.absorption_scale);
            for boundary in [
                index as f32 * WORLD_DURATION,
                (index + 1) as f32 * WORLD_DURATION - TRANSITION_DURATION,
            ] {
                let before = SceneFrame::new(SceneKind::Journey, boundary - 0.001, 1.0);
                let after = SceneFrame::new(SceneKind::Journey, boundary + 0.001, 1.0);
                assert!((before.saturation - after.saturation).abs() < 0.001);
                assert!((before.camera_roll - after.camera_roll).abs() < 0.001);
            }
        }
    }

    #[test]
    fn intensity_does_not_increase_camera_motion_or_destroy_finiteness() {
        for scene in SceneKind::value_variants() {
            let normal = SceneFrame::new(*scene, 37.0, 1.0);
            let intense = SceneFrame::new(*scene, 37.0, 4.0);
            assert_eq!(normal.camera_roll, intense.camera_roll);
            assert_eq!(normal.camera_elevation, intense.camera_elevation);
            assert_eq!(normal.fov, intense.fov);
            assert_eq!(normal.sky_phase, intense.sky_phase);
            assert!(intense.chromatic_aberration <= 0.014);
            let light_length: f32 = intense.light_dir.iter().map(|v| v * v).sum();
            assert!((light_length - 1.0).abs() < 0.00001);
        }
    }

    #[test]
    fn invalid_intensity_uses_the_new_default_without_increasing_color_energy() {
        let normal = SceneFrame::new(SceneKind::Prism, 17.0, 2.0);
        let intense = SceneFrame::new(SceneKind::Prism, 17.0, 4.0);
        assert!(intense.saturation <= normal.saturation);
        assert!(intense.sky_energy <= normal.sky_energy);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid = SceneFrame::new(SceneKind::Prism, 17.0, value);
            assert_eq!(invalid.saturation, intense.saturation);
            assert_eq!(invalid.chromatic_aberration, intense.chromatic_aberration);
        }
        let bounded = SceneFrame::new(SceneKind::Prism, 17.0, f32::MAX);
        assert_eq!(bounded.chromatic_aberration, intense.chromatic_aberration);
    }
}
