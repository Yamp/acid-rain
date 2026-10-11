use std::time::Duration;

/// Camera time stays independent of the slower, elastic surface evolution.
pub const DEFAULT_SPEED: f32 = 2.4;
pub const SURFACE_TIME_SCALE: f32 = 0.25;

pub fn surface_delta(scene_delta: Duration) -> Duration { scene_delta.mul_f32(SURFACE_TIME_SCALE) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_moves_slowly_without_slowing_the_camera_clock() {
        let scene_delta = Duration::from_secs(1);
        assert!(surface_delta(scene_delta) < scene_delta / 3);
        assert_eq!(scene_delta, Duration::from_secs(1));
    }

    #[test]
    fn surface_progress_does_not_depend_on_render_cadence() {
        let at_30 = surface_delta(Duration::from_secs_f64(1.0 / 30.0)) * 30;
        let at_60 = surface_delta(Duration::from_secs_f64(1.0 / 60.0)) * 60;
        assert!((at_30.as_secs_f64() - at_60.as_secs_f64()).abs() < 1e-6);
        assert_eq!(surface_delta(Duration::ZERO), Duration::ZERO);
    }
}
