//! Camera travel across the streamed world, independent of view rotation.

const FORWARD_SPEED: f64 = 0.04;

#[derive(Debug, Clone)]
pub struct WorldTravel {
    enabled: bool,
    elapsed: f64,
}

impl Default for WorldTravel {
    fn default() -> Self {
        Self {
            enabled: true,
            elapsed: 0.0,
        }
    }
}

impl WorldTravel {
    /// Only active travel consumes time. Pauses, stationary numerical modes and
    /// the runtime toggle therefore resume at the held point without a catch-up.
    pub fn advance(
        &mut self,
        scene_dt: f32,
        active: bool,
    ) {
        if self.enabled && active && scene_dt.is_finite() && scene_dt > 0.0 {
            self.elapsed += f64::from(scene_dt);
        }
    }

    /// World metres. A monotone forward component never closes the route;
    /// bounded, long bends change direction independently of camera rotation.
    /// Evaluate the path from a double-precision clock rather than integrating
    /// velocities at the render cadence or accumulating world positions in f32.
    pub fn center(&self) -> [f64; 2] {
        let time = self.elapsed;
        [
            FORWARD_SPEED * time + 0.20 * (time / 95.0).sin(),
            0.32 * (time / 59.0).sin() + 0.18 * (1.0 - (time / 137.0).cos()),
        ]
    }

    pub fn enabled(&self) -> bool { self.enabled }

    pub fn toggle(&mut self) { self.enabled = !self.enabled; }
}

#[cfg(test)]
mod tests {
    use super::WorldTravel;

    #[test]
    fn journey_starts_enabled_at_the_world_origin() {
        let travel = WorldTravel::default();
        assert!(travel.enabled());
        assert_eq!(travel.center(), [0.0; 2]);
    }

    #[test]
    fn camera_travels_slowly_with_lateral_bends() {
        let mut travel = WorldTravel::default();
        travel.advance(60.0, true);
        let [x, y] = travel.center();
        assert!(
            (2.1..2.7).contains(&x),
            "travel after 60 scene seconds: {x}"
        );
        assert!(y.abs() > 0.10, "route should bend across the bed: {y}");
    }

    #[test]
    fn inactive_time_neither_moves_nor_accumulates_for_resume() {
        let mut travel = WorldTravel::default();
        travel.advance(23.0, true);
        let held = travel.center();
        travel.advance(10_000.0, false);
        assert_eq!(travel.center(), held);
        assert!(travel.enabled());
        travel.advance(1.0, true);
        let mut uninterrupted = WorldTravel::default();
        uninterrupted.advance(24.0, true);
        assert_eq!(travel.center(), uninterrupted.center());
        assert_ne!(travel.center(), held);
    }

    #[test]
    fn travel_toggle_preserves_position_and_route_phase() {
        let mut travel = WorldTravel::default();
        travel.advance(17.0, true);
        let held = travel.center();
        travel.toggle();
        assert!(!travel.enabled());
        assert_eq!(travel.center(), held);
        travel.advance(100.0, true);
        assert_eq!(travel.center(), held);
        travel.toggle();
        assert!(travel.enabled());
        assert_eq!(travel.center(), held);
        travel.advance(3.0, true);
        let mut uninterrupted = WorldTravel::default();
        uninterrupted.advance(20.0, true);
        assert_eq!(travel.center(), uninterrupted.center());
        assert_ne!(travel.center(), held);
    }

    #[test]
    fn invalid_and_zero_deltas_do_not_change_the_journey() {
        let mut travel = WorldTravel::default();
        travel.advance(7.0, true);
        let held = travel.center();
        for dt in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            travel.advance(dt, true);
            assert_eq!(travel.center(), held, "bad delta {dt}");
        }
        travel.advance(1.0, true);
        assert!(travel.center().into_iter().all(f64::is_finite));
        assert_ne!(travel.center(), held);
    }

    #[test]
    fn route_is_independent_of_frame_cadence() {
        let mut at_30 = WorldTravel::default();
        let mut at_60 = WorldTravel::default();
        for _ in 0..3600 {
            at_30.advance(1.0 / 30.0, true);
        }
        for _ in 0..7200 {
            at_60.advance(1.0 / 60.0, true);
        }
        assert_eq!(at_30.center(), at_60.center());
        assert!(at_30.center()[0] > 4.0);
    }

    #[test]
    fn rotation_period_does_not_close_the_journey() {
        let mut travel = WorldTravel::default();
        let mut previous = travel.center();
        for _ in 0..8 {
            travel.advance(180.0, true);
            let position = travel.center();
            assert!(position[0] - previous[0] > 6.5);
            previous = position;
        }
    }

    #[test]
    fn world_speed_has_no_jumps_at_bends() {
        let mut travel = WorldTravel::default();
        let dt = 1.0 / 60.0;
        let mut previous = travel.center();
        for _ in 0..43_200 {
            travel.advance(dt, true);
            let position = travel.center();
            let dx = (position[0] - previous[0]) / f64::from(dt);
            let dy = (position[1] - previous[1]) / f64::from(dt);
            assert!((0.035..0.045).contains(&dx), "forward speed: {dx}");
            assert!(dy.abs() < 0.01, "lateral speed: {dy}");
            previous = position;
        }
    }

    #[test]
    fn long_sessions_remain_finite_and_keep_small_steps() {
        let mut travel = WorldTravel::default();
        for _ in 0..10_000 {
            travel.advance(86_400.0, true);
        }
        let before = travel.center();
        assert!(before.into_iter().all(f64::is_finite));
        travel.advance(1.0 / 60.0, true);
        let after = travel.center();
        assert!(after.into_iter().all(f64::is_finite));
        assert!((0.0005..0.0008).contains(&(after[0] - before[0])));
    }
}
