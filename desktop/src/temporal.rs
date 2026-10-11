//! Display clocks and radiometric/volume scheduling, independent of GPU backend.

use std::ops::Range;

use crate::quality::RenderExtent;

#[derive(Clone, Debug, Default)]
pub struct PresentationClock {
    value: f64,
    last_wall: Option<f64>,
    was_held: bool,
}
impl PresentationClock {
    pub fn value(&self) -> f64 { self.value }

    pub fn reanchor(
        &mut self,
        wall_time: f64,
    ) {
        if self.last_wall.is_some() {
            self.last_wall = Some(wall_time);
            self.was_held = true;
        }
    }

    /// All times share one monotonic epoch. `sampled_at` is the physics step's
    /// start, not the later publication time after variably expensive work.
    pub fn sample(
        &mut self,
        scene_time: f64,
        sampled_at: f64,
        speed: f64,
        now: f64,
        held: bool,
    ) -> f64 {
        let raw_age = (now - sampled_at).max(0.0);
        let age = if self.was_held { 0.0 } else { raw_age.min(0.1) };
        let target = scene_time + if held { 0.0 } else { age * speed };
        let Some(previous_wall) = self.last_wall else {
            self.value = target;
            self.last_wall = Some(now);
            self.was_held = held;
            return self.value;
        };
        let raw_delta = (now - previous_wall).max(0.0);
        self.last_wall = Some(now);
        self.was_held = held;
        // No callback arrives while hidden: do not integrate that unseen time.
        if held || raw_delta <= 0.0 || raw_delta > 0.25 {
            return self.value;
        }
        let delta = raw_delta.min(0.1);
        let nominal_advance = delta * speed;
        let predicted = self.value + nominal_advance;
        let alpha = -(-delta / 0.18).exp_m1();
        let correction = (target - predicted) * alpha;
        let advance = (nominal_advance + correction)
            .min(nominal_advance * 1.5)
            .max(nominal_advance * 0.25);
        let lead_limit = scene_time + 0.1 * speed;
        if raw_age >= 0.1 || self.value <= lead_limit {
            self.value = self.value.max(lead_limit.min(self.value + advance));
        } else {
            // Reducing speed shrinks the allowed lead. Converge by slowing,
            // rather than reversing the camera or freezing it for seconds.
            self.value += advance;
        }
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetinaHistoryStep {
    pub time: f32,
    pub alphas: [f32; 3],
    pub update: bool,
    pub seed: bool,
}
#[derive(Clone, Debug, Default)]
pub struct RetinaHistoryPolicy {
    valid: bool,
    time: f64,
}
impl RetinaHistoryPolicy {
    pub const HALF_LIVES: [f64; 3] = [0.10, 0.35, 0.80];

    pub fn invalidate(&mut self) { self.valid = false; }

    pub fn alpha(
        delta: f64,
        half_life: f64,
    ) -> f64 {
        -(-std::f64::consts::LN_2 * delta.max(0.0) / half_life).exp_m1()
    }

    pub fn extent(
        width: u32,
        height: u32,
    ) -> RenderExtent {
        crate::quality::RenderPolicy::extent(
            width.div_ceil(2),
            height.div_ceil(2),
            crate::quality::RenderQuality::Auto,
            350_000,
        )
    }

    pub fn advance(
        &mut self,
        delta: f64,
        held: bool,
    ) -> RetinaHistoryStep {
        let gap = !held && delta > 0.25;
        if gap {
            self.valid = false;
        }
        let active_delta = if held || gap { 0.0 } else { delta.max(0.0) };
        self.time += active_delta;
        let seed = !self.valid;
        let update = seed || active_delta > 0.0;
        let alphas = if seed {
            [1.0; 3]
        } else {
            Self::HALF_LIVES.map(|half_life| Self::alpha(active_delta, half_life) as f32)
        };
        if update {
            self.valid = true;
        }
        RetinaHistoryStep {
            time: self.time as f32,
            alphas,
            update,
            seed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudStep {
    pub delta: f64,
    pub shift: [i32; 2],
    pub clear: bool,
}
#[derive(Clone, Debug, Default)]
pub struct CloudClock {
    time: Option<f64>,
    origin: [f64; 2],
    revision: Option<u64>,
}
impl CloudClock {
    /// The volume consumes physical time, never the number of display callbacks.
    pub fn consume(
        &mut self,
        time: f64,
        origin: [f64; 2],
        revision: u64,
    ) -> CloudStep {
        let clear = self.revision != Some(revision);
        let elapsed = if clear {
            0.0
        } else {
            (time - self.time.unwrap_or(time)).max(0.0)
        };
        let shift = if clear {
            [0; 2]
        } else {
            [
                ((origin[0] - self.origin[0]) * 48.0)
                    .round()
                    .clamp(-96.0, 96.0) as i32,
                ((origin[1] - self.origin[1]) * 32.0)
                    .round()
                    .clamp(-64.0, 64.0) as i32,
            ]
        };
        self.time = Some(time);
        self.origin = origin;
        self.revision = Some(revision);
        CloudStep {
            delta: elapsed.min(0.128),
            shift,
            clear,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudTransportPlan {
    pub substep_count: usize,
    pub substep_delta: f64,
    pub source_order: Vec<usize>,
    pub injections: Vec<Range<usize>>,
}
impl CloudTransportPlan {
    /// Inject at the first CFL boundary at or after contact. A source can be
    /// one substep late, but never advects or bleaches before impact.
    pub fn new(
        delta: f64,
        end_time: f64,
        impact_times: &[f64],
    ) -> Self {
        assert!((0.0..=0.128).contains(&delta));
        assert!(end_time.is_finite() && impact_times.iter().all(|time| time.is_finite()));
        let substep_count = if delta > 0.0 {
            (delta / 0.008).ceil() as usize
        } else {
            0
        };
        let substep_delta = if substep_count > 0 {
            delta / substep_count as f64
        } else {
            0.0
        };
        let mut source_order: Vec<usize> = (0..impact_times.len()).collect();
        source_order.sort_by(|a, b| {
            impact_times[*a]
                .partial_cmp(&impact_times[*b])
                .unwrap()
                .then_with(|| a.cmp(b))
        });
        let mut counts = vec![0; substep_count + 1];
        let start = end_time - delta;
        for source in &source_order {
            let bucket = if substep_count == 0 {
                0
            } else {
                ((impact_times[*source] - start) / substep_delta)
                    .ceil()
                    .clamp(0.0, substep_count as f64) as usize
            };
            counts[bucket] += 1;
        }
        let mut cursor = 0;
        let injections = counts
            .into_iter()
            .map(|count| {
                let range = cursor..cursor + count;
                cursor += count;
                range
            })
            .collect();
        Self {
            substep_count,
            substep_delta,
            source_order,
            injections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_matches_native_publication_jitter_and_cadence_fixtures() {
        for cadence in [1.0 / 30.0, 1.0 / 60.0, 1.0 / 120.0] {
            let mut clock = PresentationClock::default();
            let mut previous = -1.0;
            for i in 0..=240 {
                let now = f64::from(i) * cadence;
                let sampled_at = (((now - 0.008) * 60.0).floor() / 60.0).max(0.0);
                let scene = clock.sample(sampled_at * 2.4, sampled_at, 2.4, now, false);
                assert!(scene >= previous && (scene - now * 2.4).abs() < 1e-8);
                previous = scene;
            }
        }
        let mut clock = PresentationClock::default();
        clock.sample(10.0, 0.0, 2.4, 0.0, false);
        let before = clock.sample(10.0, 0.0, 2.4, 0.025, false);
        let after = clock.sample(10.0, 0.0, 2.4, 0.026, false);
        assert!(after >= before && (after - before - 0.0024).abs() < 1e-8);
        let held = clock.sample(10.04, 0.026, 2.4, 0.05, true);
        for now in [0.07, 0.2, 1.0, 100.0] {
            assert_eq!(clock.sample(10.04, 0.026, 2.4, now, true), held);
        }
        let resumed = clock.sample(10.04, 100.0, 0.4, 100.01, false);
        assert!(resumed >= held && resumed - held < 0.02);
        for i in 1..=1000 {
            clock.sample(10.04, 100.0, 0.4, 100.01 + f64::from(i) / 60.0, false);
        }
        assert!(clock.value() <= 10.04 + 0.4 * 0.1 + 1e-9);
    }

    #[test]
    fn clock_matches_native_hidden_reanchor_and_speed_reduction_fixtures() {
        let mut occluded = PresentationClock::default();
        let visible = occluded.sample(2.0, 0.0, 2.4, 0.015, false);
        assert_eq!(occluded.sample(2.0, 0.0, 2.4, 90.0, false), visible);
        let mut hidden = PresentationClock::default();
        let before = hidden.sample(2.0, 0.0, 2.4, 0.01, false);
        hidden.reanchor(0.12);
        let again = hidden.sample(2.0, 0.0, 2.4, 0.121, false);
        assert!(again >= before && again - before <= 0.003);
        let mut slowed = PresentationClock::default();
        let mut prior = slowed.sample(10.0, 0.0, 3.0, 0.1, false);
        assert!((prior - 10.3).abs() < 1e-12);
        let mut error = prior - 10.0;
        for frame in 1..=240 {
            let now = 0.1 + f64::from(frame) / 60.0;
            let scene = 10.0 + f64::from(frame) / 60.0 * 0.1;
            let current = slowed.sample(scene, now, 0.1, now, false);
            assert!(current > prior && current - prior <= 0.1 / 60.0 * 1.5 + 1e-12);
            assert!(current - scene <= error + 1e-12);
            prior = current;
            error = current - scene;
        }
        assert!(error < 0.02);
    }

    #[test]
    fn retinal_history_matches_native_half_lives_hold_reset_gap_and_memory_budget() {
        for half_life in RetinaHistoryPolicy::HALF_LIVES {
            for fps in [30, 60, 120] {
                let delta = 1.0 / f64::from(fps);
                let mut value = 0.0;
                for _ in 0..fps {
                    value += (1.0 - value) * RetinaHistoryPolicy::alpha(delta, half_life);
                }
                assert!((value - (1.0 - 2.0_f64.powf(-1.0 / half_life))).abs() < 1e-12);
            }
        }
        let mut retina = RetinaHistoryPolicy::default();
        let seed = retina.advance(1.0 / 60.0, false);
        assert!(seed.update && seed.seed && seed.alphas == [1.0; 3]);
        let active = retina.advance(0.02, false);
        assert!(active.update && !active.seed);
        for delta in [0.016, 0.2, 40.0] {
            let held = retina.advance(delta, true);
            assert!(!held.update && held.time == active.time && !held.seed);
        }
        retina.invalidate();
        let reset = retina.advance(10.0, true);
        assert!(reset.seed && reset.update && reset.time == active.time);
        let gap = retina.advance(1.0, false);
        assert!(gap.seed && gap.update && gap.time == active.time);
        assert!(!retina.advance(0.0, false).update);
        let wide = RetinaHistoryPolicy::extent(8192, 4096);
        assert!(wide.width * wide.height <= 350_000);
        assert!((f64::from(wide.width) / f64::from(wide.height) - 2.0).abs() < 0.01);
    }

    #[test]
    fn cloud_clock_matches_native_physical_time_rebase_and_bounded_recovery() {
        let mut clock = CloudClock::default();
        let first = clock.consume(0.0, [-1.0; 2], 0);
        assert!(first.clear && first.delta == 0.0);
        let origin = [-1.0 + 2.0 / 96.0, -1.0 - 2.0 / 64.0];
        let moving = clock.consume(0.03, origin, 0);
        assert!((moving.delta - 0.03).abs() < 1e-8 && moving.shift == [1, -1] && !moving.clear);
        let held = clock.consume(0.03, origin, 0);
        assert!(held.delta == 0.0 && held.shift == [0; 2] && !held.clear);
        let reset = clock.consume(0.03, [0.0; 2], 1);
        assert!(reset.clear && reset.delta == 0.0);
        assert!(clock.consume(60.0, [0.0; 2], 1).delta <= 0.128);
    }

    #[test]
    fn cloud_sources_match_native_cfl_contact_timing_and_order() {
        let plan = CloudTransportPlan::new(0.06, 0.30, &[0.29]);
        let birth_step = plan
            .injections
            .iter()
            .position(|range| !range.is_empty())
            .unwrap();
        assert_eq!(plan.substep_count, 8);
        assert_eq!(birth_step, 7);
        let birth = 0.24 + birth_step as f64 * plan.substep_delta;
        assert!(birth >= 0.29 && birth - 0.29 <= 0.008);
        let reordered = CloudTransportPlan::new(0.06, 0.30, &[0.299, 0.241, 0.23, 0.29]);
        assert_eq!(reordered.source_order, [2, 1, 3, 0]);
        assert_eq!(reordered.injections[0], 0..1);
        let paused = CloudTransportPlan::new(0.0, 0.30, &[0.29]);
        assert_eq!(paused.substep_count, 0);
        assert_eq!(paused.injections.len(), 1);
        assert_eq!(paused.injections[0], 0..1);
        let repeated = CloudTransportPlan::new(0.0, 0.30, &[]);
        assert_eq!(repeated.injections.len(), 1);
        assert_eq!(repeated.injections[0], 0..0);
        let mut random = 0xD09C10_u64;
        for _ in 0..300 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let delta = 0.001 + (random % 127_000) as f64 / 1_000_000.0;
            let start = 10.0 - delta;
            let mut times = Vec::new();
            for _ in 0..17 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                times.push(start + delta * (random % 10_000) as f64 / 10_000.0);
            }
            let plan = CloudTransportPlan::new(delta, 10.0, &times);
            assert!(plan.substep_count <= 16 && plan.substep_delta <= 0.008);
            let mut seen = std::collections::HashSet::new();
            for (bucket, range) in plan.injections.iter().enumerate() {
                let birth = start + bucket as f64 * plan.substep_delta;
                for source in range.clone() {
                    let original = plan.source_order[source];
                    assert!(seen.insert(original));
                    assert!(
                        birth + 1e-12 >= times[original]
                            && birth - times[original] <= 0.008 + 1e-12
                    );
                }
            }
            assert_eq!(seen.len(), times.len());
        }
    }
}
