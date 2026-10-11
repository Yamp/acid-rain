use std::time::{
    Duration,
    Instant,
};

use crate::term::RenderQuality;

/// Changes quality only after sustained pressure or sustained spare capacity.
pub struct FrameBudget {
    quality: RenderQuality,
    pace_level: u8,
    average: f64,
    overloaded: u32,
    comfortable: u32,
}

impl FrameBudget {
    pub fn new() -> Self {
        Self {
            quality: RenderQuality::Balanced,
            pace_level: 0,
            average: 0.0,
            overloaded: 0,
            comfortable: 0,
        }
    }

    pub fn quality(&self) -> RenderQuality { self.quality }

    pub fn frame_interval(
        &self,
        requested: Duration,
    ) -> Duration {
        requested.mul_f64(1.0 + f64::from(self.pace_level) * 0.5)
    }

    pub fn observe(
        &mut self,
        work: Duration,
        target: Duration,
    ) -> RenderQuality {
        let ratio = (work.as_secs_f64() / target.as_secs_f64().max(0.001)).min(2.0);
        self.average = if self.average == 0.0 {
            ratio
        } else {
            self.average * 0.9 + ratio * 0.1
        };
        let recovery_threshold = 0.40;
        if self.average > 0.90 && ratio > 0.90 {
            self.overloaded += 1;
            self.comfortable = 0;
        } else if self.average < recovery_threshold {
            self.comfortable += 1;
            self.overloaded = 0;
        } else {
            self.overloaded = 0;
            self.comfortable = 0;
        }
        if self.overloaded >= 8 {
            if self.quality == RenderQuality::High {
                self.quality = RenderQuality::Balanced;
            } else {
                // Terminal cells are already coarse pixels: preserve detail
                // under load and relax the requested cadence (60 -> 40 -> 30).
                self.pace_level = (self.pace_level + 1).min(2);
            }
            self.overloaded = 0;
            self.average = 0.7;
        } else if self.comfortable >= 120 {
            if self.pace_level > 0 {
                self.pace_level -= 1;
            } else {
                self.quality = RenderQuality::High;
            }
            self.comfortable = 0;
            self.average = 0.7;
        }
        self.quality
    }
}

pub fn simulation_size(
    width: u16,
    height: u16,
) -> (u16, u16) {
    (width.clamp(2, 240), height.clamp(2, 120))
}

pub fn next_frame_deadline(
    previous: Instant,
    started: Instant,
    finished: Instant,
    interval: Duration,
) -> Instant {
    // An input-triggered early redraw starts a fresh interval. Work that has
    // already overrun the deadline must not be followed by another full sleep.
    (previous.min(started) + interval).max(finished)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_load_preserves_full_resolution_and_relaxes_the_frame_cap() {
        let mut budget = FrameBudget::new();
        let target = Duration::from_millis(16);
        assert_eq!(
            budget.observe(Duration::from_millis(80), target),
            RenderQuality::Balanced
        );
        for _ in 0..20 {
            budget.observe(Duration::from_millis(24), target);
        }
        assert_eq!(budget.quality(), RenderQuality::Balanced);
        assert!(budget.frame_interval(target) > target);
    }

    #[test]
    fn spare_capacity_recovers_quality_slowly() {
        let mut budget = FrameBudget::new();
        for _ in 0..89 {
            budget.observe(Duration::from_millis(1), Duration::from_millis(16));
        }
        assert_eq!(budget.quality(), RenderQuality::Balanced);
        for _ in 0..100 {
            budget.observe(Duration::from_millis(1), Duration::from_millis(16));
        }
        assert_eq!(budget.quality(), RenderQuality::High);
    }

    #[test]
    fn near_budget_does_not_oscillate() {
        let mut budget = FrameBudget::new();
        for i in 0..500 {
            budget.observe(Duration::from_millis(10 + i % 3), Duration::from_millis(16));
        }
        assert_eq!(budget.quality(), RenderQuality::Balanced);
    }

    #[test]
    fn prolonged_overload_never_selects_blocky_resolution() {
        let mut budget = FrameBudget::new();
        let requested = Duration::from_secs_f64(1.0 / 60.0);
        for _ in 0..500 {
            let target = budget.frame_interval(requested);
            assert_ne!(
                budget.observe(Duration::from_millis(50), target),
                RenderQuality::Low
            );
        }
        assert_eq!(budget.frame_interval(requested), requested * 2);
    }

    #[test]
    fn spare_capacity_restores_cadence_before_supersampling() {
        let mut budget = FrameBudget::new();
        budget.pace_level = 2;
        let requested = Duration::from_millis(16);
        for _ in 0..120 {
            budget.observe(Duration::from_millis(1), requested);
        }
        assert_eq!(budget.quality(), RenderQuality::Balanced);
        assert_eq!(budget.frame_interval(requested), Duration::from_millis(24));
        for _ in 0..260 {
            budget.observe(Duration::from_millis(1), requested);
        }
        assert_eq!(budget.quality(), RenderQuality::High);
        assert_eq!(budget.frame_interval(requested), requested);
    }

    #[test]
    fn a_single_stall_does_not_poison_later_frames() {
        let mut budget = FrameBudget::new();
        let target = Duration::from_millis(16);
        budget.observe(Duration::from_secs(2), target);
        for _ in 0..20 {
            assert_eq!(
                budget.observe(Duration::from_millis(8), target),
                RenderQuality::Balanced
            );
        }
    }

    #[test]
    fn early_input_redraws_cannot_accumulate_a_future_deadline() {
        let now = Instant::now();
        let interval = Duration::from_millis(16);
        assert_eq!(
            next_frame_deadline(
                now + interval,
                now,
                now + Duration::from_millis(2),
                interval
            ),
            now + interval
        );
    }

    #[test]
    fn over_budget_frames_do_not_add_an_extra_sleep() {
        let now = Instant::now();
        let finished = now + Duration::from_millis(24);
        assert_eq!(
            next_frame_deadline(now, now, finished, Duration::from_millis(16)),
            finished
        );
    }

    #[test]
    fn simulation_allocation_is_bounded_and_singletons_are_safe() {
        assert_eq!(simulation_size(0, 1), (2, 2));
        assert_eq!(simulation_size(u16::MAX, u16::MAX), (240, 120));
        assert_eq!(simulation_size(120, 40), (120, 40));
    }
}
