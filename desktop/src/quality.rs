//! Pixel budgets and slow/fast-frame hysteresis shared by every GPU backend.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum RenderQuality {
    Auto,
    #[default]
    High,
    Balanced,
    Low,
    Native,
}

impl RenderQuality {
    pub const ALL: [Self; 5] = [
        Self::Auto,
        Self::High,
        Self::Balanced,
        Self::Low,
        Self::Native,
    ];

    pub fn name(self) -> &'static str {
        ["Auto", "High", "Balanced", "Low", "Native"][self as usize]
    }

    pub fn next(self) -> Self { Self::ALL[(self as usize + 1) % Self::ALL.len()] }

    pub fn shader_level(self) -> f32 {
        match self {
            Self::Low => 0.0,
            Self::Balanced => 1.0,
            _ => 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderExtent {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct RenderPolicy {
    automatic_pixel_budget: u32,
    slow_frames: u32,
    fast_frames: u32,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        Self {
            automatic_pixel_budget: 1_200_000,
            slow_frames: 0,
            fast_frames: 0,
        }
    }
}

impl RenderPolicy {
    pub fn automatic_pixel_budget(&self) -> u32 { self.automatic_pixel_budget }

    pub fn extent(
        width: u32,
        height: u32,
        quality: RenderQuality,
        automatic_pixel_budget: u32,
    ) -> RenderExtent {
        let width = width.max(1);
        let height = height.max(1);
        let pixels = f64::from(width) * f64::from(height);
        let budget = match quality {
            RenderQuality::Native => pixels,
            RenderQuality::High => 1_600_000.0,
            RenderQuality::Balanced => 900_000.0,
            RenderQuality::Low => 450_000.0,
            RenderQuality::Auto => f64::from(automatic_pixel_budget),
        };
        let scale = (budget / pixels).sqrt().min(1.0);
        RenderExtent {
            width: ((f64::from(width) * scale) as u32).max(1),
            height: ((f64::from(height) * scale) as u32).max(1),
        }
    }

    pub fn observe(
        &mut self,
        gpu_milliseconds: f64,
    ) {
        if !gpu_milliseconds.is_finite() || gpu_milliseconds <= 0.0 {
            return;
        }
        if gpu_milliseconds > 20.0 {
            self.slow_frames += 1;
            self.fast_frames = 0;
        } else if gpu_milliseconds < 11.0 {
            self.fast_frames += 1;
            self.slow_frames = 0;
        } else {
            self.slow_frames = 0;
            self.fast_frames = 0;
        }
        if self.slow_frames >= 30 {
            self.automatic_pixel_budget =
                ((f64::from(self.automatic_pixel_budget) * 0.8) as u32).max(350_000);
            self.slow_frames = 0;
        } else if self.fast_frames >= 180 {
            self.automatic_pixel_budget =
                ((f64::from(self.automatic_pixel_budget) * 1.1) as u32).min(1_600_000);
            self.fast_frames = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_quality_fixture_budgets_aspect_and_cycle() {
        let hd = RenderPolicy::extent(3840, 2160, RenderQuality::High, 1_200_000);
        assert!(u64::from(hd.width) * u64::from(hd.height) <= 1_600_000);
        assert!((f64::from(hd.width) / f64::from(hd.height) - 16.0 / 9.0).abs() < 0.01);
        for side in [0, 1] {
            assert_eq!(
                RenderPolicy::extent(side, side, RenderQuality::Low, 1_200_000),
                RenderExtent {
                    width: 1,
                    height: 1
                }
            );
        }
        assert_eq!(RenderQuality::High.next(), RenderQuality::Balanced);
        assert_eq!(RenderQuality::Low.next(), RenderQuality::Native);
        assert_eq!(RenderQuality::Native.next(), RenderQuality::Auto);
        assert_eq!(RenderQuality::Auto.next(), RenderQuality::High);
        assert_eq!(
            RenderPolicy::extent(3024, 1964, RenderQuality::Native, 1),
            RenderExtent {
                width: 3024,
                height: 1964
            }
        );
        assert_eq!(RenderQuality::Low.shader_level(), 0.0);
        assert_eq!(RenderQuality::Balanced.shader_level(), 1.0);
        assert_eq!(RenderQuality::High.shader_level(), 2.0);
    }

    #[test]
    fn native_adaptation_fixture_has_slow_hysteresis_and_bounded_budget() {
        let mut policy = RenderPolicy::default();
        for _ in 0..29 {
            policy.observe(40.0);
        }
        assert_eq!(policy.automatic_pixel_budget(), 1_200_000);
        policy.observe(40.0);
        assert_eq!(policy.automatic_pixel_budget(), 960_000);
        for _ in 0..30 {
            policy.observe(10.0);
        }
        assert_eq!(policy.automatic_pixel_budget(), 960_000);
        for _ in 0..150 {
            policy.observe(3.0);
        }
        assert_eq!(policy.automatic_pixel_budget(), 1_056_000);
        for _ in 0..3000 {
            policy.observe(40.0);
        }
        assert_eq!(policy.automatic_pixel_budget(), 350_000);
        for _ in 0..18000 {
            policy.observe(3.0);
        }
        assert_eq!(policy.automatic_pixel_budget(), 1_600_000);
        for value in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            policy.observe(value);
        }
        assert_eq!(policy.automatic_pixel_budget(), 1_600_000);
    }
}
