//! Validated startup settings shared by interactive and deterministic capture paths.
use std::path::PathBuf;

use acid_rain_desktop::quality::RenderQuality;
use acid_rain_native_core::Simulation;
use clap::{
    Parser,
    ValueEnum,
};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Vulkan,
    Dx12,
    Metal,
}
impl Backend {
    pub fn backends(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::PRIMARY,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Metal => wgpu::Backends::METAL,
        }
    }
}
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum Quality {
    Auto,
    High,
    Balanced,
    Low,
    Native,
}
impl Quality {
    pub fn render(self) -> RenderQuality {
        match self {
            Self::Auto => RenderQuality::Auto,
            Self::High => RenderQuality::High,
            Self::Balanced => RenderQuality::Balanced,
            Self::Low => RenderQuality::Low,
            Self::Native => RenderQuality::Native,
        }
    }
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Mode {
    Classic,
    Drain,
    Rapids,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Scene {
    Journey,
    Acid,
    Aurora,
    Storm,
    Abyss,
    Lava,
    Prism,
}

#[derive(Clone, Debug, Parser)]
#[command(
    version,
    about = "Flowing water, drifting colour, impossible light.",
    after_help = "In the window: H opens all controls. Space pauses. Click the water or press O \
                  to add a drop. N/M change the subjective effects; Shift+N restores them."
)]
pub struct Options {
    /// Render without opening a window (requires a graphics adapter).
    #[arg(long)]
    pub headless: bool,
    /// Number of measured frames after warm-up in headless mode.
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u32).range(1..=1000000))]
    pub frames: u32,
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(0..=100000))]
    pub warmup: u32,
    /// Window logical size, or exact headless pixel size (WIDTHxHEIGHT).
    #[arg(long, default_value = "1280x800", value_parser = parse_size)]
    pub size: [u32; 2],
    /// Save the final headless frame as a PNG.
    #[arg(long, requires = "headless")]
    pub output: Option<PathBuf>,
    /// Save headless adapter, timing and image diagnostics as JSON.
    #[arg(long, requires = "headless")]
    pub report: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Request a software adapter for diagnostics; fail if unavailable.
    #[arg(long)]
    pub software: bool,
    #[arg(long, value_enum, default_value_t = Quality::High)]
    pub quality: Quality,
    #[arg(long, value_enum, default_value_t = Mode::Classic)]
    pub mode: Mode,
    #[arg(long, value_enum, default_value_t = Scene::Journey)]
    pub scene: Scene,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value = "2.4", value_parser = speed)]
    pub speed: f32,
    #[arg(long, default_value = "4", value_parser = trip)]
    pub trip: f32,
    #[arg(long, default_value = "2", value_parser = rgb)]
    pub rgb_shift: f32,
    #[arg(long, default_value = "1", value_parser = lsd)]
    pub lsd: f32,
    #[arg(long, default_value = "32", value_parser = rain)]
    pub rain: f32,
    #[arg(long, default_value = "1.5", value_parser = drop_size)]
    pub drop_size: f32,
    #[arg(long, default_value = "1", value_parser = bleach)]
    pub bleach: f32,
    #[arg(long)]
    pub no_travel: bool,
    #[arg(long)]
    pub no_current: bool,
    #[arg(long)]
    pub no_effects: bool,
    #[arg(long)]
    pub no_interpolation: bool,
    #[arg(long)]
    pub cloud_study: bool,
    /// Isolate one large drop with quiet observation lighting.
    #[arg(long)]
    pub drop_demo: bool,
    #[arg(long, conflicts_with = "hide_drops")]
    pub show_drops: bool,
    #[arg(long, conflicts_with = "show_drops")]
    pub hide_drops: bool,
    #[arg(long)]
    pub fullscreen: bool,
}
fn bounded(
    raw: &str,
    min: f32,
    max: f32,
) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| format!("expected a number in {min}..={max}"))?;
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(format!("expected a finite number in {min}..={max}"))
    }
}
fn speed(s: &str) -> Result<f32, String> { bounded(s, 0.1, 3.0) }
fn trip(s: &str) -> Result<f32, String> { bounded(s, 0.0, 4.0) }
fn rgb(s: &str) -> Result<f32, String> { bounded(s, 0.0, 12.0) }
fn lsd(s: &str) -> Result<f32, String> { bounded(s, 0.0, 2.0) }
fn rain(s: &str) -> Result<f32, String> { bounded(s, 0.0, 60.0) }
fn drop_size(s: &str) -> Result<f32, String> { bounded(s, 0.25, 3.0) }
fn bleach(s: &str) -> Result<f32, String> { bounded(s, 0.0, 4.0) }
fn parse_size(raw: &str) -> Result<[u32; 2], String> {
    let (w, h) = raw.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
    let (w, h) = (
        w.parse::<u32>().map_err(|_| "invalid width")?,
        h.parse::<u32>().map_err(|_| "invalid height")?,
    );
    if (32..=8192).contains(&w)
        && (32..=8192).contains(&h)
        && u64::from(w) * u64::from(h) <= 33_554_432
    {
        Ok([w, h])
    } else {
        Err("size must be 32..8192 per side and at most 32 megapixels".into())
    }
}
impl Options {
    pub fn simulation(&self) -> anyhow::Result<Simulation> {
        let mut sim = Simulation::new(self.seed)?;
        for (code, value) in [
            (2, self.speed),
            (3, self.rain),
            (4, self.drop_size),
            (5, self.bleach),
            (6, self.trip),
            (7, self.rgb_shift),
            (9, self.mode as u32 as f32),
            (10, self.scene as u32 as f32),
        ] {
            sim.command(code, value)?;
        }
        if self.no_travel || self.drop_demo {
            sim.command(11, 0.0)?;
        }
        if self.no_current || self.drop_demo {
            sim.command(12, 0.0)?;
        }
        if self.no_effects {
            sim.command(13, 0.0)?;
        }
        if self.drop_demo {
            for (code, value) in [(3, 0.0), (4, 3.0), (15, 0.0)] {
                sim.command(code, value)?;
            }
        }
        Ok(sim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preferred_defaults() {
        let o = Options::try_parse_from(["water"]).unwrap();
        assert_eq!(
            (o.speed, o.trip, o.rgb_shift, o.lsd, o.rain, o.drop_size),
            (2.4, 4., 2., 1., 32., 1.5)
        );
        assert_eq!(o.quality, Quality::High);
        assert!(!o.show_drops);
    }
    #[test]
    fn finite_ranges_and_all_boundaries() {
        for (flag, good, bad) in [
            ("--lsd", ["0", "2"], ["-0.1", "2.1"]),
            ("--speed", ["0.1", "3"], ["0", "3.1"]),
            ("--drop-size", ["0.25", "3"], ["0", "4"]),
        ] {
            for value in good {
                assert!(Options::try_parse_from(["water", flag, value]).is_ok());
            }
            for value in bad.into_iter().chain(["NaN", "inf", "-inf"]) {
                assert!(Options::try_parse_from(["water", &format!("{flag}={value}")]).is_err());
            }
        }
    }
    #[test]
    fn headless_contract() {
        let o = Options::try_parse_from([
            "water",
            "--headless",
            "--backend",
            "vulkan",
            "--software",
            "--frames",
            "12",
            "--warmup",
            "2",
            "--size",
            "320x200",
            "--output",
            "image.png",
            "--report",
            "report.json",
        ])
        .unwrap();
        assert_eq!(o.size, [320, 200]);
        assert_eq!(o.backend, Backend::Vulkan);
        assert!(o.software);
        assert!(Options::try_parse_from(["water", "--output", "x.png"]).is_err());
    }
    #[test]
    fn demo_is_isolated_without_changing_regular_defaults() {
        let regular = Options::try_parse_from(["water"]).unwrap();
        let state = regular.simulation().unwrap().state();
        assert_eq!((state.speed, state.rain, state.drop_size), (2.4, 32., 1.5));
        assert!(state.travel && state.current);
        let demo = Options::try_parse_from(["water", "--drop-demo"]).unwrap();
        let mut sim = demo.simulation().unwrap();
        let snapshot = sim.snapshot();
        assert_eq!((snapshot.rain, snapshot.drop_size), (0., 3.));
        assert!(!snapshot.travel && !snapshot.current);
        assert_eq!(snapshot.drops.len(), 1);
        assert!(!demo.show_drops);
    }
    #[test]
    fn explicit_startup_controls_reach_the_simulation() {
        let options = Options::try_parse_from([
            "water",
            "--speed",
            "1.7",
            "--rain",
            "12",
            "--mode",
            "rapids",
            "--scene",
            "prism",
            "--no-effects",
            "--no-travel",
        ])
        .unwrap();
        let state = options.simulation().unwrap().state();
        assert_eq!(
            (state.speed, state.rain, state.mode, state.scene),
            (1.7, 12., 2, 6)
        );
        assert!(!state.effects && !state.travel && state.current);
    }
    #[test]
    fn invalid_sizes_are_rejected_before_allocation() {
        for size in [
            "0x1",
            "8193x32",
            "8192x8192",
            "-1x32",
            "32xNaN",
            "32",
            "32x32x32",
        ] {
            assert!(
                Options::try_parse_from(["water", "--size", size]).is_err(),
                "{size}"
            );
        }
    }
}
