use std::path::PathBuf;

use clap::{
    Parser,
    ValueEnum,
};

use crate::scene::SceneKind;

pub const DEFAULT_SUBJECTIVE_INTENSITY: f32 = 1.0;
const MAX_SUBJECTIVE_INTENSITY: f32 = 2.0;
pub const DEFAULT_RAIN: f32 = 32.0;
pub const DEFAULT_DROP_SIZE: f32 = 1.5;
const MAX_RAIN: f32 = 60.0;
const MIN_DROP_SIZE: f32 = 0.25;
const MAX_DROP_SIZE: f32 = 3.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum WaterMode {
    Drain,
    #[default]
    Classic,
    Rapids,
}

impl WaterMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Drain => "drain",
            Self::Classic => "classic",
            Self::Rapids => "rapids",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Drain => "Drain",
            Self::Classic => "Classic",
            Self::Rapids => "Rapids",
        }
    }

    pub fn status(self) -> &'static str {
        match self {
            Self::Drain => "Water: Drain | d: classic | f: rapids",
            Self::Classic => "Water: Classic | d: drain | f: rapids",
            Self::Rapids => "Water: Rapids (experimental) | f: classic | d: drain",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum QualityMode {
    Auto,
    #[default]
    High,
    Balanced,
    Low,
}

impl QualityMode {
    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::High,
            Self::High => Self::Balanced,
            Self::Balanced => Self::Low,
            Self::Low => Self::Auto,
        }
    }

    pub fn status(self) -> &'static str {
        match self {
            Self::Auto => "Quality: Auto (adaptive) | k: cycle | 0/1/2/3: select",
            Self::High => "Quality: High (4 samples) | k: cycle | 0: Auto",
            Self::Balanced => "Quality: Balanced | k: cycle | 0: Auto",
            Self::Low => "Quality: Low (half resolution) | k: cycle | 0: Auto",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Living, psychedelic water in your terminal",
    after_help = "Keys: g/Tab next scene · t automatic journey · k cycle quality · 0 Auto / 1 \
                  High / 2 Balanced / 3 Low · +/- intensity · [/ ] or Left/Right speed · s reset \
                  speed · Up/Down rain rate · ,/. drop size · R reset rain · v/b bleaching · B \
                  reset bleaching · c camera travel · w current (Classic) · d drain whirlpool · f \
                  rapids · e RGB/retinal effects · i frame blending · j/l RGB shift · J reset \
                  shift · n/m subjective LSD · N reset LSD · r rain burst · Space pause · h/? \
                  controls · q/Esc/Ctrl-C quit (Esc closes help first)"
)]
pub struct Options {
    /// Display refresh target. Auto adapts scene rendering independently.
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u16).range(10..=120))]
    pub fps: u16,
    #[arg(long, value_enum, default_value_t = QualityMode::High)]
    pub quality: QualityMode,
    #[arg(long, value_enum, default_value_t = SceneKind::Journey)]
    pub scene: SceneKind,
    /// Water model: drain whirlpool, classic waves or experimental river rapids.
    #[arg(long, value_enum, default_value_t = WaterMode::Classic)]
    pub water: WaterMode,
    /// Shortcut for --water rapids.
    #[arg(long, conflicts_with = "water")]
    pub rapids: bool,
    /// Psychedelic effect strength (0..4); the default is 4.
    #[arg(long, default_value_t = crate::scene::DEFAULT_INTENSITY, value_parser = parse_intensity)]
    pub intensity: f32,
    /// Subjective retinal dose (0..2); n/m changes it, N resets to 1.
    #[arg(long = "lsd", default_value_t = DEFAULT_SUBJECTIVE_INTENSITY, value_parser = parse_subjective_intensity)]
    pub subjective_intensity: f32,
    /// RGB channel displacement radius in display pixels (0..12); j/l adjusts it.
    #[arg(long, default_value_t = crate::retina::DEFAULT_RGB_SHIFT, value_parser = parse_rgb_shift)]
    pub rgb_shift: f32,
    /// Simulation and camera speed (0.1..3); also selects the headless fixture speed.
    #[arg(long, default_value_t = crate::motion::DEFAULT_SPEED, value_parser = parse_speed)]
    pub speed: f32,
    /// Disable camera travel in Classic water; camera rotation continues. Toggle with c.
    #[arg(long)]
    pub no_travel: bool,
    /// Disable the independent uniform current in Classic water. Toggle with w.
    #[arg(long)]
    pub no_current: bool,
    /// Rain impacts per physical water second (0..60); Up/Down changes the rate.
    #[arg(long, default_value_t = DEFAULT_RAIN, value_parser = parse_rain)]
    pub rain: f32,
    /// Relative drop diameter (0.25..3); comma/period changes the size.
    #[arg(long, default_value_t = DEFAULT_DROP_SIZE, value_parser = parse_drop_size)]
    pub drop_size: f32,
    /// Pigment bleaching (0..4); 1 halves pigment every 8 physical seconds, 0 disables it.
    #[arg(long, default_value_t = crate::dye::DEFAULT_BLEACH, value_parser = parse_bleach)]
    pub bleach: f32,
    /// Seed for repeatable interactive rain patterns.
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    /// Measure a fixed 60 Hz fixture at the selected speed without terminal UI.
    #[arg(long)]
    pub benchmark: bool,
    /// Write benchmark CSV to this path as well as stdout.
    #[arg(long, requires = "benchmark")]
    pub report: Option<PathBuf>,
    /// Save a rendered frame as a portable pixmap (PPM), without opening the UI.
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
    /// Terminal columns x rows for headless rendering.
    #[arg(long, default_value = "200x60", value_parser = parse_size)]
    pub size: (u16, u16),
    /// Measured frames after 30 warmup frames in headless mode.
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u32).range(1..=100_000))]
    pub frames: u32,
}

impl Options {
    pub fn water_mode(&self) -> WaterMode {
        if self.rapids {
            WaterMode::Rapids
        } else {
            self.water
        }
    }

    pub fn toggle_water(
        &mut self,
        target: WaterMode,
    ) {
        self.water = if self.water_mode() == target {
            WaterMode::Classic
        } else {
            target
        };
        self.rapids = false;
    }

    pub fn adjust_subjective_intensity(
        &mut self,
        delta: f32,
    ) {
        if delta.is_finite() {
            self.subjective_intensity =
                (self.subjective_intensity + delta).clamp(0.0, MAX_SUBJECTIVE_INTENSITY);
        }
    }

    pub fn reset_subjective_intensity(&mut self) {
        self.subjective_intensity = DEFAULT_SUBJECTIVE_INTENSITY;
    }

    pub fn adjust_rgb_shift(
        &mut self,
        delta: f32,
    ) {
        self.rgb_shift = (self.rgb_shift + delta).clamp(0.0, crate::retina::MAX_RGB_SHIFT);
    }

    pub fn reset_rgb_shift(&mut self) { self.rgb_shift = crate::retina::DEFAULT_RGB_SHIFT; }

    pub fn adjust_rain(
        &mut self,
        delta: f32,
    ) {
        self.rain = (self.rain + delta).clamp(0.0, MAX_RAIN);
    }

    pub fn adjust_drop_size(
        &mut self,
        delta: f32,
    ) {
        self.drop_size = (self.drop_size + delta).clamp(MIN_DROP_SIZE, MAX_DROP_SIZE);
    }

    pub fn reset_rain(&mut self) {
        self.rain = DEFAULT_RAIN;
        self.drop_size = DEFAULT_DROP_SIZE;
    }

    pub fn adjust_bleach(
        &mut self,
        delta: f32,
    ) {
        self.bleach = (self.bleach + delta).clamp(0.0, crate::dye::MAX_BLEACH);
    }

    pub fn reset_bleach(&mut self) { self.bleach = crate::dye::DEFAULT_BLEACH; }
}

fn finite_range(
    value: &str,
    min: f32,
    max: f32,
) -> Result<f32, String> {
    let n = value
        .parse::<f32>()
        .map_err(|_| "expected a number".to_owned())?;
    if n.is_finite() && n >= min && n <= max {
        Ok(n)
    } else {
        Err(format!("expected a finite number between {min} and {max}"))
    }
}

fn parse_intensity(value: &str) -> Result<f32, String> {
    finite_range(value, 0.0, crate::scene::MAX_INTENSITY)
}
fn parse_subjective_intensity(value: &str) -> Result<f32, String> {
    finite_range(value, 0.0, MAX_SUBJECTIVE_INTENSITY)
}
fn parse_rgb_shift(value: &str) -> Result<f32, String> {
    finite_range(value, 0.0, crate::retina::MAX_RGB_SHIFT)
}
fn parse_speed(value: &str) -> Result<f32, String> { finite_range(value, 0.1, 3.0) }
fn parse_rain(value: &str) -> Result<f32, String> { finite_range(value, 0.0, MAX_RAIN) }
fn parse_drop_size(value: &str) -> Result<f32, String> {
    finite_range(value, MIN_DROP_SIZE, MAX_DROP_SIZE)
}
fn parse_bleach(value: &str) -> Result<f32, String> {
    finite_range(value, 0.0, crate::dye::MAX_BLEACH)
}

fn parse_size(value: &str) -> Result<(u16, u16), String> {
    let (w, h) = value.split_once('x').ok_or("expected WIDTHxHEIGHT")?;
    let w = w.parse::<u16>().map_err(|_| "invalid width")?;
    let h = h.parse::<u16>().map_err(|_| "invalid height")?;
    if w == 0 || h == 0 || w > 1000 || h > 500 {
        return Err("headless size must be between 1x1 and 1000x500".into());
    }
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subjective_lsd_accepts_only_the_finite_range_and_defaults_to_one() {
        assert_eq!(Options::parse_from(["acid-rain"]).subjective_intensity, 1.0);
        for value in ["0", "0.25", "1", "2"] {
            let options = Options::try_parse_from(["acid-rain", "--lsd", value]).unwrap();
            assert_eq!(options.subjective_intensity, value.parse::<f32>().unwrap());
        }
        for value in ["-0.1", "2.01", "NaN", "inf", "-inf"] {
            assert!(Options::try_parse_from(["acid-rain", "--lsd", value]).is_err());
        }
    }

    #[test]
    fn subjective_lsd_adjustments_clamp_and_reset_independently() {
        let mut options = Options::parse_from(["acid-rain", "--intensity", "3", "--rain", "12"]);
        for _ in 0..100 {
            options.adjust_subjective_intensity(0.1);
        }
        assert_eq!(options.subjective_intensity, 2.0);
        for _ in 0..100 {
            options.adjust_subjective_intensity(-0.1);
        }
        assert_eq!(options.subjective_intensity, 0.0);
        options.reset_subjective_intensity();
        assert_eq!(options.subjective_intensity, 1.0);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            options.adjust_subjective_intensity(invalid);
            assert_eq!(options.subjective_intensity, 1.0);
        }
        assert_eq!(options.intensity, 3.0);
        assert_eq!(options.rain, 12.0);
        assert_eq!(options.speed, 2.4);
        assert_eq!(options.quality, QualityMode::High);
    }

    #[test]
    fn streaming_world_flags_parse_independently_without_changing_the_view() {
        let defaults = Options::parse_from(["acid-rain"]);
        assert!(!defaults.no_travel);
        assert!(!defaults.no_current);
        for flags in [vec!["--no-travel"], vec!["--no-current"], vec![
            "--no-travel",
            "--no-current",
        ]] {
            let expected = (
                flags.contains(&"--no-travel"),
                flags.contains(&"--no-current"),
            );
            let mut args = vec!["acid-rain"];
            args.extend(flags);
            let options = Options::try_parse_from(args)
                .expect("camera travel and uniform current must be independently switchable");
            assert_eq!(options.water_mode(), WaterMode::Classic);
            assert_eq!(options.quality, QualityMode::High);
            assert_eq!(options.scene, SceneKind::Journey);
            assert_eq!(options.speed, 2.4);
            assert_eq!(options.rgb_shift, 2.0);
            assert_eq!((options.no_travel, options.no_current), expected);
        }
    }

    #[test]
    fn selected_startup_defaults_match_the_requested_view() {
        let options = Options::parse_from(["acid-rain"]);
        assert_eq!(options.quality, QualityMode::High);
        assert_eq!(options.water_mode(), WaterMode::Classic);
        assert_eq!(options.scene, SceneKind::Journey);
        assert_eq!(
            [
                options.speed,
                options.intensity,
                options.rgb_shift,
                options.rain,
                options.drop_size,
                options.bleach
            ],
            [2.4, 4.0, 2.0, 32.0, 1.5, 1.0]
        );
    }

    #[test]
    fn rgb_shift_accepts_finite_display_pixel_values_with_a_subtle_default() {
        assert_eq!(Options::parse_from(["acid-rain"]).rgb_shift, 2.0);
        for value in ["0", "1.5", "6.25", "12"] {
            let options = Options::try_parse_from(["acid-rain", "--rgb-shift", value]).unwrap();
            assert_eq!(options.rgb_shift, value.parse::<f32>().unwrap());
        }
        for value in ["-0.1", "12.01", "NaN", "inf", "-inf"] {
            assert!(Options::try_parse_from(["acid-rain", "--rgb-shift", value]).is_err());
        }
    }

    #[test]
    fn held_rgb_shift_controls_clamp_and_reset_without_changing_intensity() {
        let mut options = Options::parse_from(["acid-rain", "--intensity", "3", "--rain", "12"]);
        for _ in 0..100 {
            options.adjust_rgb_shift(0.25);
        }
        assert_eq!(options.rgb_shift, 12.0);
        for _ in 0..100 {
            options.adjust_rgb_shift(-0.25);
        }
        assert_eq!(options.rgb_shift, 0.0);
        options.reset_rgb_shift();
        assert_eq!(options.rgb_shift, 2.0);
        assert_eq!(options.intensity, 3.0);
        assert_eq!(options.rain, 12.0);
    }

    #[test]
    fn water_mode_defaults_to_classic_and_all_modes_can_be_selected() {
        assert_eq!(
            Options::parse_from(["acid-rain"]).water_mode(),
            WaterMode::Classic
        );
        for mode in [WaterMode::Drain, WaterMode::Classic, WaterMode::Rapids] {
            let options = Options::try_parse_from(["acid-rain", "--water", mode.name()]).unwrap();
            assert_eq!(options.water_mode(), mode);
        }
        assert!(Options::try_parse_from(["acid-rain", "--water", "invalid"]).is_err());
        assert!(Options::try_parse_from(["acid-rain", "--water", "drain", "--rapids"]).is_err());
    }

    #[test]
    fn water_toggles_enter_requested_mode_and_return_to_classic() {
        let mut options = Options::parse_from(["acid-rain"]);
        for (target, expected) in [
            (WaterMode::Drain, WaterMode::Drain),
            (WaterMode::Drain, WaterMode::Classic),
            (WaterMode::Rapids, WaterMode::Rapids),
            (WaterMode::Drain, WaterMode::Drain),
            (WaterMode::Rapids, WaterMode::Rapids),
            (WaterMode::Rapids, WaterMode::Classic),
        ] {
            options.toggle_water(target);
            assert_eq!(options.water_mode(), expected);
        }
        let mut shortcut = Options::parse_from(["acid-rain", "--rapids"]);
        shortcut.toggle_water(WaterMode::Rapids);
        assert_eq!(shortcut.water_mode(), WaterMode::Classic);
        assert!(!shortcut.rapids);
    }

    #[test]
    fn startup_speed_defaults_to_two_point_four() {
        assert_eq!(Options::parse_from(["acid-rain"]).speed, 2.4);
    }

    #[test]
    fn bleach_is_independent_of_rain_and_drop_size() {
        let options = Options::try_parse_from(["acid-rain", "--bleach", "2.5"])
            .expect("bleaching must be configurable independently");
        assert_eq!(options.bleach, 2.5);
        assert_eq!(options.rain, DEFAULT_RAIN);
        assert_eq!(options.drop_size, DEFAULT_DROP_SIZE);
        for value in ["0", "1", "4"] {
            assert!(Options::try_parse_from(["acid-rain", "--bleach", value]).is_ok());
        }
        for value in ["-0.1", "4.01", "NaN", "inf"] {
            assert!(Options::try_parse_from(["acid-rain", "--bleach", value]).is_err());
        }
    }

    #[test]
    fn held_bleach_controls_stay_bounded_and_reset_independently() {
        let mut options = Options::parse_from(["acid-rain", "--rain", "12"]);
        assert_eq!(options.bleach, 1.0);
        for _ in 0..100 {
            options.adjust_bleach(0.25);
        }
        assert_eq!(options.bleach, 4.0);
        for _ in 0..100 {
            options.adjust_bleach(-0.25);
        }
        assert_eq!(options.bleach, 0.0);
        options.reset_bleach();
        assert_eq!(options.bleach, 1.0);
        assert_eq!(options.rain, 12.0);
        assert_eq!(options.drop_size, DEFAULT_DROP_SIZE);
    }

    #[test]
    fn psychedelic_default_is_four_and_the_full_range_remains_selectable() {
        assert_eq!(Options::parse_from(["acid-rain"]).intensity, 4.0);
        for dose in ["0", "2", "3", "4"] {
            let options = Options::try_parse_from(["acid-rain", "--intensity", dose])
                .expect("the full intensity range must remain adjustable");
            assert_eq!(options.intensity, dose.parse::<f32>().unwrap());
        }
        for dose in ["-0.1", "4.1", "inf", "NaN"] {
            assert!(Options::try_parse_from(["acid-rain", "--intensity", dose]).is_err());
        }
    }

    #[test]
    fn rapids_mode_is_opt_in() {
        assert!(!Options::parse_from(["acid-rain"]).rapids);
        assert!(Options::parse_from(["acid-rain", "--rapids"]).rapids);
        assert_eq!(
            Options::parse_from(["acid-rain", "--rapids"]).water_mode(),
            WaterMode::Rapids
        );
    }

    #[test]
    fn rain_rate_and_drop_diameter_are_independent_startup_controls() {
        let options = Options::try_parse_from(["acid-rain", "--rain", "12", "--drop-size", "1.75"])
            .expect("rain rate and drop diameter must be configurable together");
        assert_eq!(options.rain, 12.0);
        assert_eq!(options.drop_size, 1.75);
    }

    #[test]
    fn held_rain_controls_reach_limits_without_changing_the_other_setting() {
        let mut options = Options::parse_from(["acid-rain"]);
        for _ in 0..100 {
            options.adjust_rain(2.0);
        }
        assert_eq!(options.rain, 60.0);
        assert_eq!(options.drop_size, 1.5);
        for _ in 0..100 {
            options.adjust_rain(-2.0);
            options.adjust_drop_size(-0.25);
        }
        assert_eq!(options.rain, 0.0);
        assert_eq!(options.drop_size, 0.25);
        for _ in 0..100 {
            options.adjust_drop_size(0.25);
        }
        assert_eq!(options.drop_size, 3.0);
        assert_eq!(options.rain, 0.0);
        options.reset_rain();
        assert_eq!(options.rain, 32.0);
        assert_eq!(options.drop_size, 1.5);
    }

    #[test]
    fn unsafe_numeric_arguments_are_rejected() {
        for (flag, value) in [
            ("--fps", "0"),
            ("--fps", "121"),
            ("--intensity", "NaN"),
            ("--rain", "inf"),
            ("--rain", "-1"),
            ("--rain", "60.1"),
            ("--drop-size", "0"),
            ("--drop-size", "3.1"),
            ("--drop-size", "NaN"),
            ("--drop-size", "inf"),
            ("--speed", "0"),
            ("--size", "0x0"),
            ("--size", "65535x65535"),
            ("--frames", "0"),
        ] {
            assert!(
                Options::try_parse_from(["acid-rain", flag, value]).is_err(),
                "{flag} {value}"
            );
        }
    }

    #[test]
    fn supported_modes_and_tiny_snapshot_parse() {
        for scene in [
            "journey", "acid", "aurora", "storm", "abyss", "lava", "prism",
        ] {
            let o = Options::try_parse_from([
                "acid-rain",
                "--scene",
                scene,
                "--size",
                "1x1",
                "--quality",
                "low",
            ])
            .unwrap();
            assert_eq!(o.size, (1, 1));
            assert_eq!(o.quality, QualityMode::Low);
        }
    }
}
