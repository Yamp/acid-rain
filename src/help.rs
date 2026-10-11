use std::fmt::Write as _;

use crossterm::event::KeyCode;

use crate::cli::{
    Options,
    WaterMode,
};

const CONTROLS: &[&str] = &[
    "SCENES",
    "g / Tab         Next world",
    "t               Automatic journey",
    "d               Toggle drain whirlpool / classic",
    "f               Toggle experimental river rapids",
    "",
    "QUALITY",
    "k               Cycle Auto / High / Balanced / Low",
    "0 / 1 / 2 / 3   Select Auto / High / Balanced / Low",
    "",
    "MOTION",
    "c               Toggle camera travel (rotation continues)",
    "w               Toggle independent current (Classic)",
    "[ / ]           Slower / faster (step 0.1)",
    "Left / Right    Slower / faster (range 0.1 - 3.0)",
    "s               Reset speed to 2.40x",
    "Space           Pause / resume",
    "i               Toggle intermediate-frame blending",
    "",
    "RAIN",
    "Up / Down       More / less rain (0 - 60 impacts/s)",
    ", / .           Smaller / larger drops (0.25 - 3.00x)",
    "R               Reset rain rate and drop size",
    "r               Rain burst",
    "v / b           Less / more bleaching (0 off; max 4x)",
    "B               Reset bleaching to 1x (8s half-life)",
    "",
    "OPTICS",
    "+ / =           More psychedelic intensity",
    "-               Less intensity (0 - 4; default 4)",
    "e               Toggle RGB echoes, channel drift and retinal diffraction",
    "n / m           Less / more subjective LSD (0 - 2; step 0.1) [hold]",
    "N               Reset subjective LSD to 1.0 (press only)",
    "j / l           Less / more RGB shift (0 - 12px; step 0.25)",
    "J               Reset RGB shift to 2.00px",
    "",
    "HELP AND EXIT",
    "h / ?           Open / close this screen",
    "Esc             Close help; otherwise exit",
    "q / Ctrl-C      Exit and restore the terminal",
];

#[derive(Default)]
pub struct HelpScreen {
    scroll: usize,
}

impl HelpScreen {
    pub fn navigate(
        &mut self,
        key: KeyCode,
        height: u16,
    ) -> bool {
        let previous = self.scroll;
        let page = if height >= 8 {
            usize::from(height - 7)
        } else {
            usize::from(height).max(1)
        };
        self.scroll = match key {
            KeyCode::Down => self.scroll.saturating_add(1),
            KeyCode::Up => self.scroll.saturating_sub(1),
            KeyCode::PageDown => self.scroll.saturating_add(page),
            KeyCode::PageUp => self.scroll.saturating_sub(page),
            KeyCode::Home => 0,
            KeyCode::End => CONTROLS.len(),
            _ => return false,
        };
        self.scroll = self.scroll.min(CONTROLS.len().saturating_sub(page));
        previous != self.scroll
    }

    pub fn encode(
        &mut self,
        width: u16,
        height: u16,
        options: &Options,
        paused: bool,
        trails: bool,
        interpolation: bool,
    ) -> String {
        if width == 0 || height == 0 {
            return String::new();
        }
        let w = usize::from(width);
        let h = usize::from(height);
        let panel_width = w.min(76);
        let x = (w - panel_width) / 2 + 1;
        let page = if h >= 8 { h - 7 } else { h.max(1) };
        self.scroll = self.scroll.min(CONTROLS.len().saturating_sub(page));
        let mut screen = String::from("\x1b[?2026h\x1b[48;2;15;17;27m\x1b[2J");
        let mut line = |row: usize, content: &str, heading: bool| {
            let color = if heading {
                "232;189;219"
            } else {
                "216;222;236"
            };
            let clipped = &content[..content.len().min(panel_width)]; // All strings are ASCII.
            let _ = write!(screen, "\x1b[{};{}H\x1b[38;2;{}m{}", row, x, color, clipped);
        };
        let start = if h >= 8 {
            line(1, "ACID RAIN / CONTROLS", true);
            let animation = format!(
                "Animation held | Interpolation: {} | RGB shift: {:.2}px",
                if interpolation { "on" } else { "off" },
                options.rgb_shift
            );
            line(2, &animation, false);
            let settings = format!(
                "Scene: {} | Speed: {:.2}x | Trip: {:.1} | LSD: {:.1} | RGB: {}",
                options.scene.name(),
                options.speed,
                options.intensity,
                options.subjective_intensity,
                if trails { "on" } else { "off" }
            );
            line(3, &settings, false);
            let quality = format!(
                "Quality: {:?} | Water: {} | Previous state: {}",
                options.quality,
                options.water_mode().label(),
                if paused { "paused" } else { "running" }
            );
            line(4, &quality, false);
            let rain = format!(
                "Rain: {:.1}/s | Drop size: {:.2}x (diameter) | Bleach: {:.2}x",
                options.rain, options.drop_size, options.bleach
            );
            line(5, &rain, false);
            let world = format!(
                "Travel: {} | Current: {} ({})",
                if options.no_travel { "off" } else { "on" },
                if options.no_current { "off" } else { "on" },
                if options.water_mode() == WaterMode::Classic {
                    "Classic"
                } else {
                    "Classic only; inactive here"
                }
            );
            line(6, &world, false);
            7
        } else {
            1
        };
        for (index, content) in CONTROLS.iter().skip(self.scroll).take(page).enumerate() {
            let heading = !content.is_empty() && !content.contains("   ");
            line(start + index, content, heading);
        }
        if h >= 8 {
            line(
                h,
                "h/?/Esc: back | Up/Down/PgUp/PgDn/Home/End: scroll | q: quit",
                true,
            );
        }
        screen.push_str("\x1b[0m\x1b[?2026l");
        screen
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn subjective_lsd_help_shows_current_dose_and_independent_controls() {
        let options = Options::parse_from(["acid-rain", "--lsd", "1.7"]);
        let text = HelpScreen::default().encode(76, 48, &options, false, true, true);
        for expected in [
            "LSD: 1.7",
            "Trip: 4.0",
            "n / m           Less / more subjective LSD (0 - 2; step 0.1)",
            "N               Reset subjective LSD to 1.0",
            "Ctrl-C",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn streaming_world_help_exposes_independent_motion_controls() {
        let options = Options::parse_from(["acid-rain"]);
        let text = HelpScreen::default().encode(76, 48, &options, false, true, true);
        for expected in [
            "Travel: on | Current: on (Classic)",
            "c               Toggle camera travel (rotation continues)",
            "w               Toggle independent current (Classic)",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn streaming_world_help_reports_disabled_and_inactive_settings() {
        let options = Options::parse_from(["acid-rain", "--no-travel", "--no-current"]);
        let text = HelpScreen::default().encode(76, 8, &options, false, true, true);
        assert!(text.contains("Travel: off | Current: off (Classic)"));
        assert!(text.contains("SCENES"));
        assert!(text.contains("h/?/Esc: back"));
        for mode in ["drain", "rapids"] {
            let options = Options::parse_from(["acid-rain", "--water", mode]);
            let text = HelpScreen::default().encode(76, 10, &options, false, true, true);
            assert!(text.contains("Travel: on | Current: on (Classic only; inactive here)"));
        }
    }

    #[test]
    fn streaming_world_help_never_writes_outside_short_terminal_heights() {
        let options = Options::parse_from(["acid-rain"]);
        for height in 1..=12 {
            let mut help = HelpScreen::default();
            for key in [KeyCode::Home, KeyCode::PageDown, KeyCode::End] {
                help.navigate(key, height);
                let text = help.encode(76, height, &options, false, true, true);
                for sequence in text.split("\x1b[").skip(1) {
                    let Some(cursor) = sequence.split('H').next() else {
                        continue;
                    };
                    let Some((row, column)) = cursor.split_once(';') else {
                        continue;
                    };
                    if let (Ok(row), Ok(column)) = (row.parse::<u16>(), column.parse::<u16>()) {
                        assert!((1..=height).contains(&row), "height {height}, row {row}");
                        assert!((1..=76).contains(&column));
                    }
                }
            }
        }
    }

    #[test]
    fn help_shows_rgb_shift_and_controls_without_clipping_the_header() {
        let options = Options::parse_from(["acid-rain"]);
        let text = HelpScreen::default().encode(76, 48, &options, false, true, true);
        for expected in [
            "RGB shift: 2.00px",
            "j / l",
            "J               Reset RGB shift to 2.00px",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
        let options = Options::parse_from(["acid-rain", "--rgb-shift", "12"]);
        let text = HelpScreen::default().encode(76, 10, &options, false, true, false);
        assert!(text.contains("Animation held | Interpolation: off | RGB shift: 12.00px"));
    }

    #[test]
    fn help_shows_interpolation_control_and_current_state() {
        let options = Options::parse_from(["acid-rain"]);
        for (interpolation, state) in [(true, "on"), (false, "off")] {
            let text = HelpScreen::default().encode(120, 48, &options, false, true, interpolation);
            assert!(text.contains(&format!("Animation held | Interpolation: {state}")));
            assert!(text.contains("i               Toggle intermediate-frame blending"));
        }
    }

    #[test]
    fn help_shows_classic_default_and_drain_control() {
        let options = Options::parse_from(["acid-rain"]);
        let text = HelpScreen::default().encode(120, 48, &options, false, true, true);
        for expected in [
            "Water: Classic",
            "Quality: High",
            "Speed: 2.40x",
            "Trip: 4.0",
            "Rain: 32.0/s",
            "Drop size: 1.50x",
            "Less intensity (0 - 4; default 4)",
            "d               Toggle drain whirlpool",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
        for args in [vec!["acid-rain", "--water", "rapids"], vec![
            "acid-rain",
            "--rapids",
        ]] {
            let options = Options::parse_from(args);
            assert!(
                HelpScreen::default()
                    .encode(80, 10, &options, true, true, true)
                    .contains("Water: Rapids")
            );
        }
        let options = Options::parse_from(["acid-rain", "--water", "drain"]);
        assert!(
            HelpScreen::default()
                .encode(80, 10, &options, true, true, true)
                .contains("Water: Drain")
        );
    }

    #[test]
    fn help_shows_bleaching_controls_and_speed_reset() {
        let options = Options::parse_from(["acid-rain"]);
        let text = HelpScreen::default().encode(120, 48, &options, false, true, true);
        for control in [
            "Bleach: 1.00x",
            "v / b",
            "Reset bleaching",
            "Reset speed to 2.40x",
        ] {
            assert!(text.contains(control), "missing {control}");
        }
        let options = Options::parse_from(["acid-rain", "--bleach", "2.5"]);
        assert!(
            HelpScreen::default()
                .encode(80, 10, &options, true, true, true)
                .contains("Bleach: 2.50x")
        );
    }

    #[test]
    fn help_reports_rain_rate_and_drop_size_with_their_controls() {
        let options = Options::parse_from(["acid-rain", "--rain", "12", "--drop-size", "1.75"]);
        let text = HelpScreen::default().encode(120, 48, &options, false, true, true);
        for control in [
            "Rain: 12.0/s",
            "Drop size: 1.75x",
            "Up / Down",
            ", / .",
            "R",
        ] {
            assert!(text.contains(control), "missing {control}");
        }
    }

    #[test]
    fn full_help_contains_every_control_and_current_settings() {
        let options = Options::parse_from(["acid-rain", "--speed", "1.7"]);
        let text = HelpScreen::default().encode(120, 48, &options, true, true, true);
        for control in [
            "ACID RAIN",
            "h / ?",
            "g / Tab",
            "0 / 1 / 2 / 3",
            "Left / Right",
            "s",
            "e",
            "Space",
            "Ctrl-C",
            "1.70",
            "paused",
        ] {
            assert!(text.contains(control), "missing {control}");
        }
    }

    #[test]
    fn tiny_help_scrolls_by_its_visible_page() {
        let options = Options::parse_from(["acid-rain"]);
        let mut help = HelpScreen::default();
        assert!(
            help.encode(80, 7, &options, false, true, true)
                .contains("SCENES")
        );
        help.navigate(KeyCode::PageDown, 7);
        let next = help.encode(80, 7, &options, false, true, true);
        assert!(!next.contains("SCENES"));
        assert!(next.contains("MOTION"));
    }

    #[test]
    fn small_help_can_scroll_to_the_last_controls() {
        let options = Options::parse_from(["acid-rain"]);
        let mut help = HelpScreen::default();
        let first = help.encode(80, 10, &options, false, true, true);
        assert!(help.navigate(KeyCode::End, 10));
        let last = help.encode(80, 10, &options, false, true, true);
        assert_ne!(first, last);
        assert!(last.contains("Ctrl-C"));
    }
}
