use crossterm::event::{
    KeyCode,
    KeyEvent,
    KeyEventKind,
    KeyModifiers,
};

use crate::cli::QualityMode;

#[derive(Debug, PartialEq)]
pub enum Action {
    Quit,
    Help,
    ToggleTrails,
    ToggleInterpolation,
    ToggleTravel,
    ToggleCurrent,
    ToggleRapids,
    ToggleDrain,
    NextScene,
    Journey,
    NextQuality,
    Quality(QualityMode),
    Pause,
    Intensity(f32),
    SubjectiveIntensity(f32),
    ResetSubjectiveIntensity,
    RgbShift(f32),
    ResetRgbShift,
    Speed(f32),
    ResetSpeed,
    RainRate(f32),
    DropSize(f32),
    ResetRain,
    Bleach(f32),
    ResetBleach,
    Burst,
    None,
}

pub fn key_action(key: KeyEvent) -> Action {
    if key.kind == KeyEventKind::Release {
        return Action::None;
    }
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Action::Quit;
    }
    match key.code {
        KeyCode::Char('h' | '?') if key.kind == KeyEventKind::Press => Action::Help,
        KeyCode::Char('e') if key.kind == KeyEventKind::Press => Action::ToggleTrails,
        KeyCode::Char('i') if key.kind == KeyEventKind::Press => Action::ToggleInterpolation,
        KeyCode::Char('c') if key.kind == KeyEventKind::Press => Action::ToggleTravel,
        KeyCode::Char('w') if key.kind == KeyEventKind::Press => Action::ToggleCurrent,
        KeyCode::Char('f') if key.kind == KeyEventKind::Press => Action::ToggleRapids,
        KeyCode::Char('d') if key.kind == KeyEventKind::Press => Action::ToggleDrain,
        KeyCode::Esc | KeyCode::Char('q') => Action::Quit,
        KeyCode::Tab | KeyCode::Char('g') => Action::NextScene,
        KeyCode::Char('t') => Action::Journey,
        KeyCode::Char('k') if key.kind == KeyEventKind::Press => Action::NextQuality,
        KeyCode::Char('0') => Action::Quality(QualityMode::Auto),
        KeyCode::Char('1') => Action::Quality(QualityMode::High),
        KeyCode::Char('2') => Action::Quality(QualityMode::Balanced),
        KeyCode::Char('3') => Action::Quality(QualityMode::Low),
        KeyCode::Char(' ') if key.kind == KeyEventKind::Press => Action::Pause,
        KeyCode::Char('+') | KeyCode::Char('=') => Action::Intensity(0.1),
        KeyCode::Char('-') => Action::Intensity(-0.1),
        KeyCode::Char('n' | 'т') => Action::SubjectiveIntensity(-0.1),
        KeyCode::Char('m' | 'ь') => Action::SubjectiveIntensity(0.1),
        KeyCode::Char('N' | 'Т') if key.kind == KeyEventKind::Press => {
            Action::ResetSubjectiveIntensity
        },
        KeyCode::Char('j') => Action::RgbShift(-0.25),
        KeyCode::Char('l') => Action::RgbShift(0.25),
        KeyCode::Char('J') if key.kind == KeyEventKind::Press => Action::ResetRgbShift,
        KeyCode::Right | KeyCode::Char(']') => Action::Speed(0.1),
        KeyCode::Left | KeyCode::Char('[') => Action::Speed(-0.1),
        KeyCode::Char('s') if key.kind == KeyEventKind::Press => Action::ResetSpeed,
        KeyCode::Up => Action::RainRate(2.0),
        KeyCode::Down => Action::RainRate(-2.0),
        KeyCode::Char(',') => Action::DropSize(-0.25),
        KeyCode::Char('.') => Action::DropSize(0.25),
        KeyCode::Char('R') if key.kind == KeyEventKind::Press => Action::ResetRain,
        KeyCode::Char('b') => Action::Bleach(0.25),
        KeyCode::Char('v') => Action::Bleach(-0.25),
        KeyCode::Char('B') if key.kind == KeyEventKind::Press => Action::ResetBleach,
        KeyCode::Char('r') => Action::Burst,
        _ => Action::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subjective_lsd_allows_repeat_but_reset_only_on_press_in_both_layouts() {
        for (code, expected) in [
            ('n', Action::SubjectiveIntensity(-0.1)),
            ('m', Action::SubjectiveIntensity(0.1)),
            ('т', Action::SubjectiveIntensity(-0.1)),
            ('ь', Action::SubjectiveIntensity(0.1)),
        ] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE);
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
        for code in ['N', 'Т'] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::SHIFT);
            assert_eq!(key_action(key), Action::ResetSubjectiveIntensity);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), Action::None);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
    }

    #[test]
    fn streaming_world_toggles_ignore_repeat_and_release() {
        for (code, action) in [('c', Action::ToggleTravel), ('w', Action::ToggleCurrent)] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE);
            assert_eq!(key_action(key), action);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), Action::None);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
    }

    #[test]
    fn rgb_shift_allows_key_repeat_but_reset_only_acts_on_press() {
        for (code, expected) in [
            ('j', Action::RgbShift(-0.25)),
            ('l', Action::RgbShift(0.25)),
        ] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE);
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
        let mut key = KeyEvent::new(KeyCode::Char('J'), KeyModifiers::SHIFT);
        assert_eq!(key_action(key), Action::ResetRgbShift);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn interpolation_toggle_ignores_hold_and_release() {
        let mut key = KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE);
        assert_eq!(key_action(key), Action::ToggleInterpolation);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn drain_toggle_ignores_hold_and_release() {
        let mut key = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE);
        assert_eq!(key_action(key), Action::ToggleDrain);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn bleach_adjustments_allow_hold_but_reset_does_not() {
        for (code, expected) in [('b', Action::Bleach(0.25)), ('v', Action::Bleach(-0.25))] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE);
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), expected);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
        let mut key = KeyEvent::new(KeyCode::Char('B'), KeyModifiers::SHIFT);
        assert_eq!(key_action(key), Action::ResetBleach);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn rain_controls_allow_hold_but_ignore_key_release() {
        for (code, action) in [
            (KeyCode::Up, Action::RainRate(2.0)),
            (KeyCode::Down, Action::RainRate(-2.0)),
            (KeyCode::Char(','), Action::DropSize(-0.25)),
            (KeyCode::Char('.'), Action::DropSize(0.25)),
        ] {
            let mut key = KeyEvent::new(code, KeyModifiers::NONE);
            assert_eq!(key_action(key), action);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), action);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
    }

    #[test]
    fn quality_controls_are_available_during_animation() {
        for (code, action) in [
            ('k', Action::NextQuality),
            ('0', Action::Quality(QualityMode::Auto)),
            ('1', Action::Quality(QualityMode::High)),
            ('2', Action::Quality(QualityMode::Balanced)),
            ('3', Action::Quality(QualityMode::Low)),
        ] {
            assert_eq!(
                key_action(KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE)),
                action,
                "quality key {code} must have an action"
            );
        }
    }

    #[test]
    fn holding_cycle_key_does_not_race_through_quality_modes() {
        let mut key = KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn help_and_echo_toggles_ignore_hold_and_release() {
        for (code, action) in [
            ('h', Action::Help),
            ('?', Action::Help),
            ('e', Action::ToggleTrails),
            ('f', Action::ToggleRapids),
            ('R', Action::ResetRain),
        ] {
            let mut key = KeyEvent::new(KeyCode::Char(code), KeyModifiers::NONE);
            assert_eq!(key_action(key), action);
            key.kind = KeyEventKind::Repeat;
            assert_eq!(key_action(key), Action::None);
            key.kind = KeyEventKind::Release;
            assert_eq!(key_action(key), Action::None);
        }
    }

    #[test]
    fn key_release_and_repeated_space_do_not_toggle_pause() {
        let mut key = KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE);
        assert_eq!(key_action(key), Action::Pause);
        key.kind = KeyEventKind::Release;
        assert_eq!(key_action(key), Action::None);
        key.kind = KeyEventKind::Repeat;
        assert_eq!(key_action(key), Action::None);
    }

    #[test]
    fn ctrl_c_with_extra_modifiers_still_exits() {
        assert_eq!(
            key_action(KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL | KeyModifiers::SHIFT
            )),
            Action::Quit
        );
    }
}
