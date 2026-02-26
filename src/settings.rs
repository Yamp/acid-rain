use std::time::Duration;
use once_cell::sync::Lazy;

pub struct Settings {
    pub zero_color: (u8, u8, u8),
    pub wave_speed: f32,
    pub drops_frequency: f32,
    pub attenuation_time: Duration,
    pub waves_per_second: f32,
    pub waves_in_screen: f32,
}

impl Settings {
    pub fn new(
        zero_color: (u8, u8, u8),
        wave_speed: f32,
        drops_frequency: f32,
        attenuation_time_secs: u64,
        waves_per_second: f32,
        waves_in_screen: f32,
    ) -> Self {
        Self {
            zero_color,
            wave_speed,
            drops_frequency,
            attenuation_time: Duration::from_secs(attenuation_time_secs),
            waves_per_second,
            waves_in_screen,
        }
    }
}

pub static SETTINGS: Lazy<Settings> = Lazy::new(|| Settings::new(
    (0, 0, 0),
    0.15,
    1.0,
    10,
    0.8,
    3.5,
));
