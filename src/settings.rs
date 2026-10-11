use std::time::Duration;

use once_cell::sync::Lazy;

pub struct Settings {
    /// Numerical impact retirement horizon; physical decay lives in physics.rs.
    pub attenuation_time: Duration,
}

pub static SETTINGS: Lazy<Settings> = Lazy::new(|| Settings {
    attenuation_time: Duration::from_secs(10),
});
