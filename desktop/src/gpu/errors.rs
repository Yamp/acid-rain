#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorSource {
    Rendering,
    DeviceLost,
    Profiling,
}

#[derive(Default)]
pub struct Diagnostics {
    failure: Option<(ErrorSource, String)>,
    profiling_warning: Option<String>,
}

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(
        &self,
        metadata: &log::Metadata<'_>,
    ) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(
        &self,
        record: &log::Record<'_>,
    ) {
        if self.enabled(record.metadata()) {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr().lock(),
                "{} [{}] {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

/// Opt-in application initialization; library users keep their existing logger.
pub fn init_diagnostics_logging() {
    static LOGGER: StderrLogger = StderrLogger;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
}

pub fn check(
    device: &wgpu::Device,
    diagnostics: &std::sync::Mutex<Diagnostics>,
) -> anyhow::Result<()> {
    // Release our callback-state lock before touching the backend. Querying
    // outside a callback also avoids reentering wgpu's device-lost machinery.
    let message = diagnostics.lock().unwrap().fatal().map(str::to_owned);
    if let Some(message) = message {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: only reads an HRESULT through the live guarded device;
            // no native resources are created, destroyed, or mutated.
            if let Some(raw) = unsafe { device.as_hal::<wgpu::hal::api::Dx12>() } {
                let reason = unsafe { raw.raw_device().GetDeviceRemovedReason() };
                match reason {
                    Ok(()) => anyhow::bail!("{message}; DX12 GetDeviceRemovedReason: S_OK"),
                    Err(reason) => anyhow::bail!(
                        "{message}; DX12 GetDeviceRemovedReason: 0x{:08X} ({reason})",
                        reason.code().0 as u32
                    ),
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        let _ = device;
        anyhow::bail!("{message}");
    }
    Ok(())
}

impl Diagnostics {
    pub fn record(
        &mut self,
        source: ErrorSource,
        message: String,
    ) {
        if source == ErrorSource::Profiling {
            self.profiling_warning.get_or_insert(message);
        } else if self.failure.is_none()
            || source == ErrorSource::DeviceLost
                && self
                    .failure
                    .as_ref()
                    .is_some_and(|(old, _)| *old != ErrorSource::DeviceLost)
        {
            self.failure = Some((source, message));
        }
    }

    pub fn fatal(&self) -> Option<&str> {
        self.failure.as_ref().map(|(_, message)| message.as_str())
    }

    pub fn profiling_warning(&self) -> Option<&str> { self.profiling_warning.as_deref() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_loss_survives_later_mapping_failure() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.record(ErrorSource::DeviceLost, "DX12 device removed".into());
        diagnostics.record(ErrorSource::Profiling, "Timestamp mapping failed".into());
        assert_eq!(diagnostics.fatal(), Some("DX12 device removed"));
    }

    #[test]
    fn timing_failure_alone_does_not_become_a_rendering_failure() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.record(ErrorSource::Profiling, "Timestamp mapping failed".into());
        assert_eq!(diagnostics.fatal(), None);
    }

    #[test]
    fn first_validation_failure_survives_later_cascading_errors() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.record(ErrorSource::Rendering, "Invalid water binding".into());
        diagnostics.record(ErrorSource::Rendering, "Invalid command buffer".into());
        assert_eq!(diagnostics.fatal(), Some("Invalid water binding"));
    }

    #[test]
    fn device_loss_remains_fatal_after_a_timing_warning() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.record(ErrorSource::Profiling, "Timestamp mapping failed".into());
        diagnostics.record(ErrorSource::DeviceLost, "DX12 device removed".into());
        assert_eq!(diagnostics.fatal(), Some("DX12 device removed"));
    }
}
