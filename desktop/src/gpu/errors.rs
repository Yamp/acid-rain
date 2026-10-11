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

pub fn dump_driver_messages(device: &wgpu::Device) {
    #[cfg(not(target_os = "windows"))]
    let _ = device;
    #[cfg(target_os = "windows")]
    {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_MESSAGE,
                ID3D12InfoQueue,
            },
            core::Interface,
        };
        // SAFETY: the HAL guard keeps the native device alive. We only read
        // its debug queue, never submit or mutate graphics resources.
        let Some(raw) = (unsafe { device.as_hal::<wgpu::hal::api::Dx12>() }) else {
            return;
        };
        let Ok(messages) = raw.raw_device().cast::<ID3D12InfoQueue>() else {
            eprintln!("DX12 validation message queue unavailable (debug layer missing)");
            return;
        };
        // SAFETY: each buffer is allocated with D3D12_MESSAGE alignment and
        // the native reported byte length before GetMessage writes into it.
        unsafe {
            let count = messages.GetNumStoredMessagesAllowedByRetrievalFilter();
            eprintln!("DX12 validation messages: {count}");
            for index in 0..count.min(256) {
                let mut length = 0;
                if messages.GetMessage(index, None, &mut length).is_err()
                    || length < std::mem::size_of::<D3D12_MESSAGE>()
                {
                    continue;
                }
                let mut storage = vec![
                    D3D12_MESSAGE::default();
                    length.div_ceil(std::mem::size_of::<D3D12_MESSAGE>())
                ];
                if messages
                    .GetMessage(index, Some(storage.as_mut_ptr()), &mut length)
                    .is_ok()
                {
                    let message = &storage[0];
                    let description = if message.pDescription.is_null() {
                        std::borrow::Cow::Borrowed("<no description>")
                    } else {
                        String::from_utf8_lossy(std::slice::from_raw_parts(
                            message.pDescription,
                            message.DescriptionByteLength,
                        ))
                    };
                    eprintln!(
                        "DX12 validation severity={} id={}: {}",
                        message.Severity.0,
                        message.ID.0,
                        description.trim_end_matches('\0')
                    );
                }
            }
            messages.ClearStoredMessages();
        }
    }
}


/// Explicit debug runs retain native diagnostics even if an upstream API panics.
pub struct DriverMessagesOnDrop {
    device: wgpu::Device,
}
impl DriverMessagesOnDrop {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}
impl Drop for DriverMessagesOnDrop {
    fn drop(&mut self) { dump_driver_messages(&self.device); }
}

pub fn check(
    device: &wgpu::Device,
    diagnostics: &std::sync::Mutex<Diagnostics>,
) -> anyhow::Result<()> {
    // Release our callback-state lock before touching the backend. Querying
    // outside a callback also avoids reentering wgpu's device-lost machinery.
    let message = diagnostics.lock().unwrap().fatal().map(str::to_owned);
    if let Some(message) = message {
        dump_driver_messages(device);
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
