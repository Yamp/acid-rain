//! One compiler policy for the application and every real-GPU fixture.
use std::path::{
    Path,
    PathBuf,
};

use anyhow::{
    Context,
    Result,
};

#[derive(Debug, PartialEq, Eq)]
enum CompilerChoice {
    Native,
    Fxc,
    Dxc(PathBuf),
}

fn compiler_choice(
    windows: bool,
    backends: wgpu::Backends,
    requested: Option<&str>,
    dxc_path: Option<&Path>,
    executable: &Path,
) -> Result<CompilerChoice> {
    if !windows || !backends.contains(wgpu::Backends::DX12) {
        return Ok(CompilerChoice::Native);
    }
    match requested.map(str::to_ascii_lowercase).as_deref() {
        Some("fxc") => return Ok(CompilerChoice::Fxc),
        None | Some("dxc" | "dynamicdxc") => {},
        Some(value) => anyhow::bail!(
            "Unsupported WGPU_DX12_COMPILER={value}; use dxc, or fxc only for explicit diagnostics"
        ),
    }
    let path = if let Some(path) = dxc_path {
        anyhow::ensure!(
            path.is_absolute(),
            "ACID_RAIN_DXC_PATH must be an absolute path to dxcompiler.dll"
        );
        path.to_owned()
    } else {
        executable
            .parent()
            .context("Cannot find the executable directory for bundled DXC")?
            .join("dxcompiler.dll")
    };
    anyhow::ensure!(
        path.is_file(),
        "DX12 requires bundled DXC: {} is missing. Extract the complete Windows release beside \
         acid-rain-desktop.exe, or set ACID_RAIN_DXC_PATH to an absolute dxcompiler.dll path; \
         --backend vulkan does not require DXC",
        path.display()
    );
    Ok(CompilerChoice::Dxc(path.canonicalize().with_context(
        || format!("Cannot resolve DXC path {}", path.display()),
    )?))
}

/// Shared by production and GPU fixtures, including tests located in `deps/`.
pub struct ConfiguredInstance {
    pub instance: wgpu::Instance,
    dx12_compiler: &'static str,
}

impl ConfiguredInstance {
    pub fn shader_compiler(
        &self,
        backend: wgpu::Backend,
    ) -> &'static str {
        match backend {
            wgpu::Backend::Dx12 => self.dx12_compiler,
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Vulkan => "SPIR-V",
            _ => "backend-native",
        }
    }
}

pub fn create_instance(
    backends: wgpu::Backends,
    flags: wgpu::InstanceFlags,
) -> Result<ConfiguredInstance> {
    let choice = compiler_choice(
        cfg!(target_os = "windows"),
        backends,
        std::env::var("WGPU_DX12_COMPILER").ok().as_deref(),
        std::env::var_os("ACID_RAIN_DXC_PATH")
            .as_deref()
            .map(Path::new),
        &std::env::current_exe().context("Cannot locate the executable for GPU initialization")?,
    )?;
    let mut descriptor = wgpu::InstanceDescriptor {
        backends,
        flags,
        ..Default::default()
    };
    let dx12_compiler = match choice {
        CompilerChoice::Native => "not selected",
        CompilerChoice::Fxc => {
            eprintln!(
                "Using FXC by explicit WGPU_DX12_COMPILER override; optimized FXC fails the \
                 verified water workload on Windows WARP"
            );
            descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc;
            "FXC (explicit diagnostic override)"
        },
        CompilerChoice::Dxc(path) => {
            descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
                dxc_path: path
                    .to_str()
                    .context("DXC path cannot be represented as UTF-8")?
                    .to_owned(),
                max_shader_model: wgpu::DxcShaderModel::V6_7,
            };
            "DXC"
        },
    };
    Ok(ConfiguredInstance {
        instance: wgpu::Instance::new(&descriptor),
        dx12_compiler,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{
        AtomicUsize,
        Ordering,
    };

    use super::*;
    struct Package(PathBuf);
    impl Package {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "acid-rain-dxc-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn executable(&self) -> PathBuf { self.0.join("acid-rain-desktop.exe") }

        fn dll(&self) -> PathBuf { self.0.join("dxcompiler.dll") }
    }
    impl Drop for Package {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    #[test]
    fn windows_uses_absolute_executable_adjacent_dxc_by_default() {
        let package = Package::new();
        std::fs::write(package.dll(), b"fixture").unwrap();
        assert_eq!(
            compiler_choice(
                true,
                wgpu::Backends::DX12,
                None,
                None,
                &package.executable()
            )
            .unwrap(),
            CompilerChoice::Dxc(package.dll().canonicalize().unwrap())
        );
    }
    #[test]
    fn missing_dxc_is_actionable_and_does_not_fall_back_to_fxc() {
        let package = Package::new();
        let error = compiler_choice(
            true,
            wgpu::Backends::DX12,
            None,
            None,
            &package.executable(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("dxcompiler.dll")
                && error.contains("ACID_RAIN_DXC_PATH")
                && error.contains("--backend vulkan"),
            "{error}"
        );
    }
    #[test]
    fn relative_dxc_override_is_rejected() {
        let error = compiler_choice(
            true,
            wgpu::Backends::DX12,
            None,
            Some(Path::new("dxcompiler.dll")),
            Path::new("unused.exe"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("absolute"), "{error}");
    }
    #[test]
    fn explicit_vulkan_and_non_windows_do_not_require_windows_dlls() {
        for (windows, backends) in [
            (true, wgpu::Backends::VULKAN),
            (false, wgpu::Backends::all()),
        ] {
            assert_eq!(
                compiler_choice(windows, backends, None, None, Path::new("missing.exe")).unwrap(),
                CompilerChoice::Native
            );
        }
    }
    #[test]
    fn fxc_requires_an_explicit_diagnostic_override() {
        assert_eq!(
            compiler_choice(
                true,
                wgpu::Backends::DX12,
                Some("FxC"),
                None,
                Path::new("missing.exe")
            )
            .unwrap(),
            CompilerChoice::Fxc
        );
        assert!(
            compiler_choice(
                true,
                wgpu::Backends::DX12,
                Some("misspelled-dxc"),
                None,
                Path::new("missing.exe")
            )
            .is_err()
        );
    }
    #[test]
    fn absolute_dxc_override_is_used_for_test_executables_in_deps() {
        let package = Package::new();
        std::fs::write(package.dll(), b"fixture").unwrap();
        assert_eq!(
            compiler_choice(
                true,
                wgpu::Backends::DX12,
                Some("dxc"),
                Some(&package.dll()),
                &package.0.join("deps/test.exe")
            )
            .unwrap(),
            CompilerChoice::Dxc(package.dll().canonicalize().unwrap())
        );
    }
}
