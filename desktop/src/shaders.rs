//! Embedded production WGSL, shared by the renderer and GPU contract tests.
#[derive(Clone, Copy, Debug)]
pub enum ShaderKind {
    Water,
    Droplets,
    Retina,
    Exposure,
    Cloud,
    Caustics,
}

pub fn source(
    kind: ShaderKind,
    hardware_filtering: bool,
) -> String {
    let sampling = if hardware_filtering {
        include_str!("../shaders/sampling_hardware.wgsl")
    } else {
        include_str!("../shaders/sampling_manual.wgsl")
    };
    let mut result = String::from(include_str!("../shaders/types.wgsl"));
    result.push_str(sampling);
    if matches!(kind, ShaderKind::Water | ShaderKind::Caustics) {
        result.push_str(include_str!("../shaders/caustics_common.wgsl"));
    }
    if matches!(kind, ShaderKind::Water) {
        result.push_str(include_str!("../shaders/bed_life.wgsl"));
    }
    result.push_str(match kind {
        ShaderKind::Water => include_str!("../shaders/water.wgsl"),
        ShaderKind::Droplets => include_str!("../shaders/droplets.wgsl"),
        ShaderKind::Retina => include_str!("../shaders/retina.wgsl"),
        ShaderKind::Exposure => include_str!("../shaders/exposure.wgsl"),
        ShaderKind::Cloud => include_str!("../shaders/cloud.wgsl"),
        ShaderKind::Caustics => include_str!("../shaders/caustics.wgsl"),
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_production_shaders_validate_for_both_filtering_paths() {
        for kind in [
            ShaderKind::Water,
            ShaderKind::Droplets,
            ShaderKind::Retina,
            ShaderKind::Exposure,
            ShaderKind::Cloud,
            ShaderKind::Caustics,
        ] {
            for hardware in [false, true] {
                let text = source(kind, hardware);
                let module = naga::front::wgsl::parse_str(&text).unwrap_or_else(|error| {
                    panic!(
                        "{kind:?} hardware={hardware}: {}",
                        error.emit_to_string(&text)
                    )
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|error| {
                    panic!(
                        "{kind:?} hardware={hardware}: {}",
                        error.emit_to_string(&text)
                    )
                });
                let expected: &[&str] = match kind {
                    ShaderKind::Water => &["waterVertex", "waterFragment"],
                    ShaderKind::Droplets => &["dropletVertex", "dropletFragment"],
                    ShaderKind::Retina => &["waterVertex", "postFragment", "retinaHistoryUpdate"],
                    ShaderKind::Exposure => &["adaptExposure"],
                    ShaderKind::Cloud => &[
                        "cloudNormalize",
                        "cloudPrepare",
                        "cloudVelocity",
                        "cloudAdvect",
                    ],
                    ShaderKind::Caustics => &["causticsMap", "causticsVertex", "causticsFragment"],
                };
                for entry in expected {
                    assert!(
                        module.entry_points.iter().any(|e| e.name == *entry),
                        "{kind:?} missing required entry point {entry}"
                    );
                }
            }
        }
    }
}
