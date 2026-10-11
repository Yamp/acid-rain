//! Production sampling helpers against an independent f64 interpolation oracle.
//! The portable path requests no FP32 filtering feature. Hardware parity is
//! additionally checked when the selected adapter supports FLOAT32_FILTERABLE.
mod common;

use wgpu::util::DeviceExt;

const SIZE_2D: [usize; 2] = [5, 3];
const SIZE_3D: [usize; 3] = [3, 4, 5];
const PROBE: &str = r#"
@group(0) @binding(0) var probePlane: texture_2d<f32>;
@group(0) @binding(1) var probeVolume: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> probeQueries: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> probeResults: array<vec4f>;
@compute @workgroup_size(32)
fn samplingProbe(@builtin(global_invocation_id) index: vec3u) {
    if index.x >= arrayLength(&probeQueries) { return; }
    let query = probeQueries[index.x];
    switch u32(query.w) {
        case 0u: { probeResults[index.x] = sample_clamp2(probePlane, query.xy); }
        case 1u: { probeResults[index.x] = sample_repeat2(probePlane, query.xy); }
        case 2u: { probeResults[index.x] = sample_mirror2(probePlane, query.xy); }
        default: { probeResults[index.x] = sample_clamp3(probeVolume, query.xyz); }
    }
}
"#;

fn pixels_2d() -> Vec<[f32; 4]> {
    (0..SIZE_2D[0] * SIZE_2D[1])
        .map(|index| {
            let x = (index % SIZE_2D[0]) as f32;
            let y = (index / SIZE_2D[0]) as f32;
            [
                0.125 + x * 0.5 + y * 1.25,
                -0.25 + x * x * 0.125 - y * 0.5,
                (1.0 + x) * (2.0 + y) * 0.25,
                0.5 + x * 0.25 - y * 0.125,
            ]
        })
        .collect()
}

fn pixels_3d() -> Vec<[f32; 4]> {
    (0..SIZE_3D.iter().product())
        .map(|index| {
            let x = (index % SIZE_3D[0]) as f32;
            let y = (index / SIZE_3D[0] % SIZE_3D[1]) as f32;
            let z = (index / (SIZE_3D[0] * SIZE_3D[1])) as f32;
            [
                0.25 + x * 0.5 + y * 0.75 + z * 1.25,
                -0.5 + x * x * 0.25 - y * z * 0.125,
                (1.0 + x) * (1.0 + y) * (1.0 + z) * 0.125,
                1.0 + x * 0.125 - y * 0.25 + z * 0.5,
            ]
        })
        .collect()
}

fn queries() -> Vec<[f32; 4]> {
    let mut queries = Vec::new();
    // Binary fractions make interior interpolation weights exactly representable
    // even on adapters with quantized filter coefficients. NPOT texel centers
    // separately exercise the half-texel offset and floating-point roundoff.
    let coordinates = [
        -3.125, -1.0, -0.25, -0.015625, 0.0, 0.015625, 0.046875, 0.125, 0.375, 0.5, 0.875,
        0.984375, 1.0, 1.015625, 1.25, 3.375,
    ];
    for mode in 0..3 {
        for v in coordinates {
            for u in coordinates {
                queries.push([u, v, 0.0, mode as f32]);
            }
        }
        for y in 0..SIZE_2D[1] {
            for x in 0..SIZE_2D[0] {
                queries.push([
                    (x as f32 + 0.5) / SIZE_2D[0] as f32,
                    (y as f32 + 0.5) / SIZE_2D[1] as f32,
                    0.0,
                    mode as f32,
                ]);
            }
        }
    }
    let coordinates = [-1.25, -0.015625, 0.0, 0.125, 0.375, 0.5, 0.875, 1.0, 1.25];
    for w in coordinates {
        for v in coordinates {
            for u in coordinates {
                queries.push([u, v, w, 3.0]);
            }
        }
    }
    for z in 0..SIZE_3D[2] {
        for y in 0..SIZE_3D[1] {
            for x in 0..SIZE_3D[0] {
                queries.push([
                    (x as f32 + 0.5) / SIZE_3D[0] as f32,
                    (y as f32 + 0.5) / SIZE_3D[1] as f32,
                    (z as f32 + 0.5) / SIZE_3D[2] as f32,
                    3.0,
                ]);
            }
        }
    }
    queries
}

fn addressed(
    index: i64,
    length: usize,
    mode: u32,
) -> usize {
    let n = length as i64;
    match mode {
        1 => index.rem_euclid(n) as usize,
        2 => {
            let folded = index.rem_euclid(n * 2);
            folded.min(n * 2 - 1 - folded) as usize
        },
        _ => index.clamp(0, n - 1) as usize,
    }
}

fn reference(
    query: [f32; 4],
    plane: &[[f32; 4]],
    volume: &[[f32; 4]],
) -> [f64; 4] {
    let mode = query[3] as u32;
    let dimensions = if mode == 3 {
        SIZE_3D
    } else {
        [SIZE_2D[0], SIZE_2D[1], 1]
    };
    let coordinate: [f64; 3] = std::array::from_fn(|axis| {
        if axis == 2 && mode != 3 {
            0.0
        } else {
            f64::from(query[axis]) * dimensions[axis] as f64 - 0.5
        }
    });
    let low = coordinate.map(|value| value.floor() as i64);
    let fraction: [f64; 3] =
        std::array::from_fn(|axis| coordinate[axis] - coordinate[axis].floor());
    let mut result = [0.0; 4];
    for z in 0..if mode == 3 { 2 } else { 1 } {
        for y in 0..2 {
            for x in 0..2 {
                let offsets = [x, y, z];
                let mut position = [0; 3];
                let mut weight = 1.0;
                for axis in 0..3 {
                    let address_mode = if mode == 2 {
                        2
                    } else if mode == 1 && axis == 0 {
                        1
                    } else {
                        0
                    };
                    position[axis] =
                        addressed(low[axis] + offsets[axis], dimensions[axis], address_mode);
                    weight *= if offsets[axis] == 0 {
                        1.0 - fraction[axis]
                    } else {
                        fraction[axis]
                    };
                }
                let index =
                    position[0] + dimensions[0] * (position[1] + dimensions[1] * position[2]);
                let pixel = if mode == 3 {
                    volume[index]
                } else {
                    plane[index]
                };
                for channel in 0..4 {
                    result[channel] += weight * f64::from(pixel[channel]);
                }
            }
        }
    }
    result
}

fn execute(
    gpu: &common::Gpu,
    hardware: bool,
    queries: &[[f32; 4]],
    plane: &[[f32; 4]],
    volume: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let texture_2d = gpu.texture_2d(
        SIZE_2D[0] as u32,
        SIZE_2D[1] as u32,
        wgpu::TextureFormat::Rgba32Float,
        "sampling plane",
    );
    let texture_3d = gpu.texture_3d(
        SIZE_3D[0] as u32,
        SIZE_3D[1] as u32,
        SIZE_3D[2] as u32,
        wgpu::TextureFormat::Rgba32Float,
        "sampling volume",
    );
    gpu.upload_rgba32f(&texture_2d, plane);
    gpu.upload_rgba32f(&texture_3d, volume);
    let input = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sampling coordinates"),
            contents: bytemuck::cast_slice(queries),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sampling results"),
        size: (queries.len() * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let sample = |dimension| wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float {
            filterable: hardware,
        },
        view_dimension: dimension,
        multisampled: false,
    };
    let storage = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: wgpu::BufferSize::new(16),
    };
    let entries = [
        sample(wgpu::TextureViewDimension::D2),
        sample(wgpu::TextureViewDimension::D3),
        storage(true),
        storage(false),
    ]
    .into_iter()
    .enumerate()
    .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
        binding: binding as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    })
    .collect::<Vec<_>>();
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sampling resources"),
            entries: &entries,
        });
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sampling probes"),
            bind_group_layouts: &[&layout, &gpu.common_layout],
            push_constant_ranges: &[],
        });
    let helpers = if hardware {
        include_str!("../shaders/sampling_hardware.wgsl")
    } else {
        include_str!("../shaders/sampling_manual.wgsl")
    };
    let source = [include_str!("../shaders/types.wgsl"), helpers, PROBE].concat();
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("actual production sampling helpers"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sampling probe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("samplingProbe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let plane_view = texture_2d.create_view(&Default::default());
    let volume_view = texture_3d.create_view(&Default::default());
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sampling inputs"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&plane_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&volume_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &gpu.common_samplers, &[]);
        pass.dispatch_workgroups((queries.len() as u32).div_ceil(32), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.read_buffer(&output)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|pixel| {
            std::array::from_fn(|channel| {
                f32::from_ne_bytes(pixel[channel * 4..channel * 4 + 4].try_into().unwrap())
            })
        })
        .collect()
}

fn compare_to_reference(
    values: &[[f32; 4]],
    queries: &[[f32; 4]],
    plane: &[[f32; 4]],
    volume: &[[f32; 4]],
    path: &str,
) -> f64 {
    let mut maximum = 0.0_f64;
    for (index, (actual, query)) in values.iter().zip(queries).enumerate() {
        let expected = reference(*query, plane, volume);
        for channel in 0..4 {
            let error = (f64::from(actual[channel]) - expected[channel]).abs();
            assert!(
                actual[channel].is_finite() && error < 0.00002,
                "{path}: query {index} {query:?} channel {channel}: {actual:?} != {expected:?}"
            );
            maximum = maximum.max(error);
        }
    }
    maximum
}

#[test]
fn independent_oracle_distinguishes_wrapping_and_linear_interpolation() {
    let plane = pixels_2d();
    let volume = pixels_3d();
    assert_eq!(reference([0.5, 0.5, 0.0, 0.0], &plane, &volume), [
        2.375, -0.25, 2.25, 0.875
    ]);
    let clamp = reference([-0.25, -0.25, 0.0, 0.0], &plane, &volume);
    let repeat = reference([-0.25, -0.25, 0.0, 1.0], &plane, &volume);
    let mirror = reference([-0.25, -0.25, 0.0, 2.0], &plane, &volume);
    assert_eq!(clamp, plane[0].map(f64::from));
    assert_ne!(clamp, repeat);
    assert_ne!(clamp, mirror);
    assert_ne!(mirror, repeat);
    // Eight distinct voxels contribute to this exact binary-fraction sample.
    assert_eq!(reference([0.25, 0.5, 0.75, 3.0], &plane, &volume), [
        5.5625, -1.046875, 1.66015625, 2.28125
    ]);
}

#[test]
fn production_sampling_is_correct_without_optional_filtering_and_matches_hardware_when_available() {
    let Some(gpu) = common::gpu() else {
        return;
    };
    let plane = pixels_2d();
    let volume = pixels_3d();
    let queries = queries();
    let manual = execute(&gpu, false, &queries, &plane, &volume);
    let manual_error = compare_to_reference(&manual, &queries, &plane, &volume, "manual");
    let mut report = serde_json::json!({"adapter": gpu.adapter_info.name,
        "backend": format!("{:?}", gpu.adapter_info.backend), "queries": queries.len(),
        "manual_maximum_absolute_error": manual_error, "manual_passed": true});
    if let Some(hardware_gpu) =
        common::gpu_with_optional_features(wgpu::Features::FLOAT32_FILTERABLE)
    {
        let hardware = execute(&hardware_gpu, true, &queries, &plane, &volume);
        let hardware_error = compare_to_reference(&hardware, &queries, &plane, &volume, "hardware");
        let mut maximum_difference = 0.0_f32;
        for (a, b) in manual.iter().zip(&hardware) {
            for channel in 0..4 {
                maximum_difference = maximum_difference.max((a[channel] - b[channel]).abs());
            }
        }
        assert!(
            maximum_difference < 0.00002,
            "manual/hardware addressing or interpolation differ by {maximum_difference}"
        );
        report["hardware"] = serde_json::json!({"passed": true, "maximum_absolute_error": hardware_error,
                                               "maximum_difference_from_manual": maximum_difference});
    } else {
        report["hardware"] =
            serde_json::json!({"skipped": "adapter does not support FLOAT32_FILTERABLE"});
    }
    println!("{report}");
}
