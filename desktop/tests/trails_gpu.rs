//! Real fused retinal-history contracts. Opt in with ACID_RAIN_GPU_TESTS=1.
mod common;

use acid_rain_desktop::shaders::ShaderKind;
use bytemuck::{
    Pod,
    Zeroable,
};
use common::{
    Gpu,
    gpu,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    attack: [f32; 4],
    release: [f32; 4],
}

impl Parameters {
    fn step(
        dt: f64,
        valid: bool,
    ) -> Self {
        let alpha = |half_life| (1.0 - 2.0_f64.powf(-dt / half_life)) as f32;
        Self {
            attack: [
                alpha(0.025),
                alpha(0.05),
                alpha(0.09),
                if valid { 1.0 } else { 0.0 },
            ],
            release: [alpha(0.24), alpha(0.72), alpha(1.65), 0.0],
        }
    }
}

#[test]
fn fused_retina_parameters_match_the_two_vec4_shader_abi() {
    assert_eq!(std::mem::size_of::<Parameters>(), 32);
    assert_eq!(std::mem::offset_of!(Parameters, release), 16);
}

struct Fixture<'a> {
    gpu: &'a Gpu,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> Fixture<'a> {
    fn new(gpu: &'a Gpu) -> Self {
        let mut entries: Vec<_> = [2, 5, 6, 7, 8, 9, 10]
            .into_iter()
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding < 8 {
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }
                } else {
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(32),
            },
            count: None,
        });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fused retinal history fixture"),
                entries: &entries,
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("retinal fixture pipeline layout"),
                bind_group_layouts: &[&layout, &gpu.common_layout],
                push_constant_ranges: &[],
            });
        let shader = gpu.shader(ShaderKind::Retina);
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("production fused retinal history"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("retinaHistoryUpdate"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            gpu,
            layout,
            pipeline,
        }
    }

    fn dispatch(
        &self,
        source_size: (u32, u32),
        output_size: (u32, u32),
        current: &[[f32; 4]],
        old: &[Vec<[f32; 4]>; 3],
        parameters: Parameters,
    ) -> [Vec<[f32; 4]>; 3] {
        let g = self.gpu;
        let source = g.texture_2d(
            source_size.0,
            source_size.1,
            wgpu::TextureFormat::Rgba32Float,
            "retinal source",
        );
        g.upload_rgba32f(&source, current);
        let histories: [_; 3] = std::array::from_fn(|layer| {
            let texture = g.texture_2d(
                output_size.0,
                output_size.1,
                wgpu::TextureFormat::Rgba32Float,
                "retinal prior",
            );
            g.upload_rgba32f(&texture, &old[layer]);
            texture
        });
        let outputs: [_; 3] = std::array::from_fn(|_| {
            g.texture_2d(
                output_size.0,
                output_size.1,
                wgpu::TextureFormat::Rgba16Float,
                "retinal result",
            )
        });
        let views: Vec<_> = [
            &source,
            &histories[0],
            &histories[1],
            &histories[2],
            &outputs[0],
            &outputs[1],
            &outputs[2],
        ]
        .map(|texture| texture.create_view(&Default::default()))
        .into_iter()
        .collect();
        let uniform = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("retinal step parameters"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries: Vec<_> = [2, 5, 6, 7, 8, 9, 10]
            .into_iter()
            .zip(&views)
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 11,
            resource: uniform.as_entire_binding(),
        });
        let group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retinal fixture bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut encoder = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &g.common_samplers, &[]);
            pass.dispatch_workgroups(output_size.0.div_ceil(8), output_size.1.div_ceil(8), 1);
        }
        g.queue.submit([encoder.finish()]);
        outputs.map(|texture| {
            g.read_texture(&texture, 8)
                .as_chunks::<8>()
                .0
                .iter()
                .map(|pixel| {
                    std::array::from_fn(|c| {
                        half::f16::from_bits(u16::from_ne_bytes(
                            pixel[c * 2..c * 2 + 2].try_into().unwrap(),
                        ))
                        .to_f32()
                    })
                })
                .collect()
        })
    }
}

// Integrate intersecting cells in normalized image coordinates in f64. This
// deliberately scans every source cell, independently of the shader's bounded
// integer footprint and source-pixel-coordinate arithmetic.
fn area_reference(
    source: &[[f32; 4]],
    size: (u32, u32),
    output: (u32, u32),
) -> Vec<[f64; 3]> {
    let overlap = |a: u32, na: u32, b: u32, nb: u32| {
        (((a + 1) as f64 / na as f64).min((b + 1) as f64 / nb as f64)
            - (a as f64 / na as f64).max(b as f64 / nb as f64))
        .max(0.0)
    };
    (0..output.0 * output.1)
        .map(|pixel| {
            let mut sum = [0.0; 3];
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let area = overlap(x, size.0, pixel % output.0, output.0)
                        * overlap(y, size.1, pixel / output.0, output.1)
                        * f64::from(output.0 * output.1);
                    for (sum, value) in sum.iter_mut().zip(source[(y * size.0 + x) as usize]) {
                        *sum += f64::from(value.max(0.0)) * area;
                    }
                }
            }
            sum
        })
        .collect()
}

fn close(
    actual: f32,
    expected: f64,
    context: &str,
) {
    let tolerance = 1e-6 + expected.abs() * 0.001;
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
        "{context}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn fractional_and_large_restrictions_keep_impulse_flux_at_every_pixel_phase() {
    let Some(g) = gpu() else { return };
    let fixture = Fixture::new(&g);
    let impulse = [2.0, 0.5, 1.25, 1.0];
    for (source_size, output_size) in [((13, 9), (5, 4)), ((37, 29), (5, 4))] {
        let mut locations = std::collections::BTreeSet::new();
        // Sweep every source-pixel phase horizontally and vertically, plus the
        // boundaries: fixed sparse tap sets miss some of these one-pixel glints.
        for x in 0..source_size.0 {
            locations.insert((x, source_size.1 / 2));
        }
        for y in 0..source_size.1 {
            locations.insert((source_size.0 / 2, y));
        }
        for x in [0, source_size.0 - 1] {
            for y in [0, source_size.1 - 1] {
                locations.insert((x, y));
            }
        }
        let old =
            std::array::from_fn(|_| vec![[f32::NAN; 4]; (output_size.0 * output_size.1) as usize]);
        for (x, y) in locations {
            let mut source = vec![[0.0; 4]; (source_size.0 * source_size.1) as usize];
            source[(y * source_size.0 + x) as usize] = impulse;
            let expected = area_reference(&source, source_size, output_size);
            let result = fixture.dispatch(
                source_size,
                output_size,
                &source,
                &old,
                Parameters::step(0.0, false),
            );
            for (layer, pixels) in result.iter().enumerate() {
                for (actual, expected) in pixels.iter().zip(&expected) {
                    for (actual, expected) in actual.iter().zip(expected) {
                        close(*actual, *expected, "restricted impulse");
                    }
                    assert_eq!(actual[3], 1.0);
                }
                for (c, input) in impulse[..3].iter().enumerate() {
                    let flux = pixels.iter().map(|p| f64::from(p[c])).sum::<f64>()
                        * f64::from(source_size.0 * source_size.1)
                        / f64::from(output_size.0 * output_size.1);
                    assert!(
                        (flux - f64::from(*input)).abs() <= 0.002 * f64::from(*input),
                        "phase=({x},{y}) size={source_size:?} layer={layer} channel={c}: \
                         flux={flux}, input={input}"
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_history_is_seeded_without_nan_in_any_layer() {
    let Some(g) = gpu() else { return };
    let fixture = Fixture::new(&g);
    let source: Vec<_> = (0..35)
        .map(|i| [i as f32 * 0.1, -0.4, (i % 3) as f32 * 0.3, 1.0])
        .collect();
    let expected = area_reference(&source, (7, 5), (3, 2));
    let histories = [
        vec![[f32::NAN, 1.0, 2.0, 1.0]; 6],
        vec![[1.0, f32::INFINITY, 2.0, 1.0]; 6],
        vec![[1.0, 2.0, f32::NEG_INFINITY, 1.0]; 6],
    ];
    for valid in [false, true] {
        let result = fixture.dispatch(
            (7, 5),
            (3, 2),
            &source,
            &histories,
            Parameters::step(0.02, valid),
        );
        for layer in result {
            for (actual, expected) in layer.iter().zip(&expected) {
                for (actual, expected) in actual.iter().zip(expected) {
                    close(*actual, *expected, "invalid-history recovery");
                }
                assert_eq!(actual[1], 0.0);
                assert_eq!(actual[3], 1.0);
            }
        }
    }
}

#[test]
fn all_layers_follow_analytic_luminance_attack_release_and_preserve_constants() {
    let Some(g) = gpu() else { return };
    let fixture = Fixture::new(&g);
    let old = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.5, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.75, 0.75, 0.75, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.3, 0.6, 1.1, 1.0],
        [8.0, 8.0, 8.0, 1.0],
    ];
    let current = [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.5, 0.0, 1.0],
        old[4],
        old[5],
        old[6],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let histories = std::array::from_fn(|_| old.to_vec());
    let dt = 0.07;
    let result = fixture.dispatch(
        (4, 2),
        (4, 2),
        &current,
        &histories,
        Parameters::step(dt, true),
    );
    let luma = |p: [f32; 4]| {
        p.into_iter()
            .zip([0.2126, 0.7152, 0.0722])
            .map(|(c, w)| f64::from(c) * w)
            .sum::<f64>()
    };
    for (layer, pixels) in result.iter().enumerate() {
        for (index, ((actual, now), previous)) in pixels.iter().zip(current).zip(old).enumerate() {
            let half_life = if luma(now) > luma(previous) {
                [0.025, 0.05, 0.09][layer]
            } else {
                [0.24, 0.72, 1.65][layer]
            };
            let alpha = 1.0 - 2.0_f64.powf(-dt / half_life);
            for c in 0..3 {
                let expected =
                    f64::from(previous[c]) + (f64::from(now[c]) - f64::from(previous[c])) * alpha;
                close(actual[c], expected, "analytic RGB-shared attack/release");
                let tolerance = 0.001 * now[c].max(previous[c]);
                assert!(
                    actual[c] >= now[c].min(previous[c]) - tolerance
                        && actual[c] <= now[c].max(previous[c]) + tolerance
                );
            }
            if matches!(index, 0 | 1 | 4 | 5 | 7) {
                assert_eq!(actual[0], actual[1]);
                assert_eq!(actual[1], actual[2]);
            }
            assert_eq!(actual[3], 1.0);
        }
        assert_eq!(pixels[5], [0.0, 0.0, 0.0, 1.0]);
    }
    // A paused redraw must preserve each layer exactly up to output quantization.
    let held = fixture.dispatch(
        (4, 2),
        (4, 2),
        &current,
        &histories,
        Parameters::step(0.0, true),
    );
    for layer in held {
        for (actual, old) in layer.iter().zip(old) {
            for (actual, old) in actual.iter().zip(old) {
                close(*actual, f64::from(old), "zero-delta hold");
            }
        }
    }
}
