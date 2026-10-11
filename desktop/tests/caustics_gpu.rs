mod common;

use acid_rain_desktop::shaders::ShaderKind;
use bytemuck::{
    Pod,
    Zeroable,
};
use common::Gpu;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CausticsUniforms {
    light: [f32; 4],
    domain: [f32; 4],
    grid: [u32; 4],
}

fn texture_binding(
    binding: u32,
    visibility: wgpu::ShaderStages,
    array: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: if array {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            },
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_binding(visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(48),
        },
        count: None,
    }
}

struct CausticsFixture<'a> {
    gpu: &'a Gpu,
    map_layout: wgpu::BindGroupLayout,
    raster_layout: wgpu::BindGroupLayout,
    map: wgpu::ComputePipeline,
    raster: wgpu::RenderPipeline,
}

struct CausticsResult {
    pixels: Vec<[f32; 4]>,
    mapping: Vec<[f32; 4]>,
    flux: [f64; 3],
    mean: f64,
    variation: f64,
    maximum: f32,
}

impl<'a> CausticsFixture<'a> {
    fn new(gpu: &'a Gpu) -> Self {
        assert_eq!(std::mem::size_of::<CausticsUniforms>(), 48);
        let shader = gpu.shader(ShaderKind::Caustics);
        let map_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("caustics fixture map bindings"),
                entries: &[
                    uniform_binding(wgpu::ShaderStages::COMPUTE),
                    texture_binding(1, wgpu::ShaderStages::COMPUTE, false),
                    texture_binding(2, wgpu::ShaderStages::COMPUTE, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                ],
            });
        let raster_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("caustics fixture raster bindings"),
                entries: &[
                    uniform_binding(wgpu::ShaderStages::VERTEX_FRAGMENT),
                    texture_binding(4, wgpu::ShaderStages::VERTEX, true),
                ],
            });
        let map_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&map_layout, &gpu.common_layout],
                    push_constant_ranges: &[],
                });
        let raster_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&raster_layout, &gpu.common_layout],
                    push_constant_ranges: &[],
                });
        let map = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("caustics fixture mapping"),
                layout: Some(&map_pipeline_layout),
                module: &shader,
                entry_point: Some("causticsMap"),
                compilation_options: Default::default(),
                cache: None,
            });
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let raster = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("caustics fixture flux accumulation"),
                layout: Some(&raster_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("causticsVertex"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("causticsFragment"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: additive,
                            alpha: additive,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
        Self {
            gpu,
            map_layout,
            raster_layout,
            map,
            raster,
        }
    }

    fn render(
        &self,
        cells: u32,
        light: [f32; 3],
        surface: impl Fn(f32, f32) -> [f32; 4],
        depth: impl Fn(f32, f32, f32) -> f32,
        override_mapping: Option<&dyn Fn(u32, u32) -> [f32; 4]>,
    ) -> CausticsResult {
        let gpu = self.gpu;
        let field_size = 128;
        let output_size = 256;
        let mut water_pixels = Vec::new();
        let mut depth_pixels = Vec::new();
        for y in 0..field_size {
            for x in 0..field_size {
                let px = x as f32 * 2.0 / field_size as f32;
                let py = y as f32 * 2.0 / field_size as f32;
                let sample = surface(px, py);
                water_pixels.push(sample);
                depth_pixels.push([0.0, 0.0, 0.0, depth(px, py, sample[0])]);
            }
        }
        let water = gpu.texture_2d(
            field_size,
            field_size,
            wgpu::TextureFormat::Rgba32Float,
            "fixture water",
        );
        let depth_texture = gpu.texture_2d(
            field_size,
            field_size,
            wgpu::TextureFormat::Rgba32Float,
            "fixture depth",
        );
        gpu.upload(&water, bytemuck::cast_slice(&water_pixels), 16);
        gpu.upload(&depth_texture, bytemuck::cast_slice(&depth_pixels), 16);
        let mapping_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fixture spectral ray map"),
            size: wgpu::Extent3d {
                width: cells + 1,
                height: cells + 1,
                depth_or_array_layers: 3,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mapping_view = mapping_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let receiver = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fixture irradiance"),
            size: wgpu::Extent3d {
                width: output_size,
                height: output_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let parameters = CausticsUniforms {
            light: [light[0], light[1], light[2], 0.02],
            domain: [2.0, -0.5, 3.0, 1.0],
            grid: [cells, output_size, output_size, 0],
        };
        let uniforms = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let water_view = water.create_view(&Default::default());
        let depth_view = depth_texture.create_view(&Default::default());
        let map_bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.map_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&water_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mapping_view),
                },
            ],
        });
        let raster_bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.raster_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&mapping_view),
                },
            ],
        });
        let mut mapping = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = mapping.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.map);
            pass.set_bind_group(0, &map_bindings, &[]);
            pass.set_bind_group(1, &gpu.common_samplers, &[]);
            pass.dispatch_workgroups((cells + 1).div_ceil(8), (cells + 1).div_ceil(8), 3);
        }
        gpu.queue.submit([mapping.finish()]);
        if let Some(replace) = override_mapping {
            let mut values = Vec::new();
            for _ in 0..3 {
                for y in 0..=cells {
                    for x in 0..=cells {
                        values.push(replace(x, y));
                    }
                }
            }
            gpu.upload(&mapping_texture, bytemuck::cast_slice(&values), 16);
        }
        let receiver_view = receiver.create_view(&Default::default());
        let mut raster = gpu.device.create_command_encoder(&Default::default());
        {
            let attachment = Some(wgpu::RenderPassColorAttachment {
                view: &receiver_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            });
            let mut pass = raster.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[attachment],
                ..Default::default()
            });
            pass.set_pipeline(&self.raster);
            pass.set_bind_group(0, &raster_bindings, &[]);
            pass.set_bind_group(1, &gpu.common_samplers, &[]);
            pass.draw(0..cells * cells * 6, 0..3);
        }
        gpu.queue.submit([raster.finish()]);
        let bytes = gpu.read_texture(&receiver, 8);
        let pixels: Vec<[f32; 4]> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|c| {
                    half::f16::from_bits(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]])).to_f32()
                })
            })
            .collect();
        let bytes = gpu.read_texture(&mapping_texture, 16);
        let mapping: Vec<[f32; 4]> = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|c| f32::from_le_bytes(p[c * 4..c * 4 + 4].try_into().unwrap()))
            })
            .collect();
        let mut flux = [0.0_f64; 3];
        let mut interior = Vec::new();
        for (i, pixel) in pixels.iter().enumerate() {
            assert!(
                pixel.iter().all(|v| v.is_finite() && *v >= 0.0),
                "invalid caustic output {pixel:?}"
            );
            for c in 0..3 {
                flux[c] += f64::from(pixel[c]) * (3.0 / f64::from(output_size)).powi(2);
            }
            let x = (i % output_size as usize) as f32;
            let y = (i / output_size as usize) as f32;
            let px = (x + 0.5) * 3.0 / output_size as f32 - 0.5;
            let py = (y + 0.5) * 3.0 / output_size as f32 - 0.5;
            if px.min(py) > 0.3 && px.max(py) < 1.7 {
                interior.push(f64::from(pixel[1]));
            }
        }
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        let variance =
            interior.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / interior.len() as f64;
        let maximum = pixels.iter().map(|p| p[1]).fold(0.0, f32::max);
        CausticsResult {
            pixels,
            mapping,
            flux,
            mean,
            variation: variance.sqrt() / mean,
            maximum,
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = v.iter().map(|c| c * c).sum::<f32>().sqrt();
    v.map(|c| c / length)
}

fn fresnel(
    cosine: f64,
    ior: f64,
) -> f64 {
    let t = (1.0 - (1.0 - cosine * cosine) / (ior * ior)).sqrt();
    let rs = (cosine - ior * t) / (cosine + ior * t);
    let rp = (ior * cosine - t) / (ior * cosine + t);
    0.5 * (rs * rs + rp * rp)
}

#[test]
fn spectral_caustics_conserve_flux_and_hit_actual_bathymetry() {
    let Some(gpu) = common::gpu() else {
        return;
    };
    let fixture = CausticsFixture::new(&gpu);
    let flat = |_: f32, _: f32| [0.0; 4];
    let depth = |_: f32, _: f32, _: f32| 0.7;
    let iors = [1.331_f64, 1.333, 1.337];
    for cells in [64, 128] {
        let result = fixture.render(cells, [0.0, 0.0, 1.0], flat, depth, None);
        for (c, ior) in iors.into_iter().enumerate() {
            let expected = 4.0 * (1.0 - fresnel(1.0, ior));
            assert!(
                (result.flux[c] / expected - 1.0).abs() < 0.015,
                "flat flux {cells}: {:?}",
                result.flux
            );
        }
        assert!((result.mean - (1.0 - fresnel(1.0, 1.333))).abs() < 0.02);
        assert!(
            result.variation < 0.03,
            "flat plane has grid artifacts: {}",
            result.variation
        );
    }
    let light = normalize([0.3, -0.2, 1.0]);
    let normal = normalize([-0.08, 0.06, 1.0]);
    let cosine = (0..3).map(|i| normal[i] * light[i]).sum::<f32>();
    let tilted = fixture.render(
        128,
        light,
        |x, y| [0.08 * (x - 1.0) - 0.06 * (y - 1.0), 0.08, -0.06, 0.0],
        |_, _, h| 0.7 + h,
        None,
    );
    for (c, ior) in iors.into_iter().enumerate() {
        let expected =
            4.0 * f64::from(cosine / normal[2]) * (1.0 - fresnel(f64::from(cosine), ior));
        assert!(
            (tilted.flux[c] / expected - 1.0).abs() < 0.02,
            "tilted projected flux"
        );
    }
    let wave = |phase: f32| {
        move |x: f32, y: f32| {
            let k = std::f32::consts::PI * 4.0;
            [
                0.02 * (k * x + phase).cos() * (k * y).cos(),
                -0.02 * k * (k * x + phase).sin() * (k * y).cos(),
                -0.02 * k * (k * x + phase).cos() * (k * y).sin(),
                0.0,
            ]
        }
    };
    let focused = fixture.render(128, [0.0, 0.0, 1.0], wave(0.0), |_, _, h| 0.7 + h, None);
    let moved = fixture.render(128, [0.0, 0.0, 1.0], wave(0.002), |_, _, h| 0.7 + h, None);
    assert!(
        focused.maximum > 1.3 && focused.variation > 0.15,
        "curvature must focus light"
    );
    assert!((focused.flux[1] / (4.0 * (1.0 - fresnel(1.0, 1.333))) - 1.0).abs() < 0.025);
    let difference = focused
        .pixels
        .iter()
        .zip(&moved.pixels)
        .map(|(a, b)| (a[1] - b[1]).abs())
        .sum::<f32>()
        / focused.pixels.len() as f32;
    assert!(
        difference > 0.0 && difference < 0.03,
        "small surface motion must remain continuous"
    );
    let sloped = fixture.render(128, light, flat, |x, _, _| 0.7 - 0.2 * x, None);
    let ray = [
        -light[0] / 1.333,
        -light[1] / 1.333,
        -(1.0 - (light[0] * light[0] + light[1] * light[1]) / (1.333 * 1.333)).sqrt(),
    ];
    for [x, y] in [[32, 32], [64, 48], [96, 96]] {
        let source = [x as f32 / 64.0, y as f32 / 64.0];
        let t = (0.7 - 0.2 * source[0]) / (0.2 * ray[0] - ray[2]);
        let actual = sloped.mapping[129 * 129 + y * 129 + x];
        let error = ((actual[0] - source[0] - ray[0] * t).powi(2)
            + (actual[1] - source[1] - ray[1] * t).powi(2))
        .sqrt();
        assert!(error < 0.001, "sloped bed hit error {error}");
    }
    let partial = fixture.render(
        1,
        [0.0, 0.0, 1.0],
        flat,
        depth,
        Some(&|x, y| {
            if x == 0 {
                [0.0, y as f32 * 2.0, 0.0, 0.0]
            } else {
                [1.4, 1.4, 1.0, 0.0]
            }
        }),
    );
    let mut moment = [0.0_f64; 2];
    let mut mass = 0.0;
    for (i, pixel) in partial.pixels.iter().enumerate() {
        let p = [(i % 256) as f64, (i / 256) as f64];
        for c in 0..2 {
            moment[c] += ((p[c] + 0.5) * 3.0 / 256.0 - 0.5) * f64::from(pixel[1]);
        }
        mass += f64::from(pixel[1]);
    }
    assert!(
        (moment[0] / mass - 1.4).hypot(moment[1] / mass - 1.4) < 0.02,
        "missing neighboring rays relocated valid incident flux"
    );
    eprintln!(
        "PASS caustics on {:?}: flat/tilted flux, focusing, continuity, bathymetry, partial cells",
        gpu.adapter_info
    );
}

#[test]
fn living_bed_is_bounded_continuous_filtered_and_world_anchored() {
    let Some(gpu) = common::gpu() else {
        return;
    };
    // Invoke the production material directly, without duplicating its formulas.
    let source = [
        include_str!("../shaders/types.wgsl"),
        include_str!("../shaders/bed_life.wgsl"),
        r#"
@group(0) @binding(0) var<storage, read> points: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4f>;
@group(0) @binding(2) var<uniform> origin: vec4f;
@compute @workgroup_size(128)
fn bedLifeProbe(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&points) { return; }
    let p = points[id.x];
    let m = animatedBed(p.xy + origin.xy, p.z, vec4f(0.48, 0.03, 0.02, 0.04), p.w);
    results[3u * id.x] = vec4f(m.albedo, m.roughness);
    results[3u * id.x + 1u] = vec4f(m.emission, 0.0);
    results[3u * id.x + 2u] = vec4f(m.slope, 0.0, 0.0);
}
"#,
    ]
    .concat();
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production bed material correctness probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("bedLifeProbe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let evaluate = |points: &[[f32; 4]], origin: [f32; 2]| -> Vec<[f32; 4]> {
        let input = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(points),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: points.len() as u64 * 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let origin = [origin[0], origin[1], 0.0, 0.0];
        let origin_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&origin),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: origin_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((points.len() as u32).div_ceil(128), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.read_buffer(&output)
            .as_chunks::<16>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|c| f32::from_ne_bytes(p[c * 4..c * 4 + 4].try_into().unwrap()))
            })
            .collect()
    };
    let difference = |a: &[[f32; 4]], b: &[[f32; 4]]| -> (f32, f32) {
        let mut total = 0.0;
        let mut maximum = 0.0_f32;
        for (a, b) in a.iter().zip(b) {
            for c in 0..4 {
                let d = (a[c] - b[c]).abs();
                total += d;
                maximum = maximum.max(d);
            }
        }
        (total / (a.len() * 4) as f32, maximum)
    };
    let points: Vec<[f32; 4]> = (0..192 * 192)
        .map(|i| {
            [
                (i % 192) as f32 / 32.0 - 3.0,
                (i / 192) as f32 / 32.0 - 3.0,
                17.0,
                0.002,
            ]
        })
        .collect();
    let initial = evaluate(&points, [0.0; 2]);
    for material in initial.as_chunks::<3>().0 {
        assert!(material.iter().flatten().all(|v| v.is_finite()));
        for (albedo, emission) in material[0][..3].iter().zip(&material[1][..3]) {
            assert!((0.0199..=0.9201).contains(albedo), "albedo is unbounded");
            assert!((0.0..=0.5501).contains(emission), "emission is unbounded");
        }
        assert!(
            (0.3..=0.95).contains(&material[0][3]),
            "roughness is unbounded"
        );
        assert!(material[2][0].abs() <= 0.3201 && material[2][1].abs() <= 0.3201);
    }
    let later = evaluate(
        &points
            .iter()
            .map(|p| [p[0], p[1], p[2] + 10.0, p[3]])
            .collect::<Vec<_>>(),
        [0.0; 2],
    );
    let nearby = evaluate(
        &points
            .iter()
            .map(|p| [p[0], p[1], p[2] + 0.0001, p[3]])
            .collect::<Vec<_>>(),
        [0.0; 2],
    );
    assert!(
        difference(&initial, &later).0 > 0.001,
        "living material must evolve"
    );
    assert!(
        difference(&initial, &nearby).1 < 0.003,
        "material has temporal jumps"
    );
    let origin = [1.5, -2.25];
    let rebased = evaluate(
        &points
            .iter()
            .map(|p| [p[0] - origin[0], p[1] - origin[1], p[2], p[3]])
            .collect::<Vec<_>>(),
        origin,
    );
    assert_eq!(
        difference(&initial, &rebased).1,
        0.0,
        "world rebase moved artwork"
    );
    let coarse = evaluate(
        &points
            .iter()
            .map(|p| [p[0], p[1], p[2], 0.2])
            .collect::<Vec<_>>(),
        [0.0; 2],
    );
    for material in coarse.as_chunks::<3>().0 {
        assert_eq!(
            material,
            &coarse[..3],
            "unresolved detail must fade to its stable average"
        );
    }
    let mut seams = Vec::new();
    for cell in -4..=4 {
        for row in 0..128 {
            let x = cell as f32 * 0.72;
            let y = row as f32 / 32.0 - 2.0;
            seams.push([x - 0.00001, y, 17.0, 0.002]);
            seams.push([x + 0.00001, y, 17.0, 0.002]);
        }
    }
    let edges = evaluate(&seams, [0.0; 2]);
    for pair in edges.as_chunks::<6>().0 {
        assert!(
            difference(&pair[..3], &pair[3..]).1 < 0.005,
            "world cell boundary is visible"
        );
    }
    eprintln!(
        "PASS living bed on {:?}: bounded material, animation, time/cell continuity, rebase and \
         footprint filtering",
        gpu.adapter_info
    );
}
