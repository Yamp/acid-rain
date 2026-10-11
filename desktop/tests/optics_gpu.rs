//! Real production-WGSL optical contracts. Opt in with ACID_RAIN_GPU_TESTS=1.
mod common;

use acid_rain_desktop::shaders::{
    self,
    ShaderKind,
};
use common::{
    Gpu,
    gpu,
};
use wgpu::util::DeviceExt;

fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::all(),
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn probe(
    g: &Gpu,
    kind: ShaderKind,
    body: &str,
    textures: &[(u32, &wgpu::Texture)],
) -> Vec<[f32; 4]> {
    let source = format!(
        "{}\n@group(2) @binding(0) var<storage,read_write> probeOutput: array<vec4f>;\n@compute \
         @workgroup_size(1) fn opticsProbe() {{\n{body}\n}}",
        shaders::source(kind, false)
    );
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production optical probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entries: Vec<_> = textures
        .iter()
        .map(|(binding, texture)| {
            texture_entry(
                *binding,
                if texture.dimension() == wgpu::TextureDimension::D3 {
                    wgpu::TextureViewDimension::D3
                } else {
                    wgpu::TextureViewDimension::D2
                },
            )
        })
        .collect();
    let inputs_layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe input textures"),
            entries: &entries,
        });
    let views: Vec<_> = textures
        .iter()
        .map(|(_, t)| t.create_view(&Default::default()))
        .collect();
    let entries: Vec<_> = textures
        .iter()
        .zip(&views)
        .map(|((binding, _), view)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();
    let inputs = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe input group"),
        layout: &inputs_layout,
        entries: &entries,
    });
    let output_layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe output layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let output = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe results"),
        size: 32 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe output group"),
        layout: &output_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe layout"),
            bind_group_layouts: &[&inputs_layout, &g.common_layout, &output_layout],
            push_constant_ranges: &[],
        });
    let pipeline = g
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("optics probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("opticsProbe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &inputs, &[]);
        pass.set_bind_group(1, &g.common_samplers, &[]);
        pass.set_bind_group(2, &output_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    g.queue.submit([encoder.finish()]);
    g.read_buffer(&output)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|pixel| {
            std::array::from_fn(|c| f32::from_ne_bytes(pixel[c * 4..c * 4 + 4].try_into().unwrap()))
        })
        .collect()
}

fn close(
    actual: f32,
    expected: f64,
    tolerance: f64,
    context: &str,
) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
        "{context}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn fresnel_and_constant_volume_match_analytic_transport() {
    let Some(g) = gpu() else { return };
    let volume = g.texture_3d(
        2,
        2,
        2,
        wgpu::TextureFormat::Rgba32Float,
        "constant concentration",
    );
    g.upload_rgba32f(&volume, &[[0.2, 0.3, 0.4, 1.0]; 8]);
    let result = probe(
        &g,
        ShaderKind::Water,
        r#"
        var u: Uniforms;
        u.optics.y=0.6; u.illumination=vec4f(0.7,0.5,1.1,0.0);
        probeOutput[0]=vec4f(fresnel(1.0),1.0);
        probeOutput[1]=vec4f(fresnel(0.5),1.0);
        probeOutput[2]=vec4f(fresnel(0.1),1.0);
        let a=cloudTransfer(cloudTex,vec3f(1.0,1.0,-0.1),vec3f(0.0,0.0,-1.0),0.2,u,1u);
        let b=cloudTransfer(cloudTex,vec3f(1.0,1.0,-0.1),vec3f(0.0,0.0,-1.0),0.2,u,32u);
        probeOutput[3]=vec4f(a.transmittance,1.0); probeOutput[4]=vec4f(a.radiance,1.0);
        probeOutput[5]=vec4f(a.column,1.0); probeOutput[6]=vec4f(b.transmittance,1.0);
        probeOutput[7]=vec4f(b.radiance,1.0);
        let tiny=cloudTransfer(cloudTex,vec3f(1.0,1.0,-0.1),vec3f(0.0,0.0,-1.0),0.0000002,u,32u);
        probeOutput[8]=vec4f(tiny.radiance,1.0);
        let zero=cloudTransfer(cloudTex,vec3f(1.0,1.0,-0.1),vec3f(0.0,0.0,-1.0),0.0,u,0u);
        probeOutput[9]=vec4f(zero.transmittance,1.0); probeOutput[10]=vec4f(zero.radiance,1.0);
        u.illumination=vec4f(0.0);
        let dark=cloudTransfer(cloudTex,vec3f(1.0,1.0,-0.1),vec3f(0.0,0.0,-1.0),0.2,u,32u);
        probeOutput[11]=vec4f(dark.radiance,1.0);
    "#,
        &[(7, &volume)],
    );
    for (row, cosine) in [1.0_f64, 0.5, 0.1].into_iter().enumerate() {
        for (channel, ior) in [1.331_f64, 1.333, 1.337].into_iter().enumerate() {
            let transmitted_cosine = (1.0 - (1.0 - cosine * cosine) / (ior * ior)).sqrt();
            let rs = (cosine - ior * transmitted_cosine) / (cosine + ior * transmitted_cosine);
            let rp = (ior * cosine - transmitted_cosine) / (ior * cosine + transmitted_cosine);
            close(
                result[row][channel],
                0.5 * (rs * rs + rp * rp),
                2e-6,
                "dielectric Fresnel",
            );
        }
    }
    let c = [0.2_f64, 0.3, 0.4];
    let absorption = [[2.0, 0.30, 0.25], [0.65, 1.8, 0.65], [0.30, 0.45, 1.8]];
    let scattering = [[0.03, 0.90, 1.10], [0.85, 0.04, 0.65], [1.10, 0.50, 0.04]];
    for channel in 0..3 {
        let sum =
            |coefficients: [f64; 3]| c.iter().zip(coefficients).map(|(x, k)| x * k).sum::<f64>();
        let scatter = 0.32 * sum(scattering[channel]) + 0.015;
        let sigma = [1.2, 1.6, 1.2][channel] * 0.6
            + 0.015
            + 0.85 * sum(absorption[channel])
            + 0.32 * sum(scattering[channel]);
        let illumination = [0.7, 0.5, 1.1][channel];
        let transmission = (-sigma * 0.2).exp();
        let radiance = illumination * scatter / sigma * (1.0 - transmission);
        close(result[3][channel], transmission, 2e-6, "Beer one segment");
        close(result[6][channel], transmission, 5e-6, "Beer 32 segments");
        close(
            result[4][channel],
            radiance,
            2e-6,
            "single scattering one segment",
        );
        close(
            result[7][channel],
            radiance,
            2e-6,
            "single scattering 32 segments",
        );
        close(
            result[5][channel],
            c[channel] * 0.2,
            1e-7,
            "concentration column",
        );
        close(
            result[8][channel],
            illumination * scatter / sigma * -(-sigma * 2e-7).exp_m1(),
            2e-12,
            "tiny optical path",
        );
        close(result[9][channel], 1.0, 0.0, "zero-distance identity");
        close(result[10][channel], 0.0, 0.0, "zero-distance radiance");
        close(
            result[11][channel],
            0.0,
            0.0,
            "no illumination means no emission",
        );
        assert!(radiance <= illumination * (1.0 - transmission));
    }
}

#[test]
fn retinal_drift_has_a_strict_pixel_ceiling_and_chroma_preserves_luminance() {
    let Some(g) = gpu() else { return };
    let result = probe(
        &g,
        ShaderKind::Retina,
        r#"
        var u: Uniforms;
        u.post.x=1.0; u.time.z=4.0; u.time.w=2.0;
        var peak=0.0; var ymin=10.0; var ymax=0.0;
        for (var i=0u; i<1024u; i+=1u) {
            let uv=vec2f(f32(i%32u),f32(i/32u))/31.0;
            let state=vec4f(f32(i)*0.53,2.0,1.0,0.0);
            let dose=retinaDose(u,state);
            for (var c=0u; c<3u; c+=1u) { peak=max(peak,length(retinaChannelOffset(uv,c,u,state,dose))); }
            let color=retinaChroma(vec3f(1.0,0.08,0.65),0.8,uv,state,dose);
            ymin=min(ymin,dot(color,LUMA)); ymax=max(ymax,dot(color,LUMA));
        }
        probeOutput[0]=vec4f(peak,ymin,ymax,1.0);
        let state=vec4f(173.0,1.0,1.0,0.0);
        let a=retinaChannelOffset(vec2f(0.3,0.7),0u,u,state,retinaDose(u,state));
        let other=vec4f(174.0,1.0,1.0,0.0);
        let b=retinaChannelOffset(vec2f(0.3,0.7),0u,u,other,retinaDose(u,other));
        probeOutput[1]=vec4f(a,b);
        probeOutput[2]=vec4f(retinaChroma(vec3f(4.0),0.8,vec2f(0.3,0.7),state,2.0),1.0);
        probeOutput[3]=vec4f(retinaHistoryChromaRetention(vec3f(1.0,0.0,1.0),vec3f(4.0)),
            retinaHistoryChromaRetention(vec3f(0.0),vec3f(4.0,0.1,2.0)),0.0,1.0);
        u.gain=vec4f(1.0); u.optics=vec4f(0.0,0.0,1.0,1.0);
        probeOutput[4]=vec4f(toneMap(vec3f(1.0,0.0,1.0),1.0,u),1.0);
        probeOutput[5]=vec4f(toneMap(vec3f(0.0),1.0,u),1.0);
        u.post.x=0.0; probeOutput[6]=vec4f(retinaDose(u,state));
    "#,
        &[],
    );
    assert!(
        result[0][0] > 1.5 && result[0][0] <= 2.000002,
        "strict RGB radius: {:?}",
        result[0]
    );
    let expected_y = 0.2126 + 0.7152 * 0.08 + 0.0722 * 0.65;
    close(result[0][1], expected_y, 2e-6, "minimum chroma luminance");
    close(result[0][2], expected_y, 2e-6, "maximum chroma luminance");
    assert!(
        (result[1][0] - result[1][2]).hypot(result[1][1] - result[1][3]) > 0.01,
        "drift changes coherently"
    );
    for channel in &result[2][..3] {
        close(*channel, 4.0, 1e-6, "gray stays neutral");
    }
    close(
        result[3][0],
        0.0,
        1e-6,
        "white history cannot bleach magenta",
    );
    close(result[3][1], 1.0, 1e-6, "dark current permits afterimages");
    close(
        result[4][1],
        0.0,
        1e-7,
        "tone mapping introduces no green into magenta",
    );
    assert!(result[4][0] > 0.75 && result[4][2] > 0.75);
    assert_eq!(result[5][..3], [0.0; 3]);
    assert_eq!(result[6], [0.0; 4]);
}

fn constant_texture(
    g: &Gpu,
    color: [f32; 4],
    label: &str,
) -> wgpu::Texture {
    let texture = g.texture_2d(4, 4, wgpu::TextureFormat::Rgba32Float, label);
    g.upload_rgba32f(&texture, &[color; 16]);
    texture
}

#[test]
fn real_history_sampling_rejects_neutral_veil_but_keeps_colored_afterimages() {
    let Some(g) = gpu() else { return };
    let current = constant_texture(&g, [1.0, 0.0, 1.0, 1.0], "magenta present");
    let white = constant_texture(&g, [4.0, 4.0, 4.0, 1.0], "bright neutral history");
    let body = r#"
        var u: Uniforms;
        u.viewport=vec4f(64.0,64.0,1.0/64.0,1.0/64.0);
        u.time=vec4f(0.0,0.0,4.0,2.0); u.post=vec4f(1.0,2.0,0.13,1.0);
        let state=vec4f(173.0,2.0,1.0,0.0);
        probeOutput[0]=vec4f(retinaTemporal(retinaCurrent,retinaPrevious,retinaFast,retinaMiddle,
            retinaSlow,vec2f(0.5),u,state,retinaDose(u,state)),1.0);
        probeOutput[1]=vec4f(historySample(retinaCurrent,retinaPrevious,vec2f(0.5),u),1.0);
    "#;
    let bindings = |present, old| [(2, present), (3, old), (5, old), (6, old), (7, old)];
    let result = probe(&g, ShaderKind::Retina, body, &bindings(&current, &white));
    for row in &result[..2] {
        close(row[0], 1.0, 1e-6, "magenta retained red");
        close(row[1], 0.0, 1e-6, "neutral veil rejected");
        close(row[2], 1.0, 1e-6, "magenta retained blue");
    }
    let black = constant_texture(&g, [0.0, 0.0, 0.0, 1.0], "dark present");
    let colored = constant_texture(&g, [4.0, 0.1, 2.0, 1.0], "colored afterimage");
    let result = probe(&g, ShaderKind::Retina, body, &bindings(&black, &colored));
    let color = result[0];
    assert!(color.iter().all(|x| x.is_finite()));
    assert!(
        color[0] > 0.1 && color[2] > 0.05 && color[1] < color[0] * 0.1,
        "colored afterimage: {color:?}"
    );
}

#[test]
fn actual_post_fragment_preserves_constant_black_gray_and_pure_magenta() {
    let Some(g) = gpu() else { return };
    let module = g.shader(ShaderKind::Retina);
    let mut entries: Vec<_> = (0..2)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    entries.extend((2..8).map(|binding| texture_entry(binding, wgpu::TextureViewDimension::D2)));
    let inputs_layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
    let layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&inputs_layout, &g.common_layout],
            push_constant_ranges: &[],
        });
    let pipeline = g
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("actual retinal fragment"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("waterVertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("postFragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
    let output = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("linear post output"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output.create_view(&Default::default());
    let mut uniforms = [[0.0_f32; 4]; 13];
    uniforms[0] = [16.0, 16.0, 1.0 / 16.0, 1.0 / 16.0];
    uniforms[1] = [0.0, 0.0, 4.0, 2.0];
    uniforms[3] = [0.0, 0.0, 1.0, 1.0];
    uniforms[4] = [1.0; 4];
    uniforms[7] = [1.0, 2.0, 0.13, 1.0];
    let ub = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let state = [173.0_f32, 2.0, 1.0, 0.0];
    let sb = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let pupil = constant_texture(&g, [1.0; 4], "fixed exposure");
    let pupil_view = pupil.create_view(&Default::default());
    for color in [[0.0_f32; 3], [0.18; 3], [4.0; 3], [1.0, 0.0, 1.0]] {
        let input = constant_texture(&g, [color[0], color[1], color[2], 1.0], "constant scene");
        let view = input.create_view(&Default::default());
        let mut bindings = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sb.as_entire_binding(),
            },
        ];
        bindings.extend((2..8).map(|binding| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(if binding == 4 {
                &pupil_view
            } else {
                &view
            }),
        }));
        let group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &inputs_layout,
            entries: &bindings,
        });
        let mut encoder = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &g.common_samplers, &[]);
            pass.draw(0..3, 0..1);
        }
        g.queue.submit([encoder.finish()]);
        let peak = f64::from(color.into_iter().fold(0.0_f32, f32::max));
        let tone = if peak == 0.0 {
            0.0
        } else {
            (peak * (2.51 * peak + 0.03) / (peak * (2.43 * peak + 0.59) + 0.14)).min(0.9995)
        };
        for pixel in g.read_texture(&output, 8).as_chunks::<8>().0 {
            for channel in 0..3 {
                let value = half::f16::from_bits(u16::from_ne_bytes(
                    pixel[channel * 2..channel * 2 + 2].try_into().unwrap(),
                ))
                .to_f32();
                let expected = if peak == 0.0 {
                    0.0
                } else {
                    f64::from(color[channel]) * tone / peak
                };
                close(value, expected, 8e-4, "constant production post output");
            }
        }
    }
}
