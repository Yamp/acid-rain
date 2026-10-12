//! Portable numerical contracts for the actual water reconstruction and GGX WGSL.
//! Opt in with ACID_RAIN_GPU_TESTS=1; reference integration is independent f64.
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

fn storage_entry(
    binding: u32,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Append a probe entry point, leaving every sampled production function intact.
fn probe(
    g: &Gpu,
    water: &wgpu::Texture,
    queries: &[[f32; 4]],
    count: usize,
    body: &str,
) -> Vec<[f32; 4]> {
    let source = format!(
        "{}\n@group(2) @binding(0) var<storage,read> probeQueries: array<vec4f>;\n@group(2) \
         @binding(1) var<storage,read_write> probeOutput: array<vec4f>;\n@compute \
         @workgroup_size(32) fn waterSamplingProbe(@builtin(global_invocation_id) id:vec3u) \
         {{\nlet i=id.x; if (i>=arrayLength(&probeOutput)) {{ return; }}\n{body}\n}}",
        shaders::source(ShaderKind::Water, false)
    );
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        // DXC uses this label as a source filename; avoid Windows-invalid ':' labels.
        label: Some("water_sampling_probe.wgsl"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let water_layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water sampling input layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
    let view = water.create_view(&Default::default());
    let water_group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water sampling input"),
        layout: &water_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    let probe_layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water sampling query layout"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
    let input = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water sampling queries"),
            contents: bytemuck::cast_slice(queries),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("water sampling results"),
        size: (count * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let probe_group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water sampling queries and results"),
        layout: &probe_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water sampling pipeline layout"),
            bind_group_layouts: &[&water_layout, &g.common_layout, &probe_layout],
            push_constant_ranges: &[],
        });
    let pipeline = g
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("water sampling probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("waterSamplingProbe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &water_group, &[]);
        pass.set_bind_group(1, &g.common_samplers, &[]);
        pass.set_bind_group(2, &probe_group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(32), 1, 1);
    }
    g.queue.submit([encoder.finish()]);
    g.read_buffer(&output)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|bytes| {
            std::array::from_fn(|c| f32::from_ne_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap()))
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
fn cubic_height_and_gradient_match_independent_quadratic_and_saddle() {
    let Some(g) = gpu() else { return };
    let water = g.texture_2d(
        32,
        32,
        wgpu::TextureFormat::Rgba32Float,
        "quadratic hydraulic field",
    );
    let queries: Vec<_> = (0..61)
        .map(|i| {
            [
                0.3 + 1.2 * i as f32 / 60.0,
                0.4 + 0.8 * ((i * 17) % 61) as f32 / 60.0,
                0.0,
                0.0,
            ]
        })
        .collect();
    // [constant, x, y, xx, xy, yy]; evaluate the polynomial and its derivatives
    // directly, independently of the shader's separable cubic coefficients.
    for coefficients in [[0.0, 0.2, -0.1, 0.15, -0.1, 0.1], [
        0.05, -0.08, 0.12, -0.07, 0.25, 0.04,
    ]] {
        let polynomial = |x: f64, y: f64| {
            let [c, a, b, xx, xy, yy] = coefficients;
            [
                c + a * x + b * y + xx * x * x + xy * x * y + yy * y * y,
                a + 2.0 * xx * x + xy * y,
                b + xy * x + 2.0 * yy * y,
            ]
        };
        let pixels: Vec<_> = (0..1024)
            .map(|i| {
                let h = polynomial((i % 32) as f64 / 16.0, (i / 32) as f64 / 16.0)[0];
                // Poison the old slope payload: the shading gradient must be the
                // derivative of reconstructed height, not a separately sampled field.
                [h as f32, 19.0, -23.0, 0.0]
            })
            .collect();
        g.upload_rgba32f(&water, &pixels);
        let result = probe(
            &g,
            &water,
            &queries,
            queries.len(),
            "probeOutput[i]=vec4f(waterSurface(waterTex,probeQueries[i].xy,2.0),1.0);",
        );
        for (point, value) in queries.iter().zip(&result) {
            let expected = polynomial(f64::from(point[0]), f64::from(point[1]));
            for channel in 0..3 {
                close(
                    value[channel],
                    expected[channel],
                    2e-4,
                    "cubic height and analytic gradient",
                );
            }
        }
    }
}

#[test]
fn resolved_wave_nodes_and_secondary_cubic_crest_remain_geometrically_visible() {
    let Some(g) = gpu() else { return };
    let water = g.texture_2d(
        32,
        32,
        wgpu::TextureFormat::Rgba32Float,
        "resolved three cell waves",
    );
    let pixels: Vec<_> = (0..1024)
        .map(|i| {
            let x = (i % 32) as f64;
            let y = (i / 32) as f64;
            [
                (0.025 * (x * std::f64::consts::TAU / 3.0).sin()
                    + 0.018 * (y * std::f64::consts::TAU / 7.0).cos()) as f32,
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect();
    g.upload_rgba32f(&water, &pixels);
    let queries: Vec<_> = (3..28)
        .map(|i| [i as f32 / 16.0, ((i * 3) % 25 + 3) as f32 / 16.0, 0.0, 0.0])
        .collect();
    let result = probe(
        &g,
        &water,
        &queries,
        queries.len(),
        "probeOutput[i]=vec4f(waterSurface(waterTex,probeQueries[i].xy,2.0),1.0);",
    );
    for (point, value) in queries.iter().zip(result) {
        let x = (point[0] * 16.0) as usize;
        let y = (point[1] * 16.0) as usize;
        close(
            value[0],
            f64::from(pixels[y * 32 + x][0]),
            1e-7,
            "resolved wave node amplitude",
        );
    }
    let crest = g.texture_2d(
        8,
        8,
        wgpu::TextureFormat::Rgba32Float,
        "secondary cubic crest",
    );
    let pixels: Vec<_> = (0..64)
        .map(|i| {
            [
                if i % 8 == 2 || i % 8 == 3 { 0.1 } else { 0.0 },
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect();
    g.upload_rgba32f(&crest, &pixels);
    let result = probe(
        &g,
        &crest,
        &[[0.5, 0.8, 0.11, 0.0]],
        1,
        r#"
        var u:Uniforms; u.world=vec4f(0.0,-0.3,0.3,0.0);
        let hit=secondaryHit(waterTex,probeQueries[i].xyz,vec3f(1.0,0.0,0.0),u);
        probeOutput[i]=vec4f(hit.hit,select(0.0,1.0,hit.found));
    "#,
    )[0];
    assert_eq!(
        result[3], 1.0,
        "reflection must hit the cubic crest above the largest lattice sample"
    );
    // Between the two equal .1m knots, the exact cubic is
    // h=.1+.05*t-.05*t². Its first crossing of .11 is t=(1-sqrt(.2))/2.
    let root = 0.5 + 0.25 * (1.0 - 0.2_f64.sqrt()) / 2.0;
    close(result[0], root, 2e-5, "analytic first crest intersection");
    close(result[1], 0.8, 1e-6, "secondary ray y");
    close(result[2], 0.11, 1e-6, "secondary ray z");
    // An upward reflection can still hit a rising wave. An arbitrary ray.z
    // cutoff must not classify it as escaped before testing the surface.
    let uphill: Vec<_> = pixels.iter().map(|p| [p[0] * 2.0, 0.0, 0.0, 0.0]).collect();
    g.upload_rgba32f(&crest, &uphill);
    let uphill_hit = probe(
        &g,
        &crest,
        &[[0.25, 0.8, 0.01, 0.0]],
        1,
        r#"
        var u:Uniforms; u.world=vec4f(0.0,-0.3,0.3,0.0);
        let hit=secondaryHit(waterTex,probeQueries[i].xyz,normalize(vec3f(1.0,0.0,0.6)),u);
        probeOutput[i]=vec4f(hit.hit,select(0.0,1.0,hit.found));
    "#,
    )[0];
    assert_eq!(
        uphill_hit[3], 1.0,
        "upward reflection must hit the rising wave"
    );
    assert!(uphill_hit[..3].iter().all(|x| x.is_finite()));
    assert!(uphill_hit[0] > 0.25 && uphill_hit[0] < 0.5);
    close(
        uphill_hit[2],
        0.01 + 0.6 * (f64::from(uphill_hit[0]) - 0.25),
        1e-6,
        "uphill intersection lies on the reflected ray",
    );
}

#[test]
fn primary_and_secondary_find_unique_cubic_root_when_newton_derivative_vanishes() {
    let Some(g) = gpu() else { return };
    let water = g.texture_2d(
        128,
        128,
        wgpu::TextureFormat::Rgba32Float,
        "degenerate cubic crest",
    );
    let pixels: Vec<_> = (0..128 * 128)
        .map(|i| [if i % 128 <= 65 { 1.0 / 16.0 } else { 0.0 }, 0.0, 0.0, 0.0])
        .collect();
    g.upload_rgba32f(&water, &pixels);
    let result = probe(
        &g,
        &water,
        &[[1.25, 0.5, 0.216_308_6, 0.0]; 2],
        2,
        r#"
        var u:Uniforms; u.world=vec4f(0.0,-0.3,0.3,0.0);
        let eye=probeQueries[i].xyz; let ray=normalize(vec3f(-1.0,0.0,-0.625));
        var point:vec3f;
        if (i==0u) { point=intersectWater(waterTex,eye,ray,u); }
        else {
            let hit=secondaryHit(waterTex,eye,ray,u);
            if (!hit.found) { probeOutput[i]=vec4f(0.0,0.0,0.0,1.0); return; }
            point=hit.hit;
        }
        probeOutput[i]=vec4f(point,point.z-waterSurface(waterTex,point.xy,2.0).x);
    "#,
    );
    // At cell64, four heights [1/16,1/16,1/16,0] give the exact
    // polynomial h=1/16+t²(1-t)/32, t=64(x-1). The ray is
    // z=.21630859375+.625(x-1.25); its unique crossing is below.
    let expected_x = 1.011_186_353_624_293_8;
    let expected_z = 0.067_050_064_765_18;
    for (index, hit) in result.iter().enumerate() {
        close(hit[0], expected_x, 3e-6, "degenerate crest unique root x");
        close(hit[1], 0.5, 1e-7, "degenerate crest ray y");
        close(hit[2], expected_z, 3e-6, "degenerate crest unique root z");
        close(
            hit[3],
            0.0,
            1e-6,
            "degenerate crest reconstructed surface residual",
        );
        let t = 64.0 * (f64::from(hit[0]) - 1.0);
        let independent_height = 1.0 / 16.0 + t * t * (1.0 - t) / 32.0;
        close(
            hit[2],
            independent_height,
            1e-6,
            "degenerate crest independent polynomial residual",
        );
        eprintln!(
            "degenerate crest {} root error={}m residual={}m",
            if index == 0 { "primary" } else { "secondary" },
            (f64::from(hit[0]) - expected_x).abs(),
            hit[3]
        );
    }
}

type V3 = [f64; 3];
fn dot(
    a: V3,
    b: V3,
) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn normalized(a: V3) -> V3 {
    let length = dot(a, a).sqrt();
    a.map(|x| x / length)
}
fn offset(
    center: V3,
    dx: V3,
    dy: V3,
    x: f64,
    y: f64,
) -> V3 {
    std::array::from_fn(|c| center[c] + dx[c] * x + dy[c] * y)
}

/// Independent dielectric point BRDF; no production quadrature nodes, weights,
/// thresholds, cubic functions or shader code are reused by this reference.
fn point_radiance(
    n: V3,
    view: V3,
    light: V3,
) -> V3 {
    let ndv = dot(n, view);
    let ndl = dot(n, light);
    if ndv <= 0.0 || ndl <= 0.0 {
        return [0.0; 3];
    }
    let half = normalized(std::array::from_fn(|c| view[c] + light[c]));
    let ndh = dot(n, half).max(0.0);
    let alpha_squared = 0.0025;
    let distribution =
        alpha_squared / (std::f64::consts::PI * (ndh * ndh * (alpha_squared - 1.0) + 1.0).powi(2));
    let masking = 2.0 * ndv * ndl
        / (ndl * (alpha_squared + (1.0 - alpha_squared) * ndv * ndv).sqrt()
            + ndv * (alpha_squared + (1.0 - alpha_squared) * ndl * ndl).sqrt());
    let cosine = dot(view, half);
    [1.331_f64, 1.333, 1.337].map(|ior| {
        let transmitted = (1.0 - (1.0 - cosine * cosine) / (ior * ior)).sqrt();
        let s = (cosine - ior * transmitted) / (cosine + ior * transmitted);
        let p = (ior * cosine - transmitted) / (ior * cosine + transmitted);
        0.5 * (s * s + p * p) * distribution * masking * 0.18 / (4.0 * ndv)
    })
}

fn pixel_integral(v: &[V3; 7]) -> V3 {
    const SIDE: usize = 256;
    let mut sum = [0.0; 3];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let a = (x as f64 + 0.5) / SIDE as f64 - 0.5;
            let b = (y as f64 + 0.5) / SIDE as f64 - 0.5;
            let radiance = point_radiance(
                normalized(offset(v[0], v[3], v[4], a, b)),
                normalized(offset(v[1], v[5], v[6], a, b)),
                v[2],
            );
            for c in 0..3 {
                sum[c] += radiance[c];
            }
        }
    }
    sum.map(|x| x / (SIDE * SIDE) as f64)
}

#[test]
fn adaptive_ggx_matches_dense_independent_pixel_integrals() {
    let Some(g) = gpu() else { return };
    let water = g.texture_2d(
        1,
        1,
        wgpu::TextureFormat::Rgba32Float,
        "unused water binding",
    );
    g.upload_rgba32f(&water, &[[0.0; 4]]);
    let up = [0.0, 0.0, 1.0];
    let zero = [0.0; 3];
    let oblique = normalized([0.8, 0.0, 0.6]);
    let grazing = normalized([0.98, 0.0, 0.2]);
    let mut cases = vec![
        ("point", [up, up, up, zero, zero, zero, zero]),
        ("0.2 ribbon", [
            up,
            up,
            up,
            [0.2, 0.0, 0.0],
            zero,
            zero,
            zero,
        ]),
        ("0.2 square", [
            up,
            up,
            up,
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            zero,
            zero,
        ]),
        ("0.4 square", [
            up,
            up,
            up,
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            zero,
            zero,
        ]),
        ("oblique", [
            up,
            oblique,
            [-oblique[0], 0.0, oblique[2]],
            [0.1, 0.0, 0.0],
            zero,
            [0.01, 0.0, 0.0],
            zero,
        ]),
        ("grazing", [
            up,
            grazing,
            [-grazing[0], 0.0, grazing[2]],
            [0.08, 0.0, 0.0],
            zero,
            [0.005, 0.0, 0.0],
            zero,
        ]),
        ("light below", [
            up,
            up,
            [0.0, 0.0, -1.0],
            zero,
            zero,
            zero,
            zero,
        ]),
        ("view below", [
            up,
            [0.0, 0.0, -1.0],
            up,
            zero,
            zero,
            zero,
            zero,
        ]),
        ("rotated square", [
            up,
            up,
            up,
            [0.3, 0.3, 0.0],
            [-0.3, 0.3, 0.0],
            zero,
            zero,
        ]),
    ];
    for phase in 0..9 {
        cases.push(("0.4 ribbon phase", [
            normalized([0.4 * (phase as f64 / 8.0 - 0.5), 0.0, 1.0]),
            up,
            up,
            [0.4, 0.0, 0.0],
            zero,
            zero,
            zero,
        ]));
    }
    let queries: Vec<_> = cases
        .iter()
        .flat_map(|(_, vectors)| {
            vectors
                .iter()
                .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0])
        })
        .collect();
    let results = probe(
        &g,
        &water,
        &queries,
        cases.len(),
        r#"
        let k=i*7u;
        probeOutput[i]=vec4f(directHighlightPixel(probeQueries[k].xyz,
            probeQueries[k+1u].xyz,probeQueries[k+2u].xyz,probeQueries[k+3u].xyz,
            probeQueries[k+4u].xyz,probeQueries[k+5u].xyz,probeQueries[k+6u].xyz),1.0);
    "#,
    );
    let mut max_relative_error = 0.0_f64;
    for (index, ((name, _), result)) in cases.iter().zip(results).enumerate() {
        // Lift the uploaded f32 values, so CPU/GPU evaluate the identical input.
        let v: [V3; 7] = std::array::from_fn(|row| {
            std::array::from_fn(|c| f64::from(queries[index * 7 + row][c]))
        });
        let reference = pixel_integral(&v);
        for channel in 0..3 {
            let actual = f64::from(result[channel]);
            assert!(
                actual.is_finite() && actual >= 0.0,
                "{name} channel {channel} radiance={actual}"
            );
            if reference[channel] == 0.0 {
                close(result[channel], 0.0, 1e-7, "physical hemisphere exclusion");
            } else {
                let relative = (actual / reference[channel] - 1.0).abs();
                max_relative_error = max_relative_error.max(relative);
                assert!(
                    relative < 0.025,
                    "{name} index={index} channel={channel} GPU={actual} independent={}; \
                     relative={relative}",
                    reference[channel]
                );
            }
        }
        if *name == "0.4 square" {
            let center = point_radiance(v[0], v[1], v[2]);
            assert!(
                (center[1] / reference[1] - 1.0).abs() > 0.2,
                "negative control must distinguish point sampling from the pixel integral"
            );
        }
    }
    eprintln!("GGX independent 256x256 midpoint integral max relative error={max_relative_error}");
}
