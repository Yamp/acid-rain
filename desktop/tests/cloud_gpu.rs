//! Execute the production FP32 cloud kernels against physical invariants.
//! Run serially with ACID_RAIN_GPU_TESTS=1 cargo test -p acid-rain-desktop
//! --test cloud_gpu -- --nocapture. Default test runs only the CPU ABI assertion.
mod common;

use acid_rain_desktop::shaders::ShaderKind;
use bytemuck::{
    Pod,
    Zeroable,
};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const DEPTH: u32 = 32;
const COUNT: usize = (WIDTH * HEIGHT * DEPTH) as usize;
const CELL_VOLUME: f64 = (2.0 / WIDTH as f64) * (2.0 / HEIGHT as f64) * (0.9 / DEPTH as f64);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    step: [f32; 4],
    rebase: [f32; 4],
    domain: [f32; 4],
    options: [f32; 4],
    dispatch: [u32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            step: [0.0; 4],
            rebase: [0.0; 4],
            domain: [2.0, 2.0, -0.8, 0.9],
            options: [0.0, 1.0, 0.00001, 0.0],
            dispatch: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Drop {
    position: [f32; 4],
    pigment: [f32; 4],
    dynamics: [f32; 4],
    identity: [u32; 4],
}

impl Default for Drop {
    fn default() -> Self {
        Self {
            position: [1.0, 1.0, 0.0, 0.016],
            pigment: [0.1, 0.8, 0.1, 0.035],
            dynamics: [0.024, 0.02, 0.0, 0.25],
            identity: [1, 0, 1, 1],
        }
    }
}

#[test]
fn cloud_dispatch_abi_is_portable_and_source_offsets_are_indices() {
    assert_eq!(std::mem::size_of::<Uniforms>(), 80);
    assert_eq!(std::mem::offset_of!(Uniforms, dispatch), 64);
    assert_eq!(std::mem::size_of::<Drop>(), 64);
    assert_eq!(std::mem::offset_of!(Drop, identity), 48);
}

struct Fixture<'a> {
    gpu: &'a common::Gpu,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
    normalization: wgpu::Buffer,
    flow: wgpu::Texture,
    water: wgpu::Texture,
}

impl<'a> Fixture<'a> {
    fn new(gpu: &'a common::Gpu) -> Self {
        let buffer = |ty, size| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        };
        let texture = |dimension| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: dimension,
            multisampled: false,
        };
        let entries = [
            buffer(wgpu::BufferBindingType::Uniform, 80),
            buffer(wgpu::BufferBindingType::Storage { read_only: true }, 64),
            buffer(wgpu::BufferBindingType::Storage { read_only: false }, 16),
            texture(wgpu::TextureViewDimension::D3),
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            texture(wgpu::TextureViewDimension::D3),
            texture(wgpu::TextureViewDimension::D2),
            texture(wgpu::TextureViewDimension::D2),
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
                label: Some("cloud contract"),
                entries: &entries,
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud fixture layout"),
                bind_group_layouts: &[&layout, &gpu.common_layout],
                push_constant_ranges: &[],
            });
        let shader = gpu.shader(ShaderKind::Cloud);
        let pipelines = [
            "cloudNormalize",
            "cloudPrepare",
            "cloudVelocity",
            "cloudAdvect",
        ]
        .into_iter()
        .map(|entry| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
        .collect();
        let normalization = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud source normalization"),
            size: 512 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let flow = gpu.texture_2d(8, 8, wgpu::TextureFormat::Rgba32Float, "cloud flow");
        let water = gpu.texture_2d(8, 8, wgpu::TextureFormat::Rgba32Float, "cloud surface");
        gpu.upload_rgba32f(&flow, &[[0.0, 0.0, 0.0, 0.7]; 64]);
        gpu.upload_rgba32f(&water, &[[0.0; 4]; 64]);
        Self {
            gpu,
            layout,
            pipelines,
            normalization,
            flow,
            water,
        }
    }

    fn volume(
        &self,
        label: &str,
    ) -> wgpu::Texture {
        self.gpu.texture_3d(
            WIDTH,
            HEIGHT,
            DEPTH,
            wgpu::TextureFormat::Rgba32Float,
            label,
        )
    }

    fn dispatch(
        &self,
        entry: usize,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        velocity: &wgpu::Texture,
        uniforms: Uniforms,
        drops: &[Drop],
    ) {
        // Each dispatched snapshot has its own bytes; no queue.write_buffer
        // overwrites shared uniforms while previous dispatches are in flight.
        let uniform = self
            .gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("immutable cloud dispatch"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let placeholder = [Drop::default()];
        let drop_buffer = self
            .gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("complete source records"),
                contents: bytemuck::cast_slice(if drops.is_empty() {
                    &placeholder
                } else {
                    drops
                }),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let velocity_view =
            (if entry == 2 { input } else { velocity }).create_view(&Default::default());
        let flow_view = self.flow.create_view(&Default::default());
        let water_view = self.water.create_view(&Default::default());
        let resources = [
            uniform.as_entire_binding(),
            drop_buffer.as_entire_binding(),
            self.normalization.as_entire_binding(),
            wgpu::BindingResource::TextureView(&input_view),
            wgpu::BindingResource::TextureView(&output_view),
            wgpu::BindingResource::TextureView(&velocity_view),
            wgpu::BindingResource::TextureView(&flow_view),
            wgpu::BindingResource::TextureView(&water_view),
        ];
        let entries = resources
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect::<Vec<_>>();
        let group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cloud dispatch resources"),
                layout: &self.layout,
                entries: &entries,
            });
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[entry]);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &self.gpu.common_samplers, &[]);
            if entry == 0 {
                pass.dispatch_workgroups((uniforms.step[3] as u32).max(1), 1, 1);
            } else {
                pass.dispatch_workgroups(WIDTH.div_ceil(4), HEIGHT.div_ceil(4), DEPTH.div_ceil(4));
            }
        }
        self.gpu.queue.submit([encoder.finish()]);
    }
}

fn mass(values: &[[f32; 4]]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        values.iter().map(|p| f64::from(p[channel])).sum::<f64>() * CELL_VOLUME
    })
}

fn centroid(values: &[[f32; 4]]) -> [f64; 3] {
    let mut total = 0.0;
    let mut moment = [0.0; 3];
    for (index, pixel) in values.iter().enumerate() {
        let amount = f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]);
        let position = [
            (index % WIDTH as usize) as f64,
            (index / WIDTH as usize % HEIGHT as usize) as f64,
            (index / (WIDTH * HEIGHT) as usize) as f64,
        ];
        for axis in 0..3 {
            let point = (position[axis] + 0.5)
                * [2.0 / WIDTH as f64, 2.0 / HEIGHT as f64, 0.9 / DEPTH as f64][axis]
                + if axis == 2 { -0.8 } else { 0.0 };
            moment[axis] += point * amount;
        }
        total += amount;
    }
    moment.map(|value| value / total.max(1e-30))
}

fn valid(values: &[[f32; 4]]) {
    assert!(
        values
            .iter()
            .flatten()
            .all(|x| x.is_finite() && *x >= -1e-8),
        "negative or nonfinite cloud concentration"
    );
}

fn assert_mass(
    actual: [f64; 3],
    expected: [f64; 3],
    tolerance: f64,
) {
    for channel in 0..3 {
        assert!(
            (actual[channel] - expected[channel]).abs() < tolerance,
            "species {channel}: actual {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn production_cloud_conserves_species_and_respects_physical_time() {
    let Some(gpu) = common::gpu() else {
        return;
    };
    let fixture = Fixture::new(&gpu);
    let a = fixture.volume("cloud a");
    let b = fixture.volume("cloud b");
    let velocity = fixture.volume("cloud velocity");
    let mut u = Uniforms::default();
    u.step[3] = 1.0;
    u.rebase = [0.0, 0.0, 1.0, 1.0];
    // Nonzero source/normalization offsets catch accidental byte-offset binding
    // and an aligned but wrong event. The preceding record has overwhelming mass.
    u.dispatch = [1, 3, 0, 0];
    let mut drop = Drop::default();
    let mut poison = drop;
    poison.pigment = [1.0, 1.0, 1.0, 100.0];
    for position in [[1.0, 1.0], [0.006, 0.004], [1.994, 1.996]] {
        drop.position[..2].copy_from_slice(&position);
        let drops = [poison, drop];
        fixture.dispatch(0, &a, &b, &velocity, u, &drops);
        fixture.dispatch(1, &a, &b, &velocity, u, &drops);
        let values = gpu.read_rgba32f(&b);
        valid(&values);
        assert_mass(
            mass(&values),
            std::array::from_fn(|c| f64::from(drop.pigment[c] * drop.pigment[3])),
            2e-7,
        );
    }
    drop = Drop::default();
    u.dispatch = [0; 4];
    fixture.dispatch(0, &a, &b, &velocity, u, &[drop]);
    fixture.dispatch(1, &a, &b, &velocity, u, &[drop]);
    let initial = gpu.read_rgba32f(&b);
    let initial_mass = mass(&initial);
    u.rebase = [0.0; 4];
    fixture.dispatch(1, &b, &a, &velocity, u, &[drop]);
    assert_eq!(
        gpu.read_rgba32f(&a),
        initial,
        "non-injecting prepare changed the tracer"
    );
    u.step = [0.008, 0.2, 0.0, 1.0];
    fixture.dispatch(2, &b, &velocity, &velocity, u, &[drop]);
    let speeds = gpu.read_rgba32f(&velocity);
    let maximum = speeds
        .iter()
        .flat_map(|p| p[..3].iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        (0.08..=0.350001).contains(&maximum),
        "inactive or unsafe wake: {maximum}"
    );
    let (mut input, mut output) = (&b, &a);
    for step in 0..160 {
        u.step[1] = step as f32 * u.step[0];
        if step % 4 == 0 {
            fixture.dispatch(2, input, &velocity, &velocity, u, &[drop]);
        }
        fixture.dispatch(3, input, output, &velocity, u, &[]);
        std::mem::swap(&mut input, &mut output);
    }
    let transported = gpu.read_rgba32f(input);
    valid(&transported);
    let transported_mass = mass(&transported);
    assert_mass(transported_mass, initial_mass, 1e-7);
    assert!(
        centroid(&transported)[2] < centroid(&initial)[2] - 0.035,
        "impact did not form a descending cloud"
    );
    u.step[0] = 0.0;
    u.step[2] = 4.0;
    fixture.dispatch(3, input, output, &velocity, u, &[]);
    assert_eq!(
        gpu.read_rgba32f(output),
        transported,
        "held transport or bleaching changed"
    );
    u.step[0] = 0.008;
    u.step[2] = 2.0;
    fixture.dispatch(3, input, output, &velocity, u, &[]);
    let fade = (-std::f64::consts::LN_2 * f64::from(u.step[0] * u.step[2]) / 8.0).exp();
    assert_mass(
        mass(&gpu.read_rgba32f(output)),
        transported_mass.map(|v| v * fade),
        2e-8,
    );

    let mut marker = vec![[0.0; 4]; COUNT];
    marker[10 + WIDTH as usize * (11 + HEIGHT as usize * 12)] = [2.0, 3.0, 4.0, 0.0];
    gpu.upload_rgba32f(&a, &marker);
    u = Uniforms::default();
    u.rebase = [2.0, -3.0, 0.0, 0.0];
    fixture.dispatch(1, &a, &b, &velocity, u, &[]);
    let shifted = gpu.read_rgba32f(&b);
    assert_eq!(shifted[8 + WIDTH as usize * (14 + HEIGHT as usize * 12)], [
        2.0, 3.0, 4.0, 0.0
    ]);
    assert_eq!(mass(&shifted), mass(&marker));
    for rebase in [[1000.0, -1000.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]] {
        u.rebase = rebase;
        fixture.dispatch(1, &a, &b, &velocity, u, &[]);
        assert!(gpu.read_rgba32f(&b).iter().all(|p| *p == [0.0; 4]));
    }

    gpu.upload_rgba32f(&a, &initial);
    let uniform_velocity = [0.12, -0.03, -0.08, 1.0];
    gpu.upload_rgba32f(&velocity, &vec![uniform_velocity; COUNT]);
    u = Uniforms::default();
    u.step[0] = 0.008;
    u.options[2] = 0.0;
    input = &a;
    output = &b;
    for _ in 0..80 {
        fixture.dispatch(3, input, output, &velocity, u, &[]);
        std::mem::swap(&mut input, &mut output);
    }
    let before = centroid(&initial);
    let after = centroid(&gpu.read_rgba32f(input));
    for axis in 0..3 {
        let expected =
            before[axis] + f64::from(uniform_velocity[axis]) * f64::from(u.step[0]) * 80.0;
        assert!(
            (after[axis] - expected).abs() < 0.002,
            "incorrect physical displacement on axis {axis}: {after:?}, {expected}"
        );
    }

    let mut checker = vec![[0.0; 4]; COUNT];
    let mut harsh_velocity = checker.clone();
    for index in 0..COUNT {
        let x = index % WIDTH as usize;
        let y = index / WIDTH as usize % HEIGHT as usize;
        let z = index / (WIDTH * HEIGHT) as usize;
        checker[index] = if (x + y + z).is_multiple_of(2) {
            [1.0, 0.7, 0.2, 0.0]
        } else {
            [0.0; 4]
        };
        harsh_velocity[index] = [
            if x.is_multiple_of(3) { -0.35 } else { 0.35 },
            if y.is_multiple_of(3) { -0.35 } else { 0.35 },
            if z.is_multiple_of(3) { -0.35 } else { 0.35 },
            1.0,
        ];
    }
    gpu.upload_rgba32f(&a, &checker);
    gpu.upload_rgba32f(&velocity, &harsh_velocity);
    u.options[2] = 0.00001;
    input = &a;
    output = &b;
    for _ in 0..64 {
        fixture.dispatch(3, input, output, &velocity, u, &[]);
        std::mem::swap(&mut input, &mut output);
    }
    let harsh = gpu.read_rgba32f(input);
    valid(&harsh);
    assert_mass(mass(&harsh), mass(&checker), 2e-6);

    let mut maximum_rebase_error = 0.0_f32;
    let period = std::f64::consts::TAU;
    let cell = [2.0 / f64::from(WIDTH), 2.0 / f64::from(HEIGHT)];
    for origin in [[0.7, -0.9], [
        period - cell[0] * 0.5,
        -period + cell[1] * 0.5,
    ]] {
        u = Uniforms::default();
        u.step[1] = 1.234;
        u.options[0] = (origin[0] % period) as f32;
        u.options[3] = (origin[1] % period) as f32;
        fixture.dispatch(2, &a, &velocity, &velocity, u, &[]);
        let before = gpu.read_rgba32f(&velocity);
        u.options[0] = ((origin[0] + cell[0]) % period) as f32;
        u.options[3] = ((origin[1] - cell[1]) % period) as f32;
        fixture.dispatch(2, &a, &velocity, &velocity, u, &[]);
        let after = gpu.read_rgba32f(&velocity);
        let mut sensitivity = 0.0_f32;
        for z in 4..27 {
            for y in 1..HEIGHT as usize {
                for x in 0..WIDTH as usize - 1 {
                    let index = x + WIDTH as usize * (y + HEIGHT as usize * z);
                    let old_index = x + 1 + WIDTH as usize * (y - 1 + HEIGHT as usize * z);
                    for axis in 0..3 {
                        maximum_rebase_error = maximum_rebase_error
                            .max((after[index][axis] - before[old_index][axis]).abs());
                        sensitivity =
                            sensitivity.max((after[index][axis] - before[index][axis]).abs());
                    }
                    assert_eq!(after[index][3], before[old_index][3]);
                }
            }
        }
        assert!(
            sensitivity > 0.002,
            "world-curl fixture lacks phase sensitivity"
        );
    }
    assert!(
        maximum_rebase_error < 0.000002,
        "world curl moved on a rebase: {maximum_rebase_error}"
    );
    println!(
        "{}",
        serde_json::json!({"passed": true, "adapter": gpu.adapter_info.name,
        "backend": format!("{:?}", gpu.adapter_info.backend), "filtering": "manual",
        "grid": [WIDTH, HEIGHT, DEPTH], "initial_mass": initial_mass,
        "transport_mass": transported_mass, "maximum_velocity": maximum,
        "maximum_rebase_velocity_error": maximum_rebase_error,
        "checks": ["clipped Gaussian mass and full-buffer source indices", "no reinjection", "CFL velocity",
            "160-step conservation and positivity", "descending cloud", "pause", "physical bleach half-life",
            "integer rebase and clear", "uniform-flow moment", "worst-CFL checkerboard", "world curl continuity"]})
    );
}
