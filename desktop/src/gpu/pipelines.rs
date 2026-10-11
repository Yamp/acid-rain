use wgpu::*;

use crate::shaders::{
    self,
    ShaderKind,
};

pub struct ComputeStage {
    pub name: &'static str,
    pub pipeline: ComputePipeline,
    pub layout: BindGroupLayout,
}
pub struct RenderStage {
    pub pipeline: RenderPipeline,
    pub layout: BindGroupLayout,
}
pub struct Pipelines {
    pub water: RenderStage,
    pub drops: RenderStage,
    pub post: RenderStage,
    pub caustics_map: ComputeStage,
    pub caustics_render: RenderStage,
    pub cloud_normalize: ComputeStage,
    pub cloud_prepare: ComputeStage,
    pub cloud_velocity: ComputeStage,
    pub cloud_advect: ComputeStage,
    pub retina: ComputeStage,
    pub exposure: ComputeStage,
}
fn entry(
    binding: u32,
    ty: BindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
fn uniform(
    binding: u32,
    size: u64,
) -> BindGroupLayoutEntry {
    entry(binding, BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: BufferSize::new(size),
    })
}
fn storage(
    binding: u32,
    read_only: bool,
) -> BindGroupLayoutEntry {
    let mut result = entry(binding, BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    });
    if !read_only {
        result.visibility = ShaderStages::COMPUTE;
    }
    result
}
fn sampled(
    binding: u32,
    dimension: TextureViewDimension,
    filterable: bool,
) -> BindGroupLayoutEntry {
    entry(binding, BindingType::Texture {
        sample_type: TextureSampleType::Float { filterable },
        view_dimension: dimension,
        multisampled: false,
    })
}
fn image(
    binding: u32,
    dimension: TextureViewDimension,
    format: TextureFormat,
) -> BindGroupLayoutEntry {
    let mut result = entry(binding, BindingType::StorageTexture {
        access: StorageTextureAccess::WriteOnly,
        format,
        view_dimension: dimension,
    });
    result.visibility = ShaderStages::COMPUTE;
    result
}
pub fn shader(
    device: &Device,
    kind: ShaderKind,
    filtering: bool,
) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Acid Rain WGSL"),
        source: ShaderSource::Wgsl(shaders::source(kind, filtering).into()),
    })
}
fn layout(
    device: &Device,
    label: &str,
    entries: &[BindGroupLayoutEntry],
) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}
fn compute(
    device: &Device,
    common: &BindGroupLayout,
    module: &ShaderModule,
    name: &'static str,
    entries: &[BindGroupLayoutEntry],
) -> ComputeStage {
    let layout = layout(device, name, entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[&layout, common],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(name),
        compilation_options: Default::default(),
        cache: None,
    });
    ComputeStage {
        name,
        pipeline,
        layout,
    }
}
#[allow(clippy::too_many_arguments)]
fn render(
    device: &Device,
    common: &BindGroupLayout,
    module: &ShaderModule,
    vertex: &str,
    fragment: &str,
    format: TextureFormat,
    blend: Option<BlendState>,
    entries: &[BindGroupLayoutEntry],
) -> RenderStage {
    let layout = layout(device, fragment, entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(fragment),
        bind_group_layouts: &[&layout, common],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(fragment),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format,
                blend,
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });
    RenderStage { pipeline, layout }
}
impl Pipelines {
    pub fn new(
        device: &Device,
        common: &BindGroupLayout,
        format: TextureFormat,
        filtering: bool,
    ) -> Self {
        let d2 = TextureViewDimension::D2;
        let d3 = TextureViewDimension::D3;
        let array = TextureViewDimension::D2Array;
        let water_module = shader(device, ShaderKind::Water, filtering);
        let drop_module = shader(device, ShaderKind::Droplets, filtering);
        let retina_module = shader(device, ShaderKind::Retina, filtering);
        let exposure_module = shader(device, ShaderKind::Exposure, filtering);
        let cloud_module = shader(device, ShaderKind::Cloud, filtering);
        let caustics_module = shader(device, ShaderKind::Caustics, filtering);
        let water = render(
            device,
            common,
            &water_module,
            "waterVertex",
            "waterFragment",
            TextureFormat::Rgba16Float,
            None,
            &[
                uniform(0, 208),
                uniform(1, 16),
                sampled(2, d2, filtering),
                sampled(3, d2, filtering),
                sampled(4, d2, filtering),
                sampled(5, d2, filtering),
                sampled(6, d2, filtering),
                sampled(7, d3, filtering),
                sampled(8, d2, true),
            ],
        );
        let drops = render(
            device,
            common,
            &drop_module,
            "dropletVertex",
            "dropletFragment",
            TextureFormat::Rgba16Float,
            Some(BlendState::ALPHA_BLENDING),
            &[
                uniform(0, 208),
                storage(1, true),
                sampled(2, d2, filtering),
                sampled(3, d2, filtering),
            ],
        );
        let post = render(
            device,
            common,
            &retina_module,
            "waterVertex",
            "postFragment",
            format,
            None,
            &[
                uniform(0, 208),
                uniform(1, 16),
                sampled(2, d2, true),
                sampled(3, d2, true),
                sampled(4, d2, false),
                sampled(5, d2, true),
                sampled(6, d2, true),
                sampled(7, d2, true),
            ],
        );
        let retina = compute(device, common, &retina_module, "retinaHistoryUpdate", &[
            uniform(1, 16),
            sampled(2, d2, true),
            sampled(3, d2, true),
            image(8, d2, TextureFormat::Rgba16Float),
        ]);
        let exposure = compute(device, common, &exposure_module, "adaptExposure", &[
            uniform(0, 208),
            sampled(1, d2, true),
            sampled(2, d2, false),
            image(3, d2, TextureFormat::R32Float),
        ]);
        let cloud_entries = [
            uniform(0, 80),
            storage(1, true),
            storage(2, false),
            sampled(3, d3, filtering),
            image(4, d3, TextureFormat::Rgba32Float),
            sampled(5, d3, filtering),
            sampled(6, d2, filtering),
            sampled(7, d2, filtering),
        ];
        let cloud_normalize = compute(
            device,
            common,
            &cloud_module,
            "cloudNormalize",
            &cloud_entries,
        );
        let cloud_prepare = compute(
            device,
            common,
            &cloud_module,
            "cloudPrepare",
            &cloud_entries,
        );
        let cloud_velocity = compute(
            device,
            common,
            &cloud_module,
            "cloudVelocity",
            &cloud_entries,
        );
        let cloud_advect = compute(device, common, &cloud_module, "cloudAdvect", &cloud_entries);
        let caustics_map = compute(device, common, &caustics_module, "causticsMap", &[
            uniform(0, 48),
            sampled(1, d2, filtering),
            sampled(2, d2, filtering),
            image(3, array, TextureFormat::Rgba32Float),
        ]);
        let additive = BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        };
        let caustics_render = render(
            device,
            common,
            &caustics_module,
            "causticsVertex",
            "causticsFragment",
            TextureFormat::Rgba16Float,
            Some(BlendState {
                color: additive,
                alpha: additive,
            }),
            &[uniform(0, 48), sampled(4, array, false)],
        );
        Self {
            water,
            drops,
            post,
            caustics_map,
            caustics_render,
            cloud_normalize,
            cloud_prepare,
            cloud_velocity,
            cloud_advect,
            retina,
            exposure,
        }
    }
}
pub fn bind(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[(u32, BindingResource<'_>)],
) -> BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, resource)| BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Pass resources"),
        layout,
        entries: &entries,
    })
}
pub fn view(texture: &super::resources::Texture) -> BindingResource<'_> {
    BindingResource::TextureView(&texture.view)
}
pub fn dispatch(
    encoder: &mut CommandEncoder,
    stage: &ComputeStage,
    group: &BindGroup,
    common: &BindGroup,
    grid: [u32; 3],
) {
    dispatch_timed(encoder, stage, group, common, grid, None);
}
pub fn dispatch_timed(
    encoder: &mut CommandEncoder,
    stage: &ComputeStage,
    group: &BindGroup,
    common: &BindGroup,
    grid: [u32; 3],
    queries: Option<&QuerySet>,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Water simulation compute"),
        timestamp_writes: queries.map(|query_set| ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: None,
        }),
    });
    pass.set_pipeline(&stage.pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.set_bind_group(1, common, &[]);
    pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
}
pub fn render_pass<'a>(
    encoder: &'a mut CommandEncoder,
    target: &'a TextureView,
    label: &str,
    clear: bool,
) -> RenderPass<'a> {
    render_pass_timed(encoder, target, label, clear, None)
}
pub fn render_pass_timed<'a>(
    encoder: &'a mut CommandEncoder,
    target: &'a TextureView,
    label: &str,
    clear: bool,
    queries: Option<&QuerySet>,
) -> RenderPass<'a> {
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: if clear {
                    LoadOp::Clear(Color::TRANSPARENT)
                } else {
                    LoadOp::Load
                },
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: queries.map(|query_set| RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(1),
        }),
        occlusion_query_set: None,
    })
}
