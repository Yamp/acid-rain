//! The same bounded render graph serves the window, screenshots and GPU fixtures.
//! All optical textures are scene-linear until the final sRGB attachment.
mod diagnostic;
mod errors;
mod pipelines;
mod profiling;
mod resources;
use std::{
    collections::HashSet,
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering,
        },
    },
    time::Instant,
};

use acid_rain_native_core::{
    ARDrop,
    ARUniforms,
    Float4,
    OwnedSnapshot,
};
use anyhow::{
    Context,
    Result,
    bail,
    ensure,
};
pub use errors::init_diagnostics_logging;
use errors::{
    Diagnostics,
    ErrorSource,
};
use pipelines::*;
pub use profiling::GpuTiming;
use profiling::{
    TimingSlot,
    TimingState,
};
pub use resources::read_texture_bytes;
use resources::*;
use winit::window::Window;

use crate::{
    camera::{
        self,
        CameraPresentation,
    },
    quality::{
        RenderExtent,
        RenderPolicy,
        RenderQuality,
    },
    shaders::ShaderKind,
    temporal::{
        CloudClock,
        CloudTransportPlan,
        PresentationClock,
        RetinaHistoryPolicy,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct GpuConfig {
    pub width: u32,
    pub height: u32,
    pub backends: wgpu::Backends,
    pub force_fallback_adapter: bool,
    /// Explicit compatibility/diagnostic switch; does not change the render graph.
    pub timestamp_queries: bool,
}
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 800,
            backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            force_fallback_adapter: false,
            timestamp_queries: true,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RenderInput {
    pub delta: f64,
    pub now: f64,
    pub sample_age: f64,
    pub predict_camera: bool,
    pub held: bool,
    pub quality: RenderQuality,
    pub interpolation: bool,
    pub cloud_study: bool,
    pub show_falling_drops: bool,
    pub subjective_intensity: f32,
    pub backing_scale: f32,
}
impl Default for RenderInput {
    fn default() -> Self {
        Self {
            delta: 1.0 / 60.0,
            now: 0.0,
            sample_age: 0.0,
            predict_camera: false,
            held: false,
            quality: RenderQuality::High,
            interpolation: true,
            cloud_study: false,
            show_falling_drops: false,
            subjective_intensity: 1.0,
            backing_scale: 1.0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOutcome {
    pub submitted: bool,
    pub width: u32,
    pub height: u32,
    pub cpu_submit_ms: f64,
    /// Latest completed scene GPU span, not the frame submitted by this call.
    pub gpu_ms: Option<f64>,
    pub completion_latency_ms: Option<f64>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudUniforms {
    step: [f32; 4],
    rebase: [f32; 4],
    domain: [f32; 4],
    options: [f32; 4],
    dispatch: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CausticsUniforms {
    light: [f32; 4],
    domain: [f32; 4],
    grid: [u32; 4],
}

struct Targets {
    extent: RenderExtent,
    hdr: [Texture; 2],
    retina: [[Texture; 3]; 2],
}
impl Targets {
    fn new(
        device: &wgpu::Device,
        extent: RenderExtent,
    ) -> Self {
        let hdr = std::array::from_fn(|_| {
            Texture::new(
                device,
                "Linear HDR history",
                extent.width,
                extent.height,
                1,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            )
        });
        let reservoir = RetinaHistoryPolicy::extent(extent.width, extent.height);
        let retina = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                Texture::new(
                    device,
                    "Unwarped retinal reservoir",
                    reservoir.width,
                    reservoir.height,
                    1,
                    wgpu::TextureDimension::D2,
                    wgpu::TextureFormat::Rgba16Float,
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                )
            })
        });
        Self {
            extent,
            hdr,
            retina,
        }
    }
}

pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub format: wgpu::TextureFormat,
    pub adapter_info: wgpu::AdapterInfo,
    pub hardware_filtering: bool,
    pub timestamp_queries: bool,
    pub common_sampler_layout: wgpu::BindGroupLayout,
    pub common_samplers: wgpu::BindGroup,
    timers: Option<[TimingSlot; 3]>,
    slot_busy: [Arc<AtomicBool>; 3],
    timing_state: Arc<Mutex<TimingState>>,
    sequence: u64,
    last_policy_sequence: Option<u64>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    width: u32,
    height: u32,
    offline: Option<Texture>,
    pipelines: Pipelines,
    slots: Vec<FrameSlot>,
    next_slot: usize,
    targets: Targets,
    clouds: [Texture; 2],
    cloud_velocity: Texture,
    cloud_index: usize,
    caustics_rays: Texture,
    caustics_light: Texture,
    exposures: [Texture; 2],
    history_index: usize,
    history_valid: bool,
    retina_index: usize,
    retina_policy: RetinaHistoryPolicy,
    cloud_clock: CloudClock,
    presentation_clock: PresentationClock,
    camera: CameraPresentation,
    policy: RenderPolicy,
    consumed: HashSet<u64>,
    previous_mode: Option<[u32; 2]>,
    has_rendered: bool,
    diagnose_passes: bool,
    picking: Option<(OwnedSnapshot, ARUniforms)>,
    in_flight: Arc<AtomicUsize>,
    error: Arc<Mutex<Diagnostics>>,
}
impl GpuRenderer {
    pub async fn new(
        window: Option<Arc<Window>>,
        config: GpuConfig,
    ) -> Result<Self> {
        let backends = config.backends
            & (wgpu::Backends::DX12 | wgpu::Backends::VULKAN | wgpu::Backends::METAL);
        ensure!(
            !backends.is_empty(),
            "Choose a DX12, Vulkan or Metal backend"
        );
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = window
            .map(|window| instance.create_surface(window))
            .transpose()
            .context("Cannot create a GPU window surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: config.force_fallback_adapter,
                compatible_surface: surface.as_ref(),
            })
            .await
            .context(
                "No compatible graphics adapter: DX12/Vulkan/Metal unavailable; update the GPU \
                 driver or select another --backend",
            )?;
        let adapter_info = adapter.get_info();
        let hardware_filtering = adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        for (format, usage) in [
            (
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            ),
            (
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            (
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            ),
        ] {
            ensure!(
                adapter
                    .get_texture_format_features(format)
                    .allowed_usages
                    .contains(usage),
                "Adapter {} ({:?}) cannot use {:?} for the full water pipeline",
                adapter_info.name,
                adapter_info.backend,
                format
            );
        }
        ensure!(
            adapter
                .get_texture_format_features(wgpu::TextureFormat::Rgba16Float)
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE),
            "Adapter {} does not support HDR additive caustic blending",
            adapter_info.name
        );
        let timestamp_queries = config.timestamp_queries
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = if hardware_filtering {
            wgpu::Features::FLOAT32_FILTERABLE
        } else {
            wgpu::Features::empty()
        };
        if timestamp_queries {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let limits = wgpu::Limits::default().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Acid Rain complete GPU graph"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .context("Cannot initialize the requested GPU device")?;
        let error = Arc::new(Mutex::new(Diagnostics::default()));
        let errors = error.clone();
        device.on_uncaptured_error(Arc::new(move |failure: wgpu::Error| {
            errors
                .lock()
                .unwrap()
                .record(ErrorSource::Rendering, failure.to_string());
        }));
        let errors = error.clone();
        device.set_device_lost_callback(move |reason, message| {
            errors.lock().unwrap().record(
                ErrorSource::DeviceLost,
                format!("GPU device lost ({reason:?}): {message}"),
            );
        });
        let width = config.width.max(1);
        let height = config.height.max(1);
        ensure!(
            width <= device.limits().max_texture_dimension_2d
                && height <= device.limits().max_texture_dimension_2d,
            "Requested output exceeds GPU texture limits"
        );
        let (format, surface_config) = if let Some(surface) = surface.as_ref() {
            let capabilities = surface.get_capabilities(&adapter);
            let format = capabilities
                .formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .context("The window surface has no sRGB output format")?;
            let mut settings = surface
                .get_default_config(&adapter, width, height)
                .context("No valid surface configuration")?;
            settings.format = format;
            settings.present_mode = wgpu::PresentMode::Fifo;
            settings.desired_maximum_frame_latency = 2;
            surface.configure(&device, &settings);
            (format, Some(settings))
        } else {
            (wgpu::TextureFormat::Rgba8UnormSrgb, None)
        };
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let common_sampler_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Common optical samplers"),
                entries: &std::array::from_fn::<_, 3, _>(|binding| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }),
            });
        let sampler = |u, v| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: u,
                address_mode_v: v,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            })
        };
        let linear = sampler(
            wgpu::AddressMode::ClampToEdge,
            wgpu::AddressMode::ClampToEdge,
        );
        let sky = sampler(wgpu::AddressMode::Repeat, wgpu::AddressMode::ClampToEdge);
        let mirror = sampler(
            wgpu::AddressMode::MirrorRepeat,
            wgpu::AddressMode::MirrorRepeat,
        );
        let common_samplers = bind(&device, &common_sampler_layout, &[
            (0, wgpu::BindingResource::Sampler(&linear)),
            (1, wgpu::BindingResource::Sampler(&sky)),
            (2, wgpu::BindingResource::Sampler(&mirror)),
        ]);
        let pipelines = Pipelines::new(&device, &common_sampler_layout, format, hardware_filtering);
        if let Some(failure) = device.pop_error_scope().await {
            bail!(
                "GPU shader/pipeline initialization on {} ({:?}): {failure}",
                adapter_info.name,
                adapter_info.backend
            );
        }
        let volume = |label| {
            Texture::new(
                &device,
                label,
                96,
                64,
                32,
                wgpu::TextureDimension::D3,
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
        };
        let clouds = [volume("Pigment volume A"), volume("Pigment volume B")];
        let cloud_velocity = volume("Cloud velocity");
        let caustics_rays = Texture::new(
            &device,
            "Spectral caustic rays",
            129,
            129,
            3,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let caustics_light = Texture::new(
            &device,
            "Caustic irradiance",
            256,
            256,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let exposures = std::array::from_fn(|_| initialized_exposure(&device, &queue));
        let slots = (0..3).map(|_| FrameSlot::new(&device)).collect();
        let targets = Targets::new(
            &device,
            RenderPolicy::extent(width, height, RenderQuality::High, 1_200_000),
        );
        let offline = if surface.is_none() {
            Some(Self::offline_texture(&device, width, height, format))
        } else {
            None
        };
        let timers = timestamp_queries.then(|| std::array::from_fn(|_| TimingSlot::new(&device)));
        Ok(Self {
            device,
            queue,
            format,
            adapter_info,
            hardware_filtering,
            timestamp_queries,
            timers,
            slot_busy: std::array::from_fn(|_| Arc::new(AtomicBool::new(false))),
            timing_state: Arc::new(Mutex::new(TimingState::default())),
            sequence: 0,
            last_policy_sequence: None,
            common_sampler_layout,
            common_samplers,
            surface,
            surface_config,
            width: config.width,
            height: config.height,
            offline,
            pipelines,
            slots,
            next_slot: 0,
            targets,
            clouds,
            cloud_velocity,
            cloud_index: 0,
            caustics_rays,
            caustics_light,
            exposures,
            history_index: 0,
            history_valid: false,
            retina_index: 0,
            retina_policy: Default::default(),
            cloud_clock: Default::default(),
            presentation_clock: Default::default(),
            camera: Default::default(),
            policy: Default::default(),
            consumed: HashSet::new(),
            previous_mode: None,
            has_rendered: false,
            diagnose_passes: false,
            picking: None,
            in_flight: Arc::new(AtomicUsize::new(0)),
            error,
        })
    }

    fn offline_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Texture {
        Texture::new(
            device,
            "Offline color-managed output",
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        )
    }

    pub fn shader(
        &self,
        kind: ShaderKind,
    ) -> wgpu::ShaderModule {
        pipelines::shader(&self.device, kind, self.hardware_filtering)
    }

    /// Diagnostic-only: serialize the unchanged production graph at pass boundaries.
    /// Never valid for performance measurement or interactive presentation.
    pub fn enable_pass_diagnostics(&mut self) -> Result<()> {
        ensure!(
            self.surface.is_none(),
            "GPU pass diagnostics require headless rendering"
        );
        ensure!(
            !self.has_rendered,
            "Enable GPU pass diagnostics before rendering"
        );
        self.diagnose_passes = true;
        Ok(())
    }

    pub fn pass_diagnostics_enabled(&self) -> bool { self.diagnose_passes }

    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
        self.history_valid = false;
        self.retina_policy.invalidate();
        if width == 0 || height == 0 {
            return;
        }
        if width > self.device.limits().max_texture_dimension_2d
            || height > self.device.limits().max_texture_dimension_2d
        {
            self.error.lock().unwrap().record(
                ErrorSource::Rendering,
                "Window dimensions exceed GPU texture limits".into(),
            );
            return;
        }
        if let Some(config) = self.surface_config.as_mut() {
            config.width = width;
            config.height = height;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, config);
            }
        } else {
            self.offline = Some(Self::offline_texture(
                &self.device,
                width,
                height,
                self.format,
            ));
        }
    }

    pub fn surface_point(
        &self,
        uv: [f32; 2],
    ) -> Option<[f64; 2]> {
        let (frame, uniforms) = self.picking.as_ref()?;
        let flat = bytemuck::cast_slice(frame.water.as_ref());
        let point = camera::surface_point(
            uv,
            uniforms,
            flat,
            frame.water_width as usize,
            frame.water_height as usize,
        )?;
        Some([
            frame.world_origin[0] + f64::from(point[0]),
            frame.world_origin[1] + f64::from(point[1]),
        ])
    }

    pub fn capture_rgba(&self) -> Result<Vec<u8>> {
        let texture = self
            .offline
            .as_ref()
            .context("Capture requires the headless renderer")?;
        let result = read_texture_bytes(&self.device, &self.queue, &texture.texture, 4);
        self.check_error()?;
        result
    }

    /// Explicit synchronization for captures/tests only; interactive rendering never waits.
    pub fn wait_idle(&self) -> Result<()> {
        let result = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.check_error()?;
        result?;
        Ok(())
    }

    pub fn take_timings(&self) -> Vec<GpuTiming> {
        self.timing_state
            .lock()
            .unwrap()
            .records
            .drain(..)
            .collect()
    }

    pub fn profiling_warning(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap()
            .profiling_warning()
            .map(str::to_owned)
    }

    pub fn cloud_texture(&self) -> &wgpu::Texture { &self.clouds[self.cloud_index].texture }

    pub fn caustics_texture(&self) -> &wgpu::Texture { &self.caustics_light.texture }

    fn check_error(&self) -> Result<()> { errors::check(&self.device, &self.error) }

    pub fn render<F>(
        &mut self,
        frame: &OwnedSnapshot,
        input: &RenderInput,
        overlay: F,
    ) -> Result<RenderOutcome>
    where
        F: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            wgpu::TextureFormat,
            [u32; 2],
        ),
    {
        let started = Instant::now();
        let polled = self.device.poll(wgpu::PollType::Poll);
        self.check_error()?;
        polled?;
        if self.timers.is_some() && self.profiling_warning().is_some() {
            self.timers = None;
            self.timestamp_queries = false;
            eprintln!(
                "{}; using queue-completion measurements",
                self.profiling_warning().unwrap()
            );
        }
        if self.width == 0 || self.height == 0 {
            return Ok(RenderOutcome::default());
        }
        if self.in_flight.load(Ordering::Acquire) >= 3 {
            if self.surface.is_some() {
                return Ok(RenderOutcome::default());
            }
            let polled = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.check_error()?;
            polled?;
        }
        let slot_index = (0..3)
            .map(|offset| (self.next_slot + offset) % 3)
            .find(|&index| !self.slot_busy[index].load(Ordering::Acquire))
            .context("No free GPU frame slot after completion polling")?;
        self.next_slot = (slot_index + 1) % 3;
        let last_timing = self.timing_state.lock().unwrap().last;
        if let Some(timing) = last_timing
            && input.quality == RenderQuality::Auto
            && self.last_policy_sequence != Some(timing.sequence)
        {
            self.policy.observe(
                timing
                    .gpu_ms
                    .unwrap_or(timing.completion_ms / f64::from(timing.queue_depth.max(1))),
            );
            self.last_policy_sequence = Some(timing.sequence);
        }
        let acquired = if let Some(surface) = &self.surface {
            match surface.get_current_texture() {
                Ok(frame) => Some(frame),
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    surface.configure(&self.device, self.surface_config.as_ref().unwrap());
                    return Ok(RenderOutcome::default());
                },
                Err(wgpu::SurfaceError::Timeout) => return Ok(RenderOutcome::default()),
                Err(error) => return Err(error).context("Cannot acquire the window render target"),
            }
        } else {
            None
        };
        let destination = if let Some(acquired) = &acquired {
            acquired.texture.create_view(&Default::default())
        } else {
            self.offline.as_ref().unwrap().view.clone()
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cloud → caustics → water → retina"),
            });
        self.encode(slot_index, frame, input, &mut encoder, &destination)?;
        overlay(
            &self.device,
            &self.queue,
            &mut encoder,
            &destination,
            self.format,
            [self.width, self.height],
        );
        if let Some(timers) = &self.timers {
            timers[slot_index].resolve(&mut encoder);
            diagnostic::checkpoint(
                self.diagnose_passes,
                "timestamp resolve/copy",
                &mut encoder,
                &self.device,
                &self.queue,
                &self.slots[slot_index].uniforms,
                &self.error,
            )?;
        }
        self.slot_busy[slot_index].store(true, Ordering::Release);
        let depth = self.in_flight.fetch_add(1, Ordering::AcqRel) as u32 + 1;
        self.queue.submit([encoder.finish()]);
        self.sequence += 1;
        if let Some(timers) = &self.timers {
            timers[slot_index].complete(
                &self.queue,
                self.queue.get_timestamp_period(),
                self.sequence,
                Instant::now(),
                depth,
                self.timing_state.clone(),
                self.slot_busy[slot_index].clone(),
                self.in_flight.clone(),
                self.error.clone(),
            );
        } else {
            profiling::fallback_completion(
                &self.queue,
                self.sequence,
                Instant::now(),
                depth,
                self.timing_state.clone(),
                self.slot_busy[slot_index].clone(),
                self.in_flight.clone(),
            );
        }
        if let Some(acquired) = acquired {
            acquired.present();
        }
        if self.diagnose_passes {
            eprintln!("GPU CHECKPOINT BEGIN final completion/timestamp mapping");
            self.wait_idle()?;
            eprintln!("GPU CHECKPOINT PASS final completion/timestamp mapping");
        }
        self.has_rendered = true;
        Ok(RenderOutcome {
            submitted: true,
            width: self.targets.extent.width,
            height: self.targets.extent.height,
            cpu_submit_ms: started.elapsed().as_secs_f64() * 1000.0,
            gpu_ms: last_timing.and_then(|t| t.gpu_ms),
            completion_latency_ms: last_timing.map(|t| t.completion_ms),
        })
    }

    fn encode(
        &mut self,
        slot_index: usize,
        frame: &OwnedSnapshot,
        input: &RenderInput,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::TextureView,
    ) -> Result<()> {
        let extent = RenderPolicy::extent(
            self.width,
            self.height,
            input.quality,
            self.policy.automatic_pixel_budget(),
        );
        if self.targets.extent != extent {
            self.targets = Targets::new(&self.device, extent);
            self.history_valid = false;
            self.retina_policy.invalidate();
            self.retina_index = 0;
        }
        self.slots[slot_index].upload(&self.queue, frame)?;
        // Query boundaries belong to actual dependent work: the first cloud
        // pass (or caustics if unchanged) and final post draw, not independent
        // marker commands which a backend may overlap with the optical graph.
        let query_set = self
            .timers
            .as_ref()
            .map(|timers| timers[slot_index].queries().clone());
        let mut start_timer = query_set.as_ref();
        let airborne = self.encode_clouds(slot_index, frame, encoder, &mut start_timer)?;
        let held = input.held || frame.paused;
        let scene_mode = [frame.scene, frame.mode];
        if self.previous_mode != Some(scene_mode) {
            self.previous_mode = Some(scene_mode);
            self.history_valid = false;
            self.retina_policy.invalidate();
        }
        if self.has_rendered && input.delta == 0.0 {
            self.presentation_clock.reanchor(input.now);
        }
        let mut uniforms = frame.uniforms;
        let scene_time = self.presentation_clock.sample(
            f64::from(uniforms.time.0[0]),
            if input.predict_camera {
                input.now - input.sample_age.max(0.0)
            } else {
                input.now
            },
            f64::from(frame.speed),
            input.now,
            held,
        );
        uniforms.time.0[0] = scene_time as f32;
        uniforms.time.0[1] = if held {
            0.0
        } else {
            input.delta.clamp(0.0, 0.1) as f32
        };
        uniforms.post.0[1] = input.quality.shader_level();
        uniforms.post.0[2] = if self.history_valid && input.interpolation && !held {
            0.13
        } else {
            0.0
        };
        uniforms.post.0[3] = input.backing_scale.max(0.1);
        uniforms.illumination.0[3] = if input.cloud_study { 1.0 } else { 0.0 };
        if input.cloud_study {
            uniforms.optics.0[0] *= 0.65;
        }
        self.camera.fill(
            &mut uniforms,
            frame.world_origin,
            frame.uniforms.time.0[0],
            scene_time as f32,
            if input.delta > 0.25 { 0.0 } else { input.delta },
            held,
            self.width as f32 / self.height as f32,
        );
        self.picking = Some((frame.clone(), uniforms));
        self.encode_caustics(slot_index, &uniforms, encoder, &mut start_timer)?;
        let current = self.history_index;
        let previous = 1 - current;
        uniforms.viewport = Float4([
            extent.width as f32,
            extent.height as f32,
            1.0 / extent.width as f32,
            1.0 / extent.height as f32,
        ]);
        let slot = &mut self.slots[slot_index];
        let water_constants = slot
            .uniforms
            .push(bytemuck::cast_slice(&uniform_values(&uniforms)))?;
        let bed_world = [
            frame.world_origin[0] as f32,
            frame.world_origin[1] as f32,
            frame.physical_time as f32,
            if input.cloud_study { 0.0 } else { 1.0 },
        ];
        let bed_constants = slot.uniforms.push(bytemuck::bytes_of(&bed_world))?;
        let group = bind(&self.device, &self.pipelines.water.layout, &[
            (0, slot.uniforms.bind(water_constants)),
            (1, slot.uniforms.bind(bed_constants)),
            (2, view(&slot.water)),
            (3, view(&slot.dye)),
            (4, view(&slot.bed)),
            (5, view(&slot.sky)),
            (6, view(&slot.light)),
            (7, view(&self.clouds[self.cloud_index])),
            (8, view(&self.caustics_light)),
        ]);
        {
            let mut pass = render_pass(
                encoder,
                &self.targets.hdr[current].view,
                "Water and airborne pigment",
                true,
            );
            pass.set_pipeline(&self.pipelines.water.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &self.common_samplers, &[]);
            pass.draw(0..3, 0..1);
            if input.show_falling_drops && airborne > 0 {
                let drops = bind(&self.device, &self.pipelines.drops.layout, &[
                    (0, slot.uniforms.bind(water_constants)),
                    (1, slot.airborne.as_entire_binding()),
                    (2, view(&slot.water)),
                    (3, view(&slot.sky)),
                ]);
                pass.set_pipeline(&self.pipelines.drops.pipeline);
                pass.set_bind_group(0, &drops, &[]);
                pass.draw(0..6, 0..airborne);
            }
        }
        diagnostic::checkpoint(
            self.diagnose_passes,
            "water/droplets",
            encoder,
            &self.device,
            &self.queue,
            &slot.uniforms,
            &self.error,
        )?;
        let retina = self.retina_policy.advance(input.delta, held);
        if retina.update {
            let next = 1 - self.retina_index;
            for layer in 0..3 {
                let parameters = [
                    retina.alphas[layer],
                    if retina.seed { 0.0 } else { 1.0 },
                    0.0,
                    0.0,
                ];
                let constants = slot.uniforms.push(bytemuck::bytes_of(&parameters))?;
                let output = &self.targets.retina[next][layer];
                let group = bind(&self.device, &self.pipelines.retina.layout, &[
                    (1, slot.uniforms.bind(constants)),
                    (2, view(&self.targets.hdr[current])),
                    (3, view(&self.targets.retina[self.retina_index][layer])),
                    (8, view(output)),
                ]);
                dispatch(
                    encoder,
                    &self.pipelines.retina,
                    &group,
                    &self.common_samplers,
                    [
                        output.texture.width().div_ceil(8),
                        output.texture.height().div_ceil(8),
                        1,
                    ],
                );
                diagnostic::checkpoint(
                    self.diagnose_passes,
                    [
                        "retinal reservoir fast",
                        "retinal reservoir middle",
                        "retinal reservoir slow",
                    ][layer],
                    encoder,
                    &self.device,
                    &self.queue,
                    &slot.uniforms,
                    &self.error,
                )?;
            }
            self.retina_index = next;
        }
        let exposure = bind(&self.device, &self.pipelines.exposure.layout, &[
            (0, slot.uniforms.bind(water_constants)),
            (1, view(&self.targets.hdr[current])),
            (2, view(&self.exposures[previous])),
            (3, view(&self.exposures[current])),
        ]);
        dispatch(
            encoder,
            &self.pipelines.exposure,
            &exposure,
            &self.common_samplers,
            [1, 1, 1],
        );
        diagnostic::checkpoint(
            self.diagnose_passes,
            "pupil exposure",
            encoder,
            &self.device,
            &self.queue,
            &slot.uniforms,
            &self.error,
        )?;
        uniforms.viewport = Float4([
            self.width as f32,
            self.height as f32,
            1.0 / self.width as f32,
            1.0 / self.height as f32,
        ]);
        let post_constants = slot
            .uniforms
            .push(bytemuck::cast_slice(&uniform_values(&uniforms)))?;
        let retina_state = [
            retina.time,
            input.subjective_intensity.clamp(0.0, 2.0),
            1.0,
            0.0,
        ];
        let retina_constants = slot.uniforms.push(bytemuck::bytes_of(&retina_state))?;
        let post = bind(&self.device, &self.pipelines.post.layout, &[
            (0, slot.uniforms.bind(post_constants)),
            (1, slot.uniforms.bind(retina_constants)),
            (2, view(&self.targets.hdr[current])),
            (3, view(&self.targets.hdr[previous])),
            (4, view(&self.exposures[current])),
            (5, view(&self.targets.retina[self.retina_index][0])),
            (6, view(&self.targets.retina[self.retina_index][1])),
            (7, view(&self.targets.retina[self.retina_index][2])),
        ]);
        {
            let mut pass = render_pass_timed(
                encoder,
                destination,
                "Retinal optics and sRGB output",
                true,
                query_set.as_ref(),
            );
            pass.set_pipeline(&self.pipelines.post.pipeline);
            pass.set_bind_group(0, &post, &[]);
            pass.set_bind_group(1, &self.common_samplers, &[]);
            pass.draw(0..3, 0..1);
        }
        diagnostic::checkpoint(
            self.diagnose_passes,
            "retinal post/sRGB output",
            encoder,
            &self.device,
            &self.queue,
            &slot.uniforms,
            &self.error,
        )?;
        slot.uniforms.upload(&self.queue);
        self.history_index = previous;
        self.history_valid = true;
        Ok(())
    }

    fn encode_caustics(
        &mut self,
        slot_index: usize,
        u: &ARUniforms,
        encoder: &mut wgpu::CommandEncoder,
        start_timer: &mut Option<&wgpu::QuerySet>,
    ) -> Result<()> {
        let slot = &mut self.slots[slot_index];
        let parameters = CausticsUniforms {
            light: [u.light.0[0], u.light.0[1], u.light.0[2], 0.02],
            domain: [2.0, -0.5, 3.0, 1.0],
            grid: [128, 256, 256, 0],
        };
        let constants = slot.uniforms.push(bytemuck::bytes_of(&parameters))?;
        let mapping = bind(&self.device, &self.pipelines.caustics_map.layout, &[
            (0, slot.uniforms.bind(constants)),
            (1, view(&slot.water)),
            (2, view(&slot.dye)),
            (3, view(&self.caustics_rays)),
        ]);
        dispatch_timed(
            encoder,
            &self.pipelines.caustics_map,
            &mapping,
            &self.common_samplers,
            [17, 17, 3],
            start_timer.take(),
        );
        diagnostic::checkpoint(
            self.diagnose_passes,
            "caustics map",
            encoder,
            &self.device,
            &self.queue,
            &slot.uniforms,
            &self.error,
        )?;
        let triangles = bind(&self.device, &self.pipelines.caustics_render.layout, &[
            (0, slot.uniforms.bind(constants)),
            (4, view(&self.caustics_rays)),
        ]);
        let mut pass = render_pass(
            encoder,
            &self.caustics_light.view,
            "Conservative spectral caustics",
            true,
        );
        pass.set_pipeline(&self.pipelines.caustics_render.pipeline);
        pass.set_bind_group(0, &triangles, &[]);
        pass.set_bind_group(1, &self.common_samplers, &[]);
        pass.draw(0..128 * 128 * 6, 0..3);
        drop(pass);
        diagnostic::checkpoint(
            self.diagnose_passes,
            "caustics raster",
            encoder,
            &self.device,
            &self.queue,
            &slot.uniforms,
            &self.error,
        )?;
        Ok(())
    }

    fn encode_clouds(
        &mut self,
        slot_index: usize,
        frame: &OwnedSnapshot,
        encoder: &mut wgpu::CommandEncoder,
        start_timer: &mut Option<&wgpu::QuerySet>,
    ) -> Result<u32> {
        let step = self.cloud_clock.consume(
            frame.physical_time,
            frame.world_origin,
            frame.cloud_revision,
        );
        if step.clear {
            self.consumed.clear();
        }
        let id = |drop: &ARDrop| u64::from(drop.identity[0]) | (u64::from(drop.identity[1]) << 32);
        let retained: HashSet<_> = frame.drops.iter().map(id).collect();
        self.consumed.retain(|id| retained.contains(id));
        let pending: Vec<_> = frame
            .drops
            .iter()
            .filter(|d| d.identity[2] == 1 && !self.consumed.contains(&id(d)))
            .copied()
            .collect();
        let plan = CloudTransportPlan::new(
            step.delta,
            frame.physical_time,
            &pending
                .iter()
                .map(|d| f64::from(d.dynamics.0[2]))
                .collect::<Vec<_>>(),
        );
        let sources: Vec<_> = plan
            .source_order
            .iter()
            .map(|&index| pending[index])
            .collect();
        for source in &sources {
            self.consumed.insert(id(source));
        }
        let airborne: Vec<_> = frame
            .drops
            .iter()
            .filter(|d| d.identity[2] == 0)
            .copied()
            .collect();
        let mut wakes: Vec<_> = frame
            .drops
            .iter()
            .filter(|d| {
                let age = frame.physical_time - f64::from(d.dynamics.0[2]);
                d.identity[2] == 1 && (0.0..=2.5).contains(&age)
            })
            .copied()
            .collect();
        if wakes.len() > 24 {
            wakes.drain(..wakes.len() - 24);
        }
        ensure!(
            sources.len() <= 512 && airborne.len() <= 512,
            "Simulation impact journal exceeds GPU capacity"
        );
        let slot = &mut self.slots[slot_index];
        upload_drops(&self.queue, &slot.sources, &sources);
        upload_drops(&self.queue, &slot.airborne, &airborne);
        upload_drops(&self.queue, &slot.wakes, &wakes);
        let mut uniforms = CloudUniforms {
            step: [0.0, frame.physical_time as f32, frame.bleach, 0.0],
            rebase: [0.0; 4],
            domain: [2.0, 2.0, -0.8, 0.9],
            options: [
                (frame.world_origin[0] % std::f64::consts::TAU) as f32,
                1.0,
                0.00001,
                (frame.world_origin[1] % std::f64::consts::TAU) as f32,
            ],
            dispatch: [0; 4],
        };
        if step.clear || step.shift != [0, 0] {
            uniforms.rebase = [
                step.shift[0] as f32,
                step.shift[1] as f32,
                if step.clear { 1.0 } else { 0.0 },
                0.0,
            ];
            cloud_pass(
                &self.device,
                &self.queue,
                self.diagnose_passes,
                &self.error,
                encoder,
                &self.common_samplers,
                &self.pipelines.cloud_prepare,
                slot,
                &self.clouds[self.cloud_index],
                &self.clouds[1 - self.cloud_index],
                &self.cloud_velocity,
                &uniforms,
                false,
                start_timer,
                [24, 16, 8],
            )?;
            self.cloud_index = 1 - self.cloud_index;
            uniforms.rebase = [0.0; 4];
        }
        if !sources.is_empty() {
            uniforms.step[3] = sources.len() as f32;
            cloud_pass(
                &self.device,
                &self.queue,
                self.diagnose_passes,
                &self.error,
                encoder,
                &self.common_samplers,
                &self.pipelines.cloud_normalize,
                slot,
                &self.clouds[self.cloud_index],
                &self.clouds[1 - self.cloud_index],
                &self.cloud_velocity,
                &uniforms,
                false,
                start_timer,
                [sources.len() as u32, 1, 1],
            )?;
        }
        for boundary in 0..=plan.substep_count {
            if boundary > 0 {
                if boundary == 1 {
                    uniforms.step[1] = (frame.physical_time - step.delta * 0.5) as f32;
                    uniforms.step[3] = wakes.len() as f32;
                    uniforms.dispatch = [0; 4];
                    cloud_pass(
                        &self.device,
                        &self.queue,
                        self.diagnose_passes,
                        &self.error,
                        encoder,
                        &self.common_samplers,
                        &self.pipelines.cloud_velocity,
                        slot,
                        &self.clouds[self.cloud_index],
                        &self.cloud_velocity,
                        &self.clouds[self.cloud_index],
                        &uniforms,
                        true,
                        start_timer,
                        [24, 16, 8],
                    )?;
                }
                uniforms.step[0] = plan.substep_delta as f32;
                uniforms.step[3] = 0.0;
                uniforms.rebase = [0.0; 4];
                uniforms.dispatch = [0; 4];
                cloud_pass(
                    &self.device,
                    &self.queue,
                    self.diagnose_passes,
                    &self.error,
                    encoder,
                    &self.common_samplers,
                    &self.pipelines.cloud_advect,
                    slot,
                    &self.clouds[self.cloud_index],
                    &self.clouds[1 - self.cloud_index],
                    &self.cloud_velocity,
                    &uniforms,
                    false,
                    start_timer,
                    [24, 16, 8],
                )?;
                self.cloud_index = 1 - self.cloud_index;
            }
            let range = &plan.injections[boundary];
            if !range.is_empty() {
                uniforms.rebase = [0.0, 0.0, 0.0, 1.0];
                uniforms.step[3] = range.len() as f32;
                uniforms.dispatch = [range.start as u32, range.start as u32, 0, 0];
                cloud_pass(
                    &self.device,
                    &self.queue,
                    self.diagnose_passes,
                    &self.error,
                    encoder,
                    &self.common_samplers,
                    &self.pipelines.cloud_prepare,
                    slot,
                    &self.clouds[self.cloud_index],
                    &self.clouds[1 - self.cloud_index],
                    &self.cloud_velocity,
                    &uniforms,
                    false,
                    start_timer,
                    [24, 16, 8],
                )?;
                self.cloud_index = 1 - self.cloud_index;
                uniforms.rebase = [0.0; 4];
            }
        }
        Ok(airborne.len() as u32)
    }
}
#[allow(clippy::too_many_arguments)]
fn cloud_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    diagnose_passes: bool,
    errors: &Arc<Mutex<Diagnostics>>,
    encoder: &mut wgpu::CommandEncoder,
    common: &wgpu::BindGroup,
    stage: &ComputeStage,
    slot: &mut FrameSlot,
    input: &Texture,
    output: &Texture,
    velocity: &Texture,
    u: &CloudUniforms,
    wakes: bool,
    start_timer: &mut Option<&wgpu::QuerySet>,
    grid: [u32; 3],
) -> Result<()> {
    let constants = slot.uniforms.push(bytemuck::bytes_of(u))?;
    let events = if wakes { &slot.wakes } else { &slot.sources };
    let group = bind(device, &stage.layout, &[
        (0, slot.uniforms.bind(constants)),
        (1, events.as_entire_binding()),
        (2, slot.normalization.as_entire_binding()),
        (3, view(input)),
        (4, view(output)),
        (5, view(velocity)),
        (6, view(&slot.flow)),
        (7, view(&slot.water)),
    ]);
    dispatch_timed(encoder, stage, &group, common, grid, start_timer.take());
    diagnostic::checkpoint(
        diagnose_passes,
        stage.name,
        encoder,
        device,
        queue,
        &slot.uniforms,
        errors,
    )?;
    Ok(())
}
