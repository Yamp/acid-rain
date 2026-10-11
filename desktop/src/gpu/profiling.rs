use std::{
    collections::VecDeque,
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

use super::errors::{
    Diagnostics,
    ErrorSource,
};

#[derive(Clone, Copy, Debug)]
pub struct GpuTiming {
    pub sequence: u64,
    /// Scene GPU span from its first compute pass through the final post draw.
    /// Excludes CPU uploads and the optional UI overlay; absent without queries.
    pub gpu_ms: Option<f64>,
    /// CPU submit-to-completion latency, including queued frames and callback service.
    pub completion_ms: f64,
    pub queue_depth: u32,
}
#[derive(Default)]
pub struct TimingState {
    pub last: Option<GpuTiming>,
    pub records: VecDeque<GpuTiming>,
}
impl TimingState {
    fn push(
        &mut self,
        timing: GpuTiming,
    ) {
        self.last = Some(timing);
        if self.records.len() >= 2048 {
            self.records.pop_front();
        }
        self.records.push_back(timing);
    }
}
pub struct TimingSlot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}
impl TimingSlot {
    pub fn new(device: &wgpu::Device) -> Self {
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Scene render graph GPU timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Asynchronous timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            queries,
            resolve,
            readback,
        }
    }

    pub fn queries(&self) -> &wgpu::QuerySet { &self.queries }

    pub fn resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 16);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn complete(
        &self,
        queue: &wgpu::Queue,
        period: f32,
        sequence: u64,
        start: Instant,
        depth: u32,
        state: Arc<Mutex<TimingState>>,
        busy: Arc<AtomicBool>,
        in_flight: Arc<AtomicUsize>,
        error: Arc<Mutex<Diagnostics>>,
    ) {
        let buffer = self.readback.clone();
        let queue = queue.clone();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                match result {
                    Ok(()) => {
                        let mapped = buffer.slice(..).get_mapped_range();
                        let begin = u64::from_ne_bytes(mapped[0..8].try_into().unwrap());
                        let end = u64::from_ne_bytes(mapped[8..16].try_into().unwrap());
                        let gpu_ms = (end > begin && begin > 0).then_some(
                            (end.saturating_sub(begin)) as f64 * f64::from(period) * 1e-6,
                        );
                        drop(mapped);
                        buffer.unmap();
                        state.lock().unwrap().push(GpuTiming {
                            sequence,
                            gpu_ms,
                            completion_ms: start.elapsed().as_secs_f64() * 1000.0,
                            queue_depth: depth,
                        });
                    },
                    Err(failure) => {
                        error.lock().unwrap().record(
                            ErrorSource::Profiling,
                            format!("GPU timestamp readback failed: {failure}"),
                        );
                        // An aborted map may arrive before submitted GPU work
                        // completes. Keep this upload slot unavailable until
                        // queue completion; a lost device remains fatal in the
                        // independent diagnostics state.
                        fallback_completion(&queue, sequence, start, depth, state, busy, in_flight);
                        return;
                    },
                }
                busy.store(false, Ordering::Release);
                in_flight.fetch_sub(1, Ordering::AcqRel);
            });
    }
}

pub fn fallback_completion(
    queue: &wgpu::Queue,
    sequence: u64,
    start: Instant,
    depth: u32,
    state: Arc<Mutex<TimingState>>,
    busy: Arc<AtomicBool>,
    in_flight: Arc<AtomicUsize>,
) {
    queue.on_submitted_work_done(move || {
        state.lock().unwrap().push(GpuTiming {
            sequence,
            gpu_ms: None,
            completion_ms: start.elapsed().as_secs_f64() * 1000.0,
            queue_depth: depth,
        });
        busy.store(false, Ordering::Release);
        in_flight.fetch_sub(1, Ordering::AcqRel);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump_driver_messages(device: &wgpu::Device) {
        #[cfg(not(target_os = "windows"))]
        let _ = device;
        #[cfg(target_os = "windows")]
        {
            use windows::{
                Win32::Graphics::Direct3D12::{
                    D3D12_MESSAGE,
                    ID3D12InfoQueue,
                },
                core::Interface,
            };
            // SAFETY: the HAL guard keeps the native device alive. We only read
            // its debug queue, never submit or mutate graphics resources.
            let Some(raw) = (unsafe { device.as_hal::<wgpu::hal::api::Dx12>() }) else {
                return;
            };
            let Ok(messages) = raw.raw_device().cast::<ID3D12InfoQueue>() else {
                eprintln!("DX12 validation message queue unavailable (debug layer missing)");
                return;
            };
            // SAFETY: each buffer is allocated with D3D12_MESSAGE alignment and
            // the native reported byte length before GetMessage writes into it.
            unsafe {
                let count = messages.GetNumStoredMessagesAllowedByRetrievalFilter();
                eprintln!("DX12 validation messages: {count}");
                for index in 0..count.min(256) {
                    let mut length = 0;
                    if messages.GetMessage(index, None, &mut length).is_err()
                        || length < std::mem::size_of::<D3D12_MESSAGE>()
                    {
                        continue;
                    }
                    let mut storage = vec![
                        D3D12_MESSAGE::default();
                        length.div_ceil(std::mem::size_of::<D3D12_MESSAGE>())
                    ];
                    if messages
                        .GetMessage(index, Some(storage.as_mut_ptr()), &mut length)
                        .is_ok()
                    {
                        let message = &storage[0];
                        let description = if message.pDescription.is_null() {
                            std::borrow::Cow::Borrowed("<no description>")
                        } else {
                            String::from_utf8_lossy(std::slice::from_raw_parts(
                                message.pDescription,
                                message.DescriptionByteLength,
                            ))
                        };
                        eprintln!(
                            "DX12 validation severity={} id={}: {}",
                            message.Severity.0,
                            message.ID.0,
                            description.trim_end_matches('\0')
                        );
                    }
                }
                messages.ClearStoredMessages();
            }
        }
    }

    /// Deliberately tiny work isolates native query recording/resolution from
    /// the water shaders. It uses the same two pass-boundary writes as a frame.
    fn check_timestamp_span(serialized: bool) {
        if std::env::var("ACID_RAIN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        super::super::errors::init_diagnostics_logging();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env()
                .unwrap_or(wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12),
            flags: wgpu::InstanceFlags::debugging().with_env(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: std::env::var("ACID_RAIN_SOFTWARE").as_deref() == Ok("1"),
            ..Default::default()
        }))
        .expect("Timestamp fixture requested but no compatible adapter exists");
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("Timestamp fixture skipped: adapter does not advertise queries");
            return;
        }
        eprintln!(
            "Timestamp fixture adapter: {:?}; serialized={serialized}",
            adapter.get_info()
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .unwrap();
        let errors = Arc::new(Mutex::new(Diagnostics::default()));
        let callback_errors = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            callback_errors
                .lock()
                .unwrap()
                .record(ErrorSource::Rendering, error.to_string());
        }));
        let callback_errors = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            callback_errors.lock().unwrap().record(
                ErrorSource::DeviceLost,
                format!("Timestamp fixture device lost ({reason:?}): {message}"),
            );
        });
        let slot = TimingSlot::new(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tiny timestamp fixture: compute then render"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
                @group(0) @binding(0) var<storage, read_write> result: u32;
                @compute @workgroup_size(1) fn compute() { result = 42u; }
                @vertex fn vertex(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
                    let x = f32(i & 1u) * 4.0 - 1.0;
                    let y = f32((i >> 1u) & 1u) * 4.0 - 1.0;
                    return vec4f(x, y, 0.0, 1.0);
                }
            "#,
            )),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Timestamp fixture dependent fragment"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                @group(0) @binding(0) var<storage, read> result: u32;
                @fragment fn fragment() -> @location(0) vec4f {
                    return select(vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 1.0, 0.0, 1.0), result == 42u);
                }
            "#)),
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Timestamp fixture compute"),
            layout: None,
            module: &shader,
            entry_point: Some("compute"),
            compilation_options: Default::default(),
            cache: None,
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp fixture compute output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: result.as_entire_binding(),
            }],
        });
        let raster = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Timestamp fixture render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let fragment_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &raster.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: result.as_entire_binding(),
            }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Timestamp fixture output"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Timestamp fixture beginning"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: slot.queries(),
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
            });
            pass.set_pipeline(&compute);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if serialized {
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            dump_driver_messages(&device);
            super::super::errors::check(&device, &errors).unwrap();
            encoder = device.create_command_encoder(&Default::default());
        }
        {
            let mut pass = super::super::pipelines::render_pass_timed(
                &mut encoder,
                &view,
                "Timestamp fixture end",
                true,
                Some(slot.queries()),
            );
            pass.set_pipeline(&raster);
            pass.set_bind_group(0, &fragment_binding, &[]);
            pass.draw(0..3, 0..1);
        }
        if serialized {
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            dump_driver_messages(&device);
            super::super::errors::check(&device, &errors).unwrap();
            encoder = device.create_command_encoder(&Default::default());
        }
        slot.resolve(&mut encoder);
        queue.submit([encoder.finish()]);
        let state = Arc::new(Mutex::new(TimingState::default()));
        let busy = Arc::new(AtomicBool::new(true));
        let in_flight = Arc::new(AtomicUsize::new(1));
        slot.complete(
            &queue,
            queue.get_timestamp_period(),
            1,
            Instant::now(),
            1,
            state.clone(),
            busy.clone(),
            in_flight.clone(),
            errors.clone(),
        );
        let polled = device.poll(wgpu::PollType::wait_indefinitely());
        dump_driver_messages(&device);
        super::super::errors::check(&device, &errors).unwrap();
        polled.unwrap();
        assert!(errors.lock().unwrap().profiling_warning().is_none());
        assert!(!busy.load(Ordering::Acquire));
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
        let timing = state
            .lock()
            .unwrap()
            .last
            .expect("timestamp callback did not complete");
        // A one-pixel workload can have equal timestamps on a coarse clock.
        // Inspect the raw pair, without weakening production timing validity.
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slot.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let raw = slot.readback.slice(..).get_mapped_range();
        let begin = u64::from_ne_bytes(raw[0..8].try_into().unwrap());
        let end = u64::from_ne_bytes(raw[8..16].try_into().unwrap());
        assert!(
            begin > 0 && end >= begin,
            "invalid timestamp pair {begin}..{end}"
        );
        drop(raw);
        slot.readback.unmap();
        let pixels = super::super::read_texture_bytes(&device, &queue, &target, 4).unwrap();
        assert_eq!(
            &pixels[..4],
            &[0, 255, 0, 255],
            "query resolution must preserve rendering"
        );
        eprintln!(
            "Timestamp fixture PASS serialized={serialized}, raw={begin}..{end}, span={:?}ms",
            timing.gpu_ms
        );
    }

    #[test]
    fn timestamp_query_spans_compute_and_render() { check_timestamp_span(false); }

    #[test]
    fn timestamp_query_spans_serialized_compute_and_render() { check_timestamp_span(true); }

    #[test]
    fn cancelled_timing_map_keeps_slot_busy_until_queue_completion() {
        if std::env::var("ACID_RAIN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: std::env::var("ACID_RAIN_SOFTWARE").as_deref() == Ok("1"),
            ..Default::default()
        }))
        .expect("GPU fixture requested but adapter unavailable");
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .unwrap();
        let slot = TimingSlot::new(&device);
        let state = Arc::new(Mutex::new(TimingState::default()));
        let error = Arc::new(Mutex::new(Diagnostics::default()));
        let busy = Arc::new(AtomicBool::new(true));
        let in_flight = Arc::new(AtomicUsize::new(1));
        slot.complete(
            &queue,
            queue.get_timestamp_period(),
            7,
            Instant::now(),
            1,
            state.clone(),
            busy.clone(),
            in_flight.clone(),
            error.clone(),
        );
        slot.readback.unmap();
        assert!(
            busy.load(Ordering::Acquire),
            "failed optional map released a live frame slot"
        );
        assert_eq!(in_flight.load(Ordering::Acquire), 1);
        assert!(error.lock().unwrap().profiling_warning().is_some());
        assert!(error.lock().unwrap().fatal().is_none());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(!busy.load(Ordering::Acquire));
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
        let timing = state.lock().unwrap().last.unwrap();
        assert_eq!(timing.sequence, 7);
        assert_eq!(timing.gpu_ms, None);
    }
}
