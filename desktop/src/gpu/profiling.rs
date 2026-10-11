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
