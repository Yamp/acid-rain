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
        period: f32,
        sequence: u64,
        start: Instant,
        depth: u32,
        state: Arc<Mutex<TimingState>>,
        busy: Arc<AtomicBool>,
        in_flight: Arc<AtomicUsize>,
        error: Arc<Mutex<Option<String>>>,
    ) {
        let buffer = self.readback.clone();
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
                        *error.lock().unwrap() =
                            Some(format!("GPU timestamp readback failed: {failure}"));
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
