//! Bounded, timestamped presentation of completed display frames.

use std::{
    collections::VecDeque,
    io::{
        self,
        Write,
    },
    sync::{
        Arc,
        Condvar,
        Mutex,
    },
    thread::{
        self,
        JoinHandle,
    },
    time::{
        Duration,
        Instant,
    },
};

use crate::{
    color_pipeline::{
        decode_byte,
        encode_srgb,
    },
    output::{
        Cell,
        TerminalOutput,
    },
};

struct Frame {
    timestamp: Instant,
    width: u16,
    height: u16,
    cells: Vec<Cell>,
    status: Option<String>,
    animate: bool,
}

struct Timeline {
    frames: VecDeque<Frame>,
    cursor: Option<Instant>,
    latency: Duration,
    enabled: bool,
}

impl Timeline {
    fn new() -> Self {
        Self {
            frames: VecDeque::new(),
            cursor: None,
            latency: Duration::from_millis(100),
            enabled: true,
        }
    }

    fn push(
        &mut self,
        frame: Frame,
    ) {
        if let Some(previous) = self.frames.back() {
            if (previous.width, previous.height) != (frame.width, frame.height)
                || !frame.animate
                || !previous.animate
                || !self.enabled
            {
                self.clear();
            } else {
                if frame.timestamp <= previous.timestamp {
                    return;
                }
                // A low-pass interval estimate avoids changing the viewing delay
                // abruptly when one producer frame takes longer than usual.
                let desired = frame
                    .timestamp
                    .duration_since(previous.timestamp)
                    .mul_f32(2.0);
                self.latency = self.latency.mul_f32(0.75) + desired.mul_f32(0.25);
                self.latency = self
                    .latency
                    .clamp(Duration::from_millis(33), Duration::from_millis(150));
            }
        }
        self.frames.push_back(frame);
        while self.frames.len() > 8 {
            self.frames.pop_front();
        }
    }

    /// Return whether another timed presentation can change the visible image.
    fn sample(
        &mut self,
        now: Instant,
        cells: &mut Vec<Cell>,
    ) -> bool {
        let Some(last) = self.frames.back() else {
            return false;
        };
        let last_time = last.timestamp;
        if !self.enabled || !last.animate {
            cells.clone_from(&last.cells);
            self.cursor = Some(last_time);
            return false;
        }
        let first_time = self.frames.front().unwrap().timestamp;
        let target = now
            .checked_sub(self.latency)
            .unwrap_or(first_time)
            .max(self.cursor.unwrap_or(first_time))
            .clamp(first_time, last_time);
        self.cursor = Some(target);
        while self.frames.len() > 2 && self.frames[1].timestamp <= target {
            self.frames.pop_front();
        }
        let left = self.frames.front().unwrap();
        if target >= last_time {
            cells.clone_from(&self.frames.back().unwrap().cells);
        } else if target <= left.timestamp || self.frames.len() == 1 {
            cells.clone_from(&left.cells);
        } else {
            let right = &self.frames[1];
            let fraction = target.duration_since(left.timestamp).as_secs_f32()
                / right.timestamp.duration_since(left.timestamp).as_secs_f32();
            cells.resize(left.cells.len(), ((0, 0, 0), (0, 0, 0)));
            for ((result, a), b) in cells.iter_mut().zip(&left.cells).zip(&right.cells) {
                *result = (blend_rgb(a.0, b.0, fraction), blend_rgb(a.1, b.1, fraction));
            }
        }
        target < last_time
    }

    fn set_enabled(
        &mut self,
        enabled: bool,
    ) {
        if self.enabled != enabled {
            self.enabled = enabled;
            if let Some(latest) = self.frames.pop_back() {
                self.clear();
                self.frames.push_back(latest);
            }
        }
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.cursor = None;
        self.latency = Duration::from_millis(100);
    }
}

fn blend_rgb(
    a: (u8, u8, u8),
    b: (u8, u8, u8),
    fraction: f32,
) -> (u8, u8, u8) {
    let channel = |a, b| {
        let a = decode_byte(a);
        let b = decode_byte(b);
        (encode_srgb(a + (b - a) * fraction) * 255.0).round() as u8
    };
    (channel(a.0, b.0), channel(a.1, b.1), channel(a.2, b.2))
}

enum Command {
    Frame(Frame),
    Raw(Vec<u8>),
}
struct Mailbox {
    pending: Option<Command>,
    enabled: bool,
    epoch: u64,
    dimensions: (u16, u16),
    stop: bool,
    error: Option<(io::ErrorKind, String)>,
}
struct Shared {
    state: Mutex<Mailbox>,
    wake: Condvar,
}
pub struct FramePresenter {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl FramePresenter {
    pub fn new(
        width: u16,
        height: u16,
        fps: u16,
    ) -> io::Result<Self> {
        Self::with_writer(width, height, fps, |bytes| {
            // Never retain this lock while waiting or doing interpolation. In
            // particular, the main-thread panic hook must be able to restore
            // the terminal before stack unwinding joins this worker.
            let stdout = io::stdout();
            let mut out = stdout.lock();
            out.write_all(bytes)?;
            out.flush()
        })
    }

    fn with_writer<F>(
        width: u16,
        height: u16,
        fps: u16,
        write: F,
    ) -> io::Result<Self>
    where
        F: FnMut(&[u8]) -> io::Result<()> + Send + 'static,
    {
        if fps == 0 || usize::from(width) * usize::from(height) > 500_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid presentation size or frame rate",
            ));
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(Mailbox {
                pending: None,
                enabled: true,
                epoch: 0,
                dimensions: (width, height),
                stop: false,
                error: None,
            }),
            wake: Condvar::new(),
        });
        let worker_shared = shared.clone();
        let worker = thread::Builder::new()
            .name("frame-presenter".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    present_loop(
                        &worker_shared,
                        width,
                        height,
                        Duration::from_secs_f64(1.0 / f64::from(fps)),
                        write,
                    )
                }))
                .unwrap_or_else(|_| Err(io::Error::other("frame presenter panicked")));
                if let Err(error) = result {
                    worker_shared
                        .state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .error = Some((error.kind(), error.to_string()));
                    worker_shared.wake.notify_all();
                }
            })?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub fn submit(
        &self,
        timestamp: Instant,
        width: u16,
        height: u16,
        cells: Vec<Cell>,
        status: Option<String>,
        animate: bool,
    ) -> io::Result<()> {
        let count = usize::from(width) * usize::from(height);
        if count > 500_000 || cells.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "presentation frame dimensions do not match cells",
            ));
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        mailbox_error(&state)?;
        if state.dimensions != (width, height) || !animate {
            state.epoch = state.epoch.wrapping_add(1);
        }
        state.dimensions = (width, height);
        state.pending = Some(Command::Frame(Frame {
            timestamp,
            width,
            height,
            cells,
            status,
            animate,
        }));
        self.shared.wake.notify_one();
        Ok(())
    }

    /// Raw screens, including an empty resize barrier, stop all interpolation.
    pub fn show_raw(
        &self,
        bytes: Vec<u8>,
    ) -> io::Result<()> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        mailbox_error(&state)?;
        state.epoch = state.epoch.wrapping_add(1);
        state.pending = Some(Command::Raw(bytes));
        self.shared.wake.notify_one();
        Ok(())
    }

    /// Change the mode and discard old animation. The caller submits a fresh
    /// frame containing the new status; changing the mode alone never repaints.
    pub fn set_enabled(
        &self,
        enabled: bool,
    ) -> io::Result<()> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        mailbox_error(&state)?;
        if state.enabled != enabled {
            state.enabled = enabled;
            state.epoch = state.epoch.wrapping_add(1);
            // A frame already queued before this control change has the old
            // HUD. Raw help/resize barriers must still be consumed normally.
            if matches!(state.pending, Some(Command::Frame(_))) {
                state.pending = None;
            }
            self.shared.wake.notify_one();
        }
        Ok(())
    }

    pub fn check_error(&self) -> io::Result<()> {
        mailbox_error(&self.shared.state.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for FramePresenter {
    fn drop(&mut self) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stop = true;
        self.shared.wake.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn mailbox_error(state: &Mailbox) -> io::Result<()> {
    if let Some((kind, message)) = &state.error {
        return Err(io::Error::new(*kind, message.clone()));
    }
    if state.stop {
        return Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "frame presenter stopped",
        ));
    }
    Ok(())
}

fn present_loop<F>(
    shared: &Shared,
    width: u16,
    height: u16,
    interval: Duration,
    mut write: F,
) -> io::Result<()>
where
    F: FnMut(&[u8]) -> io::Result<()>,
{
    let mut output = TerminalOutput::new(width, height);
    let mut dimensions = (width, height);
    let mut timeline = Timeline::new();
    let mut cells = Vec::new();
    let mut bytes = Vec::new();
    let mut epoch = 0;
    let mut active = false;
    let mut dirty = false;
    let mut deadline = Instant::now();
    loop {
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.stop && state.pending.is_none() && state.epoch == epoch {
            if dirty || active {
                let wait = deadline.saturating_duration_since(Instant::now());
                if wait.is_zero() {
                    break;
                }
                state = shared
                    .wake
                    .wait_timeout(state, wait)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            } else {
                state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
        if state.stop {
            return Ok(());
        }
        let pending = state.pending.take();
        let reset = state.epoch != epoch;
        epoch = state.epoch;
        let enabled = state.enabled;
        drop(state);
        if reset {
            timeline.clear();
            output.invalidate();
            deadline = Instant::now();
            active = false;
            dirty = false;
        }
        timeline.set_enabled(enabled);
        match pending {
            Some(Command::Raw(raw)) => {
                timeline.clear();
                output.resize(dimensions.0, dimensions.1);
                active = false;
                dirty = false;
                if !raw.is_empty() {
                    write(&raw)?;
                }
                continue;
            },
            Some(Command::Frame(frame)) => {
                if dimensions != (frame.width, frame.height) {
                    dimensions = (frame.width, frame.height);
                    output.resize(frame.width, frame.height);
                }
                output.set_status(frame.status.as_deref());
                if !frame.animate || !enabled || timeline.frames.is_empty() {
                    deadline = Instant::now();
                } else if !active && !dirty && Instant::now() >= deadline {
                    // A producer stall leaves the latest image held without
                    // wake-ups. Re-enter the existing display clock phase,
                    // rather than presenting a late arrival off cadence.
                    deadline = next_presentation_deadline(deadline, Instant::now(), interval);
                }
                timeline.push(frame);
                dirty = true;
            },
            None => {},
        }
        let now = Instant::now();
        if (dirty || active) && now >= deadline {
            active = timeline.sample(now, &mut cells);
            // The empty timeline after a raw-screen barrier has no image to
            // repaint, even if a toggle woke the worker before the next frame.
            if !timeline.frames.is_empty() {
                output.encode(&cells, &mut bytes);
                if !bytes.is_empty() {
                    write(&bytes)?;
                }
            }
            dirty = false;
            // Slow sinks do not cause an unbounded catch-up burst.
            deadline = next_presentation_deadline(deadline, Instant::now(), interval);
        }
    }
}

fn next_presentation_deadline(
    deadline: Instant,
    now: Instant,
    interval: Duration,
) -> Instant {
    if now < deadline {
        return deadline;
    }
    let remainder = now.duration_since(deadline).as_nanos() % interval.as_nanos();
    now + Duration::from_nanos((interval.as_nanos() - remainder) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    const BLACK: Cell = ((0, 0, 0), (0, 0, 0));
    const WHITE: Cell = ((255, 255, 255), (255, 255, 255));
    fn frame(
        t: Instant,
        value: Cell,
    ) -> Frame {
        Frame {
            timestamp: t,
            width: 1,
            height: 1,
            cells: vec![value],
            status: None,
            animate: true,
        }
    }
    #[test]
    fn timestamps_interpolate_in_display_linear_light_without_feedback() {
        let t = Instant::now();
        let mut timeline = Timeline::new();
        timeline.push(frame(t, BLACK));
        timeline.push(frame(t + Duration::from_millis(50), WHITE));
        let mut cells = Vec::new();
        assert!(timeline.sample(t + timeline.latency + Duration::from_millis(25), &mut cells));
        assert_eq!(cells, [((188, 188, 188), (188, 188, 188))]);
        timeline.sample(t + timeline.latency + Duration::from_millis(40), &mut cells);
        assert_eq!(cells, [((231, 231, 231), (231, 231, 231))]);
    }
    #[test]
    fn sampling_never_extrapolates_or_reverses_time() {
        let t = Instant::now();
        let mut timeline = Timeline::new();
        timeline.push(frame(t, BLACK));
        timeline.push(frame(t + Duration::from_millis(50), WHITE));
        let mut cells = Vec::new();
        timeline.sample(t + timeline.latency + Duration::from_millis(40), &mut cells);
        let previous = cells.clone();
        timeline.sample(t, &mut cells);
        assert_eq!(cells, previous);
        assert!(!timeline.sample(t + Duration::from_secs(10), &mut cells));
        assert_eq!(cells, [WHITE]);
        assert_eq!(timeline.cursor, Some(t + Duration::from_millis(50)));
    }
    #[test]
    fn resize_pause_and_disabled_output_have_no_cross_frame_blend() {
        let t = Instant::now();
        let mut timeline = Timeline::new();
        timeline.push(frame(t, BLACK));
        let mut changed = frame(t + Duration::from_millis(50), WHITE);
        changed.width = 2;
        changed.cells.push(WHITE);
        timeline.push(changed);
        assert_eq!(timeline.frames.len(), 1);
        let mut paused = frame(t + Duration::from_millis(100), BLACK);
        paused.animate = false;
        timeline.push(paused);
        let mut cells = Vec::new();
        assert!(!timeline.sample(t, &mut cells));
        assert_eq!(cells, [BLACK]);
        timeline.set_enabled(false);
        timeline.push(frame(t + Duration::from_millis(150), WHITE));
        assert!(!timeline.sample(t, &mut cells));
        assert_eq!(cells, [WHITE]);
    }
    #[test]
    fn frame_history_and_latency_are_bounded_and_old_samples_are_rejected() {
        let t = Instant::now();
        let mut timeline = Timeline::new();
        for i in 0..100 {
            timeline.push(frame(t + Duration::from_millis(i * 100), WHITE));
            assert!(timeline.frames.len() <= 8);
            assert!(
                (Duration::from_millis(33)..=Duration::from_millis(150))
                    .contains(&timeline.latency)
            );
        }
        let last = timeline.frames.back().unwrap().timestamp;
        timeline.push(frame(t, BLACK));
        assert_eq!(timeline.frames.back().unwrap().timestamp, last);
        timeline.clear();
        assert!(timeline.frames.is_empty());
        assert!(timeline.cursor.is_none());
    }
    #[test]
    fn paused_and_raw_output_become_idle_and_shutdown_joins_writer() {
        use std::sync::{
            atomic::{
                AtomicBool,
                Ordering,
            },
            mpsc,
        };
        struct Finished(Arc<AtomicBool>);
        impl Drop for Finished {
            fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
        }
        let ended = Arc::new(AtomicBool::new(false));
        let finished = Finished(ended.clone());
        let (tx, rx) = mpsc::channel();
        let presenter = FramePresenter::with_writer(1, 1, 60, move |bytes| {
            let _ = &finished;
            tx.send(bytes.to_vec()).unwrap();
            Ok(())
        })
        .unwrap();
        presenter
            .submit(Instant::now(), 1, 1, vec![WHITE], None, false)
            .unwrap();
        let bytes = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(bytes.windows(7).any(|w| w == b"\x1b[?2026"));
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(60)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        presenter.show_raw(b"help screen".to_vec()).unwrap();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            b"help screen"
        );
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(60)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(presenter);
        assert!(ended.load(Ordering::SeqCst));
    }

    #[test]
    fn paused_toggle_waits_for_fresh_status_frame_and_draws_it_once() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel();
        let presenter = FramePresenter::with_writer(48, 1, 60, move |bytes| {
            tx.send(bytes.to_vec()).unwrap();
            Ok(())
        })
        .unwrap();
        presenter
            .submit(Instant::now(), 48, 1, vec![WHITE; 48], None, false)
            .unwrap();
        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        for enabled in [false, true, false, true] {
            presenter.set_enabled(enabled).unwrap();
            // Deliberately leave time for the worker to consume the toggle
            // before the producer has completed its new paused/HUD frame.
            assert!(
                matches!(
                    rx.recv_timeout(Duration::from_millis(60)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ),
                "toggle repainted a stale paused frame before fresh submission"
            );
            let status = format!(
                "Frame blending: {} | i: toggle",
                if enabled { "on" } else { "off" }
            );
            presenter
                .submit(
                    Instant::now(),
                    48,
                    1,
                    vec![WHITE; 48],
                    Some(status.clone()),
                    false,
                )
                .unwrap();
            let bytes = rx.recv_timeout(Duration::from_secs(1)).unwrap();
            let text = String::from_utf8(bytes).unwrap();
            assert_eq!(text.matches("\x1b[?2026h").count(), 1);
            assert!(text.contains(&status));
            assert!(matches!(
                rx.recv_timeout(Duration::from_millis(60)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
        }
    }
    #[test]
    fn writer_failures_are_reported_and_bad_frames_are_rejected() {
        let presenter = FramePresenter::with_writer(1, 1, 60, |_| {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test sink closed",
            ))
        })
        .unwrap();
        assert!(
            presenter
                .submit(Instant::now(), 2, 1, vec![BLACK], None, true)
                .is_err()
        );
        presenter.show_raw(vec![1]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while presenter.check_error().is_ok() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            presenter.check_error().unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert!(presenter.show_raw(vec![2]).is_err());
        assert!(FramePresenter::new(1, 1, 0).is_err());
    }
    #[test]
    fn slow_writer_keeps_only_the_latest_pending_frame() {
        use std::sync::mpsc;
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut first = true;
        let presenter = FramePresenter::with_writer(1, 1, 60, move |_| {
            if first {
                first = false;
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            }
            Ok(())
        })
        .unwrap();
        presenter.show_raw(vec![1]).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let now = Instant::now();
        for i in 0..100 {
            presenter
                .submit(
                    now + Duration::from_millis(i),
                    1,
                    1,
                    vec![WHITE],
                    None,
                    true,
                )
                .unwrap();
        }
        let state = presenter.shared.state.lock().unwrap();
        let Some(Command::Frame(frame)) = &state.pending else {
            panic!("missing latest frame");
        };
        assert_eq!(frame.timestamp, now + Duration::from_millis(99));
        drop(state);
        release_tx.send(()).unwrap();
        drop(presenter);
    }
    #[test]
    fn worker_emits_intermediate_cells_then_raw_help_cancels_pending_motion() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel();
        let presenter = FramePresenter::with_writer(1, 1, 60, move |bytes| {
            tx.send(bytes.to_vec()).unwrap();
            Ok(())
        })
        .unwrap();
        let now = Instant::now();
        presenter
            .submit(now, 1, 1, vec![BLACK], None, true)
            .unwrap();
        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        presenter
            .submit(
                now + Duration::from_millis(500),
                1,
                1,
                vec![WHITE],
                None,
                true,
            )
            .unwrap();
        let intermediate = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let text = String::from_utf8(intermediate).unwrap();
        assert!(text.contains("38;2;"));
        assert!(
            !text.contains("38;2;255;255;255"),
            "first update skipped interpolation: {text}"
        );
        assert!(!text.contains("38;2;0;0;0"));
        presenter.show_raw(b"help".to_vec()).unwrap();
        loop {
            if rx.recv_timeout(Duration::from_secs(1)).unwrap() == b"help" {
                break;
            }
        }
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(180)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        presenter
            .submit(Instant::now(), 1, 1, vec![WHITE], None, false)
            .unwrap();
        let resumed = String::from_utf8(rx.recv_timeout(Duration::from_secs(1)).unwrap()).unwrap();
        assert!(resumed.contains("\x1b[2J"));
        assert!(resumed.contains("38;2;255;255;255"));
    }
    #[test]
    fn presentation_deadlines_keep_cadence_and_skip_stalls() {
        let now = Instant::now();
        let interval = Duration::from_millis(16);
        assert_eq!(
            next_presentation_deadline(now, now + Duration::from_millis(3), interval),
            now + interval
        );
        assert_eq!(
            next_presentation_deadline(now, now + Duration::from_millis(37), interval),
            now + interval * 3
        );
    }
}
