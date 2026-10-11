//! Bounded simulation worker: one mutable owner, one newest immutable frame.

use std::{
    fmt,
    panic::{
        AssertUnwindSafe,
        catch_unwind,
    },
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            Ordering,
        },
        mpsc::{
            self,
            Receiver,
            SyncSender,
            TrySendError,
        },
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

use acid_rain_native_core::{
    OwnedSnapshot,
    Simulation,
    SimulationError,
};

const COMMAND_CAPACITY: usize = 64;
const COMMANDS_PER_UPDATE: usize = 8;
const UPDATE_PERIOD: Duration = Duration::from_nanos(1_000_000_000 / 60);
type Waker = Arc<dyn Fn() + Send + Sync>;

#[derive(Clone, Copy, Debug)]
pub enum Control {
    /// Shared native control IDs 1..=16. Toggle controls ignore their value.
    Set {
        code: u32,
        value: f32,
    },
    /// Numeric controls 2..=7 add a delta; quality/mode/scene 8..=10 cycle.
    Adjust {
        code: u32,
        delta: f32,
    },
    /// Toggle drain (1) or rapids (2) against the worker's current mode.
    ToggleMode(u32),
    Drop {
        world: [f64; 2],
        size: f32,
    },
}

#[derive(Clone, Debug)]
pub struct OwnedFrame {
    pub snapshot: Arc<OwnedSnapshot>,
    /// Wall-clock sample corresponding to snapshot.physical_time, taken before
    /// simulation work, never at the end of a variably expensive update.
    pub simulation_started: Instant,
    pub published_at: Instant,
    pub simulation_millis: f64,
    pub held: bool,
    pub publication: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandError {
    Full,
    Stopped,
    InvalidControl,
}

impl fmt::Display for CommandError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::Full => "simulation command queue is full",
            Self::Stopped => "simulation worker has stopped",
            Self::InvalidControl => "invalid simulation control or value",
        })
    }
}
impl std::error::Error for CommandError {}

struct Shared {
    latest: Mutex<Arc<OwnedFrame>>,
    held: AtomicBool,
    stop: AtomicBool,
    error: Mutex<Option<String>>,
    waker: Mutex<Option<Waker>>,
}

pub struct Worker {
    commands: SyncSender<Control>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn new(seed: u64) -> anyhow::Result<Self> { Self::from_simulation(Simulation::new(seed)?) }

    /// Configure the simulation synchronously before moving its sole owner here.
    pub fn from_simulation(mut simulation: Simulation) -> anyhow::Result<Self> {
        let started = Instant::now();
        let initial = Arc::new(OwnedFrame {
            snapshot: Arc::new(simulation.snapshot()),
            simulation_started: started,
            published_at: Instant::now(),
            simulation_millis: started.elapsed().as_secs_f64() * 1000.0,
            held: false,
            publication: 0,
        });
        let shared = Arc::new(Shared {
            latest: Mutex::new(initial),
            held: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
            waker: Mutex::new(None),
        });
        let (commands, receiver) = mpsc::sync_channel(COMMAND_CAPACITY);
        let worker_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("acid-rain-physics".into())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run_worker(simulation, receiver, &worker_shared, started)
                }));
                let failure = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error.to_string()),
                    Err(_) => Some("simulation worker panicked and has been stopped".to_owned()),
                };
                if let Some(error) = failure {
                    *worker_shared.error.lock().unwrap() = Some(error);
                    worker_shared.stop.store(true, Ordering::Release);
                    wake(&worker_shared);
                }
            })?;
        Ok(Self {
            commands,
            shared,
            thread: Some(thread),
        })
    }

    /// Loading a frame only clones one Arc. Old frames are retained solely by
    /// their readers, never by an unbounded presentation queue.
    pub fn latest(&self) -> Arc<OwnedFrame> { Arc::clone(&self.shared.latest.lock().unwrap()) }

    pub fn command(
        &self,
        control: Control,
    ) -> Result<(), CommandError> {
        if !control.is_valid() {
            return Err(CommandError::InvalidControl);
        }
        if self.shared.stop.load(Ordering::Acquire) {
            return Err(CommandError::Stopped);
        }
        self.commands
            .try_send(control)
            .map_err(|error| match error {
                TrySendError::Full(_) => CommandError::Full,
                TrySendError::Disconnected(_) => CommandError::Stopped,
            })?;
        self.thread.as_ref().unwrap().thread().unpark();
        Ok(())
    }

    /// Visibility/help hold is independent from the user's pause setting. Time
    /// spent held is discarded on resume instead of becoming a catch-up step.
    pub fn set_held(
        &self,
        held: bool,
    ) {
        if self.shared.held.swap(held, Ordering::AcqRel) != held {
            self.thread.as_ref().unwrap().thread().unpark();
        }
    }

    /// Called after publication, including controls while held. The callback
    /// should enqueue a host event and return promptly; it runs without locks.
    pub fn set_waker(
        &self,
        waker: Waker,
    ) {
        *self.shared.waker.lock().unwrap() = Some(Arc::clone(&waker));
        waker();
    }

    pub fn error(&self) -> Option<String> { self.shared.error.lock().unwrap().clone() }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

impl Control {
    fn is_valid(self) -> bool {
        match self {
            Self::Set { code, value } => {
                value.is_finite()
                    && match code {
                        1..=7 | 11..=16 => true,
                        8 | 9 => value.fract() == 0.0 && (0.0..=2.0).contains(&value),
                        10 => value.fract() == 0.0 && (0.0..=6.0).contains(&value),
                        _ => false,
                    }
            },
            Self::Adjust { code, delta } => {
                delta.is_finite()
                    && match code {
                        2..=7 => true,
                        8..=10 => delta.fract() == 0.0,
                        _ => false,
                    }
            },
            Self::Drop { world, size } => {
                world.iter().all(|v| v.is_finite())
                    && size.is_finite()
                    && (0.25..=3.0).contains(&size)
            },
            Self::ToggleMode(mode) => matches!(mode, 1 | 2),
        }
    }

    fn apply(
        self,
        simulation: &mut Simulation,
    ) -> Result<(), SimulationError> {
        match self {
            Self::Set { code, value } => simulation.command(code, value),
            Self::Drop { world, size } => simulation.spawn_drop(world, size).map(|_| ()),
            Self::ToggleMode(mode) => {
                let target = if simulation.state().mode == mode {
                    0
                } else {
                    mode
                };
                simulation.command(9, target as f32)
            },
            Self::Adjust { code, delta } => {
                let state = simulation.state();
                let current = match code {
                    2 => state.speed,
                    3 => state.rain,
                    4 => state.drop_size,
                    5 => state.bleach,
                    6 => state.trip,
                    7 => state.rgb_shift,
                    8 => state.quality as f32,
                    9 => state.mode as f32,
                    10 => state.scene as f32,
                    _ => return Err(SimulationError::InvalidCommand),
                };
                // Compute additions in f64 so any finite f32 delta stays finite
                // until bounded/cycled, including untrusted external controls.
                let adjusted = f64::from(current) + f64::from(delta);
                let value = match code {
                    8 | 9 => adjusted.rem_euclid(3.0),
                    10 => adjusted.rem_euclid(7.0),
                    _ => adjusted.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)),
                } as f32;
                simulation.command(code, value)
            },
        }
    }
}

fn wake(shared: &Shared) {
    let waker = shared.waker.lock().unwrap().clone();
    if let Some(waker) = waker {
        waker();
    }
}

fn run_worker(
    mut simulation: Simulation,
    commands: Receiver<Control>,
    shared: &Shared,
    initial_time: Instant,
) -> Result<(), SimulationError> {
    let mut last_start = initial_time;
    let mut deadline = initial_time + UPDATE_PERIOD;
    let mut was_held = false;
    let mut was_paused = simulation.state().paused;
    let mut publication = 0_u64;
    while !shared.stop.load(Ordering::Acquire) {
        let measurement_start = Instant::now();
        let held = shared.held.load(Ordering::Acquire);
        let mut changed = held != was_held;
        let mut processed = 0;
        for control in commands.try_iter().take(COMMANDS_PER_UPDATE) {
            match control.apply(&mut simulation) {
                Ok(()) | Err(SimulationError::InvalidCommand) => {},
                Err(error) => return Err(error),
            }
            changed = true;
            processed += 1;
        }
        let paused = simulation.state().paused;
        let started = Instant::now();
        if held || paused || was_held || was_paused {
            last_start = started;
            deadline = started + UPDATE_PERIOD;
        }
        if !held && !paused && started >= deadline {
            // Elapsed start-to-start time includes CPU work. Expensive updates
            // lower the publication rate without slowing simulated seconds.
            simulation.advance(started.saturating_duration_since(last_start))?;
            last_start = started;
            deadline = started + UPDATE_PERIOD;
            changed = true;
        }
        if changed {
            let snapshot = Arc::new(simulation.snapshot());
            publication = publication.wrapping_add(1);
            let frame = Arc::new(OwnedFrame {
                snapshot,
                simulation_started: last_start,
                published_at: Instant::now(),
                simulation_millis: measurement_start.elapsed().as_secs_f64() * 1000.0,
                held,
                publication,
            });
            *shared.latest.lock().unwrap() = frame;
            wake(shared);
        }
        was_held = held;
        was_paused = paused;
        if processed == COMMANDS_PER_UPDATE {
            continue;
        }
        let wait = if held || paused {
            Duration::from_millis(250)
        } else {
            deadline.saturating_duration_since(Instant::now())
        };
        thread::park_timeout(wait);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{
            Duration,
            Instant,
        },
    };

    use super::*;

    fn wait_for(
        worker: &Worker,
        predicate: impl Fn(&OwnedFrame) -> bool,
    ) -> Arc<OwnedFrame> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(worker.error().is_none(), "{:?}", worker.error());
            let frame = worker.latest();
            if predicate(&frame) {
                return frame;
            }
            assert!(
                Instant::now() < deadline,
                "worker did not publish expected state"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn held_worker_accepts_controls_wakes_host_and_keeps_owned_frames_valid() {
        use std::sync::atomic::{
            AtomicUsize,
            Ordering,
        };
        let mut simulation = Simulation::new(809).unwrap();
        simulation.command(1, 0.0).unwrap();
        let worker = Worker::from_simulation(simulation).unwrap();
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = wakes.clone();
        worker.set_waker(Arc::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        }));
        worker.set_held(true);
        let held = wait_for(&worker, |frame| frame.held);
        let previous_wakes = wakes.load(Ordering::Relaxed);
        worker
            .command(Control::Adjust {
                code: 2,
                delta: 0.2,
            })
            .unwrap();
        let changed = wait_for(&worker, |frame| {
            frame.snapshot.speed > held.snapshot.speed
                && wakes.load(Ordering::Relaxed) > previous_wakes
        });
        assert_eq!(changed.snapshot.physical_time, held.snapshot.physical_time);
        assert!(changed.publication > held.publication);
        assert!(wakes.load(Ordering::Relaxed) > previous_wakes);
        assert_eq!(held.snapshot.speed, 2.4);
        assert!(changed.published_at >= changed.simulation_started);
        drop(worker);
        assert!(held.snapshot.water.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn invalid_controls_are_rejected_before_enqueueing() {
        let worker = Worker::new(811).unwrap();
        assert_eq!(
            worker.command(Control::Set {
                code: 2,
                value: f32::NAN
            }),
            Err(CommandError::InvalidControl)
        );
        assert_eq!(
            worker.command(Control::Set {
                code: 800,
                value: 0.0
            }),
            Err(CommandError::InvalidControl)
        );
        assert_eq!(
            worker.command(Control::Adjust {
                code: 1,
                delta: 1.0
            }),
            Err(CommandError::InvalidControl)
        );
        assert_eq!(
            worker.command(Control::Drop {
                world: [f64::INFINITY, 0.0],
                size: 1.0
            }),
            Err(CommandError::InvalidControl)
        );
        assert!(worker.error().is_none());
    }

    #[test]
    fn resume_rebases_clock_instead_of_catching_up_hidden_time() {
        let mut simulation = Simulation::new(821).unwrap();
        simulation.command(1, 0.0).unwrap();
        let worker = Worker::from_simulation(simulation).unwrap();
        worker.set_held(true);
        let held = wait_for(&worker, |frame| frame.held);
        thread::sleep(Duration::from_millis(120));
        worker
            .command(Control::Set {
                code: 1,
                value: 0.0,
            })
            .unwrap();
        let unpaused = wait_for(&worker, |frame| !frame.snapshot.paused);
        assert_eq!(unpaused.snapshot.physical_time, held.snapshot.physical_time);
        // Capture the first resume publication, even if a loaded CI scheduler
        // lets several more updates run before the test thread is scheduled.
        let (sender, receiver) = mpsc::channel();
        let shared = Arc::downgrade(&worker.shared);
        worker.set_waker(Arc::new(move || {
            if let Some(shared) = shared.upgrade() {
                let frame = Arc::clone(&shared.latest.lock().unwrap());
                let _ = sender.send(frame);
            }
        }));
        let resume_time = Instant::now();
        worker.set_held(false);
        let resumed = loop {
            let frame = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
            if !frame.held {
                break frame;
            }
        };
        assert!(resumed.simulation_started >= resume_time);
        assert_eq!(resumed.snapshot.physical_time, held.snapshot.physical_time);
    }

    #[test]
    fn queued_mode_toggles_use_worker_state_instead_of_a_stale_display_frame() {
        let mut simulation = Simulation::new(823).unwrap();
        simulation.command(1, 0.0).unwrap();
        let before = simulation.state().cloud_revision;
        // Apply the same ordered commands that the worker drains in one batch;
        // no intervening presentation snapshot can supply the second mode.
        Control::ToggleMode(1).apply(&mut simulation).unwrap();
        Control::ToggleMode(1).apply(&mut simulation).unwrap();
        assert_eq!(simulation.state().mode, 0);
        assert_eq!(simulation.state().cloud_revision, before + 2);
        assert!(!Control::ToggleMode(3).is_valid());
    }
}
