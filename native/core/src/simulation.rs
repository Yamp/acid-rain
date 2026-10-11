//! Owned, platform-independent access to the same simulation used by the C host.

use std::{
    fmt,
    ops::Deref,
    panic::{
        AssertUnwindSafe,
        catch_unwind,
    },
    sync::Arc,
    time::Duration,
};

use crate::{
    ARDrop,
    ARUniforms,
    Core,
};

/// Pointer-free metadata. Textures and the drop journal live in `OwnedSnapshot`.
#[derive(Clone, Copy, Debug)]
pub struct SnapshotState {
    pub water_width: u32,
    pub water_height: u32,
    pub bed_width: u32,
    pub bed_height: u32,
    pub sky_width: u32,
    pub sky_height: u32,
    pub light_width: u32,
    pub light_height: u32,
    pub uniforms: ARUniforms,
    pub speed: f32,
    pub rain: f32,
    pub drop_size: f32,
    pub bleach: f32,
    pub trip: f32,
    pub rgb_shift: f32,
    pub mode: u32,
    pub scene: u32,
    pub quality: u32,
    pub travel: bool,
    pub current: bool,
    pub paused: bool,
    pub effects: bool,
    pub frame: u64,
    pub bed_revision: u64,
    pub cloud_revision: u64,
    pub world_origin: [f64; 2],
    pub physical_time: f64,
}

/// An immutable frame that remains valid after stepping or destroying its owner.
/// All pixels are row-major, four linear f32 components per texel. No borrowed
/// C pointers cross the thread boundary; clones only increment Arc counts.
#[derive(Clone, Debug)]
pub struct OwnedSnapshot {
    pub state: SnapshotState,
    pub water: Arc<[[f32; 4]]>,
    pub dye: Arc<[[f32; 4]]>,
    pub flow: Arc<[[f32; 4]]>,
    pub bed: Arc<[[f32; 4]]>,
    pub sky: Arc<[[f32; 4]]>,
    pub bed_light: Arc<[[f32; 4]]>,
    pub drops: Arc<[ARDrop]>,
}

impl Deref for OwnedSnapshot {
    type Target = SnapshotState;

    fn deref(&self) -> &SnapshotState { &self.state }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationError {
    InvalidCommand,
    Failed,
    Panicked,
}

impl fmt::Display for SimulationError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidCommand => "invalid simulation command or value",
            Self::Failed => "simulation is unavailable after an earlier panic",
            Self::Panicked => "simulation panicked and has been stopped",
        })
    }
}

impl std::error::Error for SimulationError {}

pub struct Simulation {
    core: Core,
    cached_bed: Option<(u64, Arc<[[f32; 4]]>)>,
}

impl Simulation {
    pub fn new(seed: u64) -> Result<Self, SimulationError> {
        catch_unwind(|| Self {
            core: Core::new(seed),
            cached_bed: None,
        })
        .map_err(|_| SimulationError::Panicked)
    }

    /// Advance by elapsed wall time. Like the native C host, a single update is
    /// capped at 100 ms; deterministic offline runs should use fixed small steps.
    pub fn advance(
        &mut self,
        wall_delta: Duration,
    ) -> Result<(), SimulationError> {
        self.with_core(|core| core.step(wall_delta.as_secs_f64()))
    }

    /// Shared C control IDs: pause=1, speed=2, rain=3, size=4, bleach=5,
    /// trip=6, RGB shift=7, quality=8, water mode=9, scene=10, travel=11,
    /// current=12, effects=13, rain burst=14, center drop=15, clear dye=16.
    /// Invalid IDs/non-finite values are rejected without poisoning the owner.
    pub fn command(
        &mut self,
        code: u32,
        value: f32,
    ) -> Result<(), SimulationError> {
        if self.with_core(|core| core.command(code, value))? {
            Ok(())
        } else {
            Err(SimulationError::InvalidCommand)
        }
    }

    /// Return false for invalid/out-of-window positions or a full drop journal.
    pub fn spawn_drop(
        &mut self,
        world: [f64; 2],
        size: f32,
    ) -> Result<bool, SimulationError> {
        self.with_core(|core| {
            if !world.iter().all(|value| value.is_finite()) || !core.spawn_drop(world, size, 0.2) {
                return false;
            }
            core.wall_delta = 0.0;
            core.refresh();
            true
        })
    }

    /// Copy only metadata; unlike snapshot(), this allocates no pixel buffers.
    pub fn state(&self) -> SnapshotState {
        let c = self.core.snapshot();
        SnapshotState {
            water_width: c.water_width,
            water_height: c.water_height,
            bed_width: c.bed_width,
            bed_height: c.bed_height,
            sky_width: c.sky_width,
            sky_height: c.sky_height,
            light_width: c.light_width,
            light_height: c.light_height,
            uniforms: c.uniforms,
            speed: c.speed,
            rain: c.rain,
            drop_size: c.drop_size,
            bleach: c.bleach,
            trip: self.core.trip,
            rgb_shift: self.core.rgb_shift,
            mode: c.mode,
            scene: c.scene,
            quality: c.quality,
            travel: c.travel != 0,
            current: c.current != 0,
            paused: c.paused != 0,
            effects: c.effects != 0,
            frame: c.frame,
            bed_revision: c.bed_revision,
            cloud_revision: c.cloud_revision,
            world_origin: [c.world_origin_x, c.world_origin_y],
            physical_time: c.physical_time,
        }
    }

    /// Freeze a frame for another thread or an offline fixture. The bed's large
    /// immutable texture is copied only when its world-space revision changes.
    pub fn snapshot(&mut self) -> OwnedSnapshot {
        let revision = self.core.bed_revision;
        if self
            .cached_bed
            .as_ref()
            .is_none_or(|(cached, _)| *cached != revision)
        {
            self.cached_bed = Some((revision, Arc::from(self.core.bed_pixels.as_slice())));
        }
        OwnedSnapshot {
            state: self.state(),
            water: Arc::from(self.core.water_pixels.as_slice()),
            dye: Arc::from(self.core.dye_pixels.as_slice()),
            flow: Arc::from(self.core.flow_pixels.as_slice()),
            bed: Arc::clone(&self.cached_bed.as_ref().expect("bed cache initialized").1),
            sky: Arc::from(self.core.sky_pixels.as_slice()),
            bed_light: Arc::from(self.core.light_pixels.as_slice()),
            drops: Arc::from(self.core.drop_pixels.as_slice()),
        }
    }

    fn with_core<T>(
        &mut self,
        operation: impl FnOnce(&mut Core) -> T,
    ) -> Result<T, SimulationError> {
        if self.core.failed {
            return Err(SimulationError::Failed);
        }
        match catch_unwind(AssertUnwindSafe(|| operation(&mut self.core))) {
            Ok(value) => Ok(value),
            Err(_) => {
                self.core.failed = true;
                Err(SimulationError::Panicked)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_frame_survives_steps_commands_and_owner_destruction() {
        fn assert_send<T: Send>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send::<Simulation>();
        assert_send_sync::<OwnedSnapshot>();
        let mut simulation = Simulation::new(731).unwrap();
        let first = simulation.snapshot();
        let original_water = first.water.to_vec();
        let original_bed = first.bed.to_vec();
        simulation.command(14, 0.0).unwrap();
        simulation.advance(Duration::from_millis(20)).unwrap();
        let second = simulation.snapshot();
        assert!(second.frame > first.frame);
        assert!(second.physical_time > first.physical_time);
        assert!(!second.drops.is_empty());
        assert!(first.drops.is_empty());
        drop(simulation);
        assert_eq!(&*first.water, original_water);
        assert_eq!(&*first.bed, original_bed);
        assert!(first.water.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(
            first.water.len(),
            (first.water_width * first.water_height) as usize
        );
        assert_eq!(first.dye.len(), first.water.len());
        assert_eq!(first.flow.len(), first.water.len());
        assert_eq!(
            first.bed.len(),
            (first.bed_width * first.bed_height) as usize
        );
        assert_eq!(
            first.sky.len(),
            (first.sky_width * first.sky_height) as usize
        );
        assert_eq!(
            first.bed_light.len(),
            (first.light_width * first.light_height) as usize
        );
    }

    #[test]
    fn cached_bed_is_reused_until_the_world_rebases() {
        let mut simulation = Simulation::new(733).unwrap();
        let first = simulation.snapshot();
        simulation.command(2, 1.2).unwrap();
        let second = simulation.snapshot();
        assert!(Arc::ptr_eq(&first.bed, &second.bed));
        simulation.core.water.set_world_center([9.0, -7.0]);
        simulation.core.refresh();
        let rebased = simulation.snapshot();
        assert_ne!(rebased.bed_revision, first.bed_revision);
        assert_ne!(rebased.world_origin, first.world_origin);
        assert!(!Arc::ptr_eq(&first.bed, &rebased.bed));
        assert_eq!(first.bed.len(), rebased.bed.len());
    }

    #[test]
    fn invalid_commands_do_not_poison_valid_commands_and_pause_freezes_time() {
        let mut simulation = Simulation::new(737).unwrap();
        assert_eq!(
            simulation.command(2, f32::NAN),
            Err(SimulationError::InvalidCommand)
        );
        assert_eq!(
            simulation.command(500, 1.0),
            Err(SimulationError::InvalidCommand)
        );
        simulation.command(2, 99.0).unwrap();
        assert_eq!(simulation.state().speed, 3.0);
        simulation.command(1, 0.0).unwrap();
        let paused = simulation.snapshot();
        simulation.advance(Duration::from_secs(100)).unwrap();
        let held = simulation.snapshot();
        assert!(held.paused);
        assert_eq!(held.physical_time, paused.physical_time);
        assert_eq!(held.frame, paused.frame);
        assert_eq!(held.world_origin, paused.world_origin);
        assert_eq!(&*held.water, &*paused.water);
        simulation.command(1, 0.0).unwrap();
        simulation.advance(Duration::from_millis(20)).unwrap();
        assert!(simulation.state().physical_time > held.physical_time);
    }

    #[test]
    fn drops_and_cloud_revision_follow_the_shared_core_contract() {
        let mut simulation = Simulation::new(739).unwrap();
        let state = simulation.state();
        let world = [state.world_origin[0] + 1.0, state.world_origin[1] + 1.0];
        assert!(!simulation.spawn_drop([f64::NAN, 0.0], 1.0).unwrap());
        assert!(!simulation.spawn_drop(world, f32::INFINITY).unwrap());
        assert!(simulation.spawn_drop(world, 1.0).unwrap());
        let first = simulation.snapshot();
        assert_eq!(first.drops.len(), 1);
        simulation.command(9, 0.0).unwrap();
        assert_eq!(simulation.state().cloud_revision, first.cloud_revision);
        simulation.command(16, 0.0).unwrap();
        let cleared = simulation.snapshot();
        assert!(cleared.drops.is_empty());
        assert_ne!(cleared.cloud_revision, first.cloud_revision);
        assert_eq!(first.drops.len(), 1);
    }

    #[test]
    fn safe_metadata_matches_the_c_abi_without_retaining_its_pointers() {
        let mut simulation = Simulation::new(743).unwrap();
        let c = simulation.core.snapshot();
        let rust = simulation.snapshot();
        assert_eq!(rust.water_width, c.water_width);
        assert_eq!(rust.water_height, c.water_height);
        assert_eq!(rust.uniforms.time, c.uniforms.time);
        assert_eq!(rust.world_origin, [c.world_origin_x, c.world_origin_y]);
        assert_eq!(rust.physical_time, c.physical_time);
        assert_eq!(rust.cloud_revision, c.cloud_revision);
        assert_eq!(std::mem::size_of::<ARUniforms>(), 208);
        assert_eq!(std::mem::size_of::<crate::ARSnapshot>(), 416);
        assert_eq!(std::mem::size_of::<ARDrop>(), 64);
    }

    #[test]
    fn a_panicking_operation_stops_mutation_but_preserves_already_owned_frames() {
        let mut simulation = Simulation::new(751).unwrap();
        let frozen = simulation.snapshot();
        let result: Result<(), _> = simulation.with_core(|_| panic!("injected core failure"));
        assert_eq!(result, Err(SimulationError::Panicked));
        assert_eq!(
            simulation.advance(Duration::from_millis(16)),
            Err(SimulationError::Failed)
        );
        assert_eq!(simulation.command(2, 1.0), Err(SimulationError::Failed));
        assert_eq!(
            simulation.spawn_drop([1.0, 1.0], 1.0),
            Err(SimulationError::Failed)
        );
        drop(simulation);
        for pixels in [
            &frozen.water,
            &frozen.dye,
            &frozen.flow,
            &frozen.bed,
            &frozen.sky,
            &frozen.bed_light,
        ] {
            assert!(pixels.iter().flatten().all(|value| value.is_finite()));
        }
    }
}
