//! C bridge for the native Metal host. The actual physical models are shared
//! verbatim with the terminal program, not a second approximate simulation.
#![allow(dead_code)]

mod drops;
pub use drops::ARDrop;
mod simulation;
pub use simulation::{
    OwnedSnapshot,
    Simulation,
    SimulationError,
    SnapshotState,
};

#[path = "../../../src/bed.rs"]
mod bed;
#[path = "../../../src/color_pipeline.rs"]
mod color_pipeline;
#[path = "../../../src/colors.rs"]
mod colors;
#[path = "../../../src/drain.rs"]
mod drain;
#[path = "../../../src/dye.rs"]
mod dye;
#[path = "../../../src/motion.rs"]
mod motion;
#[path = "../../../src/optics.rs"]
mod optics;
#[path = "../../../src/physics.rs"]
mod physics;
#[path = "../../../src/rapids.rs"]
mod rapids;
#[path = "../../../src/ripples.rs"]
mod ripples;
#[path = "../../../src/scene.rs"]
mod scene;
#[path = "../../../src/settings.rs"]
mod settings;
#[path = "../../../src/stirring.rs"]
mod stirring;
#[path = "../../../src/vortex.rs"]
mod vortex;
#[path = "../../../src/water.rs"]
mod water;
#[path = "../../../src/waves.rs"]
mod waves;
#[path = "../../../src/world.rs"]
mod world;
#[path = "../../../src/world_flow.rs"]
mod world_flow;

use std::{
    panic::{
        AssertUnwindSafe,
        catch_unwind,
    },
    time::Duration,
};

use rand::{
    Rng,
    SeedableRng,
    rngs::StdRng,
};
use scene::{
    SceneFrame,
    SceneKind,
};

const WATER_WIDTH: usize = 192;
const WATER_HEIGHT: usize = 128;
const BED_SIZE: usize = 256;
const SKY_WIDTH: usize = 128;
const SKY_HEIGHT: usize = 64;
const LIGHT_SIZE: usize = 64;
const BED_MIN: f32 = -1.5;
const BED_SPAN: f32 = 5.0;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Float4(pub [f32; 4]);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ARUniforms {
    pub viewport: Float4,
    pub time: Float4,
    pub camera: Float4,
    pub optics: Float4,
    pub gain: Float4,
    pub light: Float4,
    pub world: Float4,
    pub post: Float4,
    pub eye: Float4,
    pub forward: Float4,
    pub right: Float4,
    pub up: Float4,
    pub illumination: Float4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ARSnapshot {
    pub water_width: u32,
    pub water_height: u32,
    pub bed_width: u32,
    pub bed_height: u32,
    pub sky_width: u32,
    pub sky_height: u32,
    pub light_width: u32,
    pub light_height: u32,
    pub water: *const f32,
    pub dye: *const f32,
    pub bed: *const f32,
    pub sky: *const f32,
    pub bed_light: *const f32,
    pub uniforms: ARUniforms,
    pub speed: f32,
    pub rain: f32,
    pub drop_size: f32,
    pub bleach: f32,
    pub mode: u32,
    pub scene: u32,
    pub travel: u32,
    pub current: u32,
    pub paused: u32,
    pub quality: u32,
    pub effects: u32,
    pub frame: u64,
    pub bed_revision: u64,
    pub world_origin_x: f64,
    pub world_origin_y: f64,
    pub physical_time: f64,
    pub drops: *const ARDrop,
    pub drop_count: u32,
    pub reserved: u32,
    pub cloud_revision: u64,
    pub flow: *const f32,
}

pub struct Core {
    water: water::Water,
    bed: bed::BedWorld,
    travel: world::WorldTravel,
    rng: StdRng,
    seed: u64,
    elapsed: f32,
    wall_delta: f32,
    rain_accumulator: f32,
    speed: f32,
    rain: f32,
    drop_size: f32,
    bleach: f32,
    trip: f32,
    rgb_shift: f32,
    mode: u32,
    scene: SceneKind,
    quality: u32,
    current: bool,
    paused: bool,
    effects: bool,
    failed: bool,
    frame: u64,
    bed_revision: u64,
    bed_origin: Option<[f64; 2]>,
    uniforms: ARUniforms,
    water_pixels: Vec<[f32; 4]>,
    dye_pixels: Vec<[f32; 4]>,
    bed_pixels: Vec<[f32; 4]>,
    sky_pixels: Vec<[f32; 4]>,
    light_pixels: Vec<[f32; 4]>,
    sky_directions: Vec<[f32; 3]>,
    drops: Vec<drops::Drop>,
    drop_pixels: Vec<ARDrop>,
    next_drop_id: u64,
    cloud_revision: u64,
    flow_pixels: Vec<[f32; 4]>,
}

impl Core {
    fn new(seed: u64) -> Self {
        let mut water = water::Water::new(WATER_WIDTH as u16, WATER_HEIGHT as u16);
        water.configure_world(seed, true);
        water.set_bleach(1.0);
        water.update(Duration::ZERO);
        let sky_directions = (0..SKY_WIDTH * SKY_HEIGHT)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i % SKY_WIDTH) as f32 / SKY_WIDTH as f32
                    + std::f32::consts::PI / SKY_WIDTH as f32;
                let polar =
                    std::f32::consts::PI * ((i / SKY_WIDTH) as f32 + 0.5) / SKY_HEIGHT as f32;
                let (radial, z) = polar.sin_cos();
                let (y, x) = theta.sin_cos();
                [radial * x, radial * y, z]
            })
            .collect();
        let mut result = Self {
            water,
            bed: bed::BedWorld::new(seed),
            travel: world::WorldTravel::default(),
            rng: StdRng::seed_from_u64(seed),
            seed,
            elapsed: 0.0,
            wall_delta: 0.0,
            rain_accumulator: 0.0,
            speed: motion::DEFAULT_SPEED,
            rain: 32.0,
            drop_size: 1.5,
            bleach: 1.0,
            trip: scene::DEFAULT_INTENSITY,
            rgb_shift: 2.0,
            mode: 0,
            scene: SceneKind::Journey,
            quality: 2,
            current: true,
            paused: false,
            effects: true,
            failed: false,
            frame: 0,
            bed_revision: 0,
            bed_origin: None,
            uniforms: ARUniforms::default(),
            water_pixels: vec![[0.0; 4]; WATER_WIDTH * WATER_HEIGHT],
            dye_pixels: vec![[0.0; 4]; WATER_WIDTH * WATER_HEIGHT],
            bed_pixels: vec![[0.0; 4]; BED_SIZE * BED_SIZE],
            sky_pixels: vec![[0.0; 4]; SKY_WIDTH * SKY_HEIGHT],
            light_pixels: vec![[0.0; 4]; LIGHT_SIZE * LIGHT_SIZE],
            sky_directions,
            drops: Vec::with_capacity(drops::MAX_DROPS),
            drop_pixels: Vec::with_capacity(drops::MAX_DROPS),
            next_drop_id: 1,
            cloud_revision: 1,
            flow_pixels: vec![[0.0; 4]; WATER_WIDTH * WATER_HEIGHT],
        };
        result.refresh();
        result
    }

    fn add_rain(&mut self) {
        let strength = 0.08 + self.rng.r#gen::<f32>().powi(2) * 0.28;
        let origin = self.water.world_origin();
        let world = [
            origin[0] + self.rng.r#gen::<f64>() * 2.0,
            origin[1] + self.rng.r#gen::<f64>() * 2.0,
        ];
        self.spawn_drop(world, self.drop_size, strength);
    }

    fn surface_height(
        &self,
        world: [f64; 2],
    ) -> Option<f32> {
        let origin = self.water.world_origin();
        let local: [f64; 2] = std::array::from_fn(|i| world[i] - origin[i]);
        if local
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=2.0).contains(v))
        {
            return None;
        }
        let x = (local[0] as f32 * 0.5 * WATER_WIDTH as f32).min((WATER_WIDTH - 1) as f32);
        let y = (local[1] as f32 * 0.5 * WATER_HEIGHT as f32).min((WATER_HEIGHT - 1) as f32);
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = (x0 + 1).min(WATER_WIDTH - 1);
        let y1 = (y0 + 1).min(WATER_HEIGHT - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let levels = &self.water.levels;
        let a = levels[[x0, y0]] * (1.0 - tx) + levels[[x1, y0]] * tx;
        let b = levels[[x0, y1]] * (1.0 - tx) + levels[[x1, y1]] * tx;
        Some(a * (1.0 - ty) + b * ty)
    }

    fn spawn_drop(
        &mut self,
        world: [f64; 2],
        size: f32,
        strength: f32,
    ) -> bool {
        if !size.is_finite()
            || !(0.25..=3.0).contains(&size)
            || self.drops.len() >= drops::MAX_DROPS
            || self.next_drop_id == u64::MAX
        {
            return false;
        }
        let Some(surface) = self.surface_height(world) else {
            return false;
        };
        self.drops.push(drops::Drop::new(
            self.next_drop_id,
            world,
            self.water.elapsed().as_secs_f64(),
            surface,
            strength,
            size,
        ));
        self.next_drop_id += 1;
        true
    }

    fn advance_drops(&mut self) {
        let time = self.water.elapsed().as_secs_f64();
        let origin = self.water.world_origin();
        // Sample before mutating the source list; at most 512 short-lived records.
        let surfaces: Vec<_> = self
            .drops
            .iter()
            .map(|drop| self.surface_height(drop.world))
            .collect();
        let mut impacted = false;
        for (drop, surface) in self.drops.iter_mut().zip(surfaces) {
            if let Some(surface) = surface
                && drop.advance(time, surface)
            {
                self.water.add_ballistic_drop(
                    [
                        ((drop.world[0] - origin[0]) * 0.5) as f32,
                        ((drop.world[1] - origin[1]) * 0.5) as f32,
                    ],
                    drop.radius,
                    drop.impact_speed,
                    Duration::from_secs_f64((time - drop.impact.unwrap()).max(0.0)),
                );
                impacted = true;
            }
        }
        self.drops.retain(|drop| {
            drop.retained(time)
                && (drop.impact.is_some()
                    || (0..2).all(|i| (0.0..=2.0).contains(&(drop.world[i] - origin[i]))))
        });
        if impacted {
            self.water.update(Duration::ZERO);
        }
    }

    fn step(
        &mut self,
        wall_delta: f64,
    ) {
        if self.paused || wall_delta == 0.0 {
            self.wall_delta = 0.0;
            self.uniforms.time.0[1] = 0.0;
            return;
        }
        // App activation and display stalls cannot inject a huge unstable catch-up.
        self.wall_delta = wall_delta.min(0.1) as f32;
        let scene_dt = Duration::from_secs_f32(self.wall_delta * self.speed);
        self.elapsed += scene_dt.as_secs_f32();
        let physics_dt = motion::surface_delta(scene_dt);
        self.travel.advance(
            scene_dt.as_secs_f32(),
            self.mode == 0 && !self.water.has_numerical_flow(),
        );
        self.water.set_world_center(self.travel.center());
        self.rain_accumulator += physics_dt.as_secs_f32() * self.rain;
        let count = (self.rain_accumulator as usize).min(32);
        self.rain_accumulator -= count as f32;
        for _ in 0..count {
            self.add_rain();
        }
        self.water.update(physics_dt);
        self.advance_drops();
        self.frame = self.frame.wrapping_add(1);
        self.refresh();
    }

    fn command(
        &mut self,
        command: u32,
        value: f32,
    ) -> bool {
        if !value.is_finite() {
            return false;
        }
        match command {
            1 => self.paused = !self.paused,
            2 => self.speed = value.clamp(0.1, 3.0),
            3 => self.rain = value.clamp(0.0, 60.0),
            4 => self.drop_size = value.clamp(0.25, 3.0),
            5 => {
                self.bleach = value.clamp(0.0, 4.0);
                self.water.set_bleach(self.bleach);
            },
            6 => self.trip = value.clamp(0.0, 4.0),
            7 => self.rgb_shift = value.clamp(0.0, 12.0),
            8 if value.fract() == 0.0 && (0.0..=2.0).contains(&value) => {
                self.quality = value as u32
            },
            9 if value.fract() == 0.0 && (0.0..=2.0).contains(&value) => {
                if self.mode != value as u32 {
                    // The hydraulic modes have different wet columns. Leaving
                    // pigment below a newly raised bed would freeze it outside
                    // the fluid and resurrect it when returning to deeper water.
                    self.drops.clear();
                    self.rain_accumulator = 0.0;
                    self.cloud_revision = self.cloud_revision.wrapping_add(1);
                }
                self.mode = value as u32;
                self.water.set_drain(self.mode == 1);
                self.water.set_rapids(self.mode == 2);
            },
            10 if value.fract() == 0.0 && (0.0..=6.0).contains(&value) => {
                self.scene = [
                    SceneKind::Journey,
                    SceneKind::Acid,
                    SceneKind::Aurora,
                    SceneKind::Storm,
                    SceneKind::Abyss,
                    SceneKind::Lava,
                    SceneKind::Prism,
                ][value as usize];
            },
            11 => self.travel.toggle(),
            12 => {
                self.current = !self.current;
                self.water.set_current(self.current);
            },
            13 => self.effects = !self.effects,
            14 => {
                for _ in 0..12 {
                    self.add_rain();
                }
            },
            15 => {
                if !self.spawn_drop(self.travel.center(), self.drop_size, 0.2) {
                    return false;
                }
            },
            16 => {
                self.drops.clear();
                self.rain_accumulator = 0.0;
                self.cloud_revision = self.cloud_revision.wrapping_add(1);
            },
            _ => return false,
        }
        self.water.update(Duration::ZERO);
        self.wall_delta = 0.0;
        self.refresh();
        true
    }

    fn pack_water(&mut self) -> [f32; 2] {
        let mut bounds = [f32::INFINITY, f32::NEG_INFINITY];
        let levels = &self.water.levels;
        for y in 0..WATER_HEIGHT {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(WATER_HEIGHT - 1);
            for x in 0..WATER_WIDTH {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(WATER_WIDTH - 1);
                let h = levels[[x, y]];
                bounds[0] = bounds[0].min(h);
                bounds[1] = bounds[1].max(h);
                let source = x * WATER_HEIGHT + y;
                let target = y * WATER_WIDTH + x;
                self.water_pixels[target] = [
                    h,
                    (levels[[xp, y]] - levels[[xm, y]]) * WATER_WIDTH as f32
                        / (2.0 * (xp - xm) as f32),
                    (levels[[x, yp]] - levels[[x, ym]]) * WATER_HEIGHT as f32
                        / (2.0 * (yp - ym) as f32),
                    self.water.foam[source],
                ];
                let dye = self.water.dye[source];
                self.dye_pixels[target] = [dye[0], dye[1], dye[2], self.water.depths[source]];
                let velocity = self.water.native_transport_velocity(x, y);
                self.flow_pixels[target] =
                    [velocity[0], velocity[1], 0.0, self.water.depths[source]];
            }
        }
        bounds
    }

    fn refresh(&mut self) {
        let bounds = self.pack_water();
        let origin = self.water.world_origin();
        self.drop_pixels.clear();
        self.drop_pixels
            .extend(self.drops.iter().map(|drop| drop.export(origin)));
        if self.bed_origin != Some(origin) {
            self.bed.prepare(origin, self.seed);
            for (i, pixel) in self.bed_pixels.iter_mut().enumerate() {
                let local = [i % BED_SIZE, i / BED_SIZE]
                    .map(|v| BED_MIN + BED_SPAN * (v as f32 + 0.5) / BED_SIZE as f32);
                *pixel = self.bed.sample_patch(local.map(|v| v / 2.0));
            }
            self.bed_origin = Some(origin);
            self.bed_revision = self.bed_revision.wrapping_add(1);
        }
        let scene = SceneFrame::new(self.scene, self.elapsed, self.trip);
        let mut diffuse = [0.0_f32; 3];
        let mut weight_sum = 0.0;
        for ((pixel, direction), i) in self
            .sky_pixels
            .iter_mut()
            .zip(&self.sky_directions)
            .zip(0..)
        {
            let (r, g, b) = colors::sky_color(self.elapsed, *direction, &scene);
            *pixel = [r, g, b, 1.0];
            if direction[2] > 0.0 {
                let polar =
                    std::f32::consts::PI * ((i / SKY_WIDTH) as f32 + 0.5) / SKY_HEIGHT as f32;
                let weight = polar.sin() * direction[2];
                weight_sum += weight;
                for c in 0..3 {
                    diffuse[c] += pixel[c] * weight;
                }
            }
        }
        for value in &mut diffuse {
            *value *= 0.98 / weight_sum;
        }
        for (i, pixel) in self.light_pixels.iter_mut().enumerate() {
            let local = [i % LIGHT_SIZE, i / LIGHT_SIZE]
                .map(|v| BED_MIN + BED_SPAN * (v as f32 + 0.5) / LIGHT_SIZE as f32);
            let point = std::array::from_fn(|c| origin[c] + f64::from(local[c]));
            let light = bed::projected_light(point, scene.sky_phase[2], self.seed);
            let strength = (0.5 * self.trip).clamp(0.0, 1.0);
            let light = light.map(|v| 1.0 + (0.12 + 0.88 * v - 1.0) * strength);
            *pixel = [light[0], light[1], light[2], 1.0];
        }
        let center = self.travel.center();
        self.uniforms = ARUniforms {
            time: Float4([self.elapsed, self.wall_delta, self.trip, self.rgb_shift]),
            camera: Float4([
                (center[0] - origin[0]) as f32,
                (center[1] - origin[1]) as f32,
                scene.camera_roll,
                scene.camera_elevation,
            ]),
            optics: Float4([
                scene.fov,
                scene.absorption_scale,
                scene.exposure,
                scene.saturation,
            ]),
            gain: Float4([scene.gain[0], scene.gain[1], scene.gain[2], scene.contrast]),
            light: Float4([
                scene.light_dir[0],
                scene.light_dir[1],
                scene.light_dir[2],
                scene.interference_strength,
            ]),
            world: Float4([
                self.water.elapsed().as_secs_f32(),
                bounds[0],
                bounds[1],
                self.scene as u32 as f32,
            ]),
            post: Float4([
                u32::from(self.effects) as f32,
                self.quality as f32,
                0.0,
                0.0,
            ]),
            illumination: Float4([diffuse[0], diffuse[1], diffuse[2], 0.0]),
            ..ARUniforms::default()
        };
    }

    fn snapshot(&self) -> ARSnapshot {
        ARSnapshot {
            water_width: WATER_WIDTH as u32,
            water_height: WATER_HEIGHT as u32,
            bed_width: BED_SIZE as u32,
            bed_height: BED_SIZE as u32,
            sky_width: SKY_WIDTH as u32,
            sky_height: SKY_HEIGHT as u32,
            light_width: LIGHT_SIZE as u32,
            light_height: LIGHT_SIZE as u32,
            water: self.water_pixels.as_ptr().cast(),
            dye: self.dye_pixels.as_ptr().cast(),
            bed: self.bed_pixels.as_ptr().cast(),
            sky: self.sky_pixels.as_ptr().cast(),
            bed_light: self.light_pixels.as_ptr().cast(),
            uniforms: self.uniforms,
            speed: self.speed,
            rain: self.rain,
            drop_size: self.drop_size,
            bleach: self.bleach,
            mode: self.mode,
            scene: self.scene as u32,
            travel: u32::from(self.travel.enabled()),
            current: u32::from(self.current),
            paused: u32::from(self.paused),
            quality: self.quality,
            effects: u32::from(self.effects),
            frame: self.frame,
            bed_revision: self.bed_revision,
            world_origin_x: self.water.world_origin()[0],
            world_origin_y: self.water.world_origin()[1],
            physical_time: self.water.elapsed().as_secs_f64(),
            drops: self.drop_pixels.as_ptr(),
            drop_count: self.drop_pixels.len() as u32,
            reserved: 0,
            cloud_revision: self.cloud_revision,
            flow: self.flow_pixels.as_ptr().cast(),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ar_core_abi_version() -> u32 { 2 }

#[unsafe(no_mangle)]
pub extern "C" fn ar_core_create(seed: u64) -> *mut Core {
    catch_unwind(|| Box::into_raw(Box::new(Core::new(seed)))).unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `core` must be null or a live pointer returned by `ar_core_create`, destroyed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ar_core_destroy(core: *mut Core) {
    if !core.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(core));
        }));
    }
}

/// # Safety
/// `core` must be null or a live, uniquely borrowed core. Calls must be serialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ar_core_step(
    core: *mut Core,
    wall_delta_seconds: f64,
) -> i32 {
    if core.is_null() || !wall_delta_seconds.is_finite() || wall_delta_seconds < 0.0 {
        return 0;
    }
    let core = unsafe { &mut *core };
    if core.failed {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| core.step(wall_delta_seconds))) {
        Ok(()) => 1,
        Err(_) => {
            core.failed = true;
            0
        },
    }
}

/// # Safety
/// `core` must be null or a live, uniquely borrowed core. Calls must be serialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ar_core_command(
    core: *mut Core,
    command: u32,
    value: f32,
) -> i32 {
    if core.is_null() {
        return 0;
    }
    let core = unsafe { &mut *core };
    if core.failed {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| core.command(command, value))) {
        Ok(accepted) => i32::from(accepted),
        Err(_) => {
            core.failed = true;
            0
        },
    }
}

/// # Safety
/// `core` must be null or a live, uniquely borrowed core. Calls are serialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ar_core_drop(
    core: *mut Core,
    world_x: f64,
    world_y: f64,
    size: f32,
) -> i32 {
    if core.is_null() || !world_x.is_finite() || !world_y.is_finite() || !size.is_finite() {
        return 0;
    }
    let core = unsafe { &mut *core };
    if core.failed {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        if !core.spawn_drop([world_x, world_y], size, 0.2) {
            return false;
        }
        core.wall_delta = 0.0;
        core.refresh();
        true
    })) {
        Ok(accepted) => i32::from(accepted),
        Err(_) => {
            core.failed = true;
            0
        },
    }
}

/// # Safety
/// Pointers must be null or valid for the declared access. Borrowed texture
/// pointers expire on the next mutating call. All core calls must be serialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ar_core_snapshot(
    core: *const Core,
    snapshot: *mut ARSnapshot,
) -> i32 {
    if core.is_null() || snapshot.is_null() {
        return 0;
    }
    let core = unsafe { &*core };
    if core.failed {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        snapshot.write(core.snapshot());
    })) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[test]
    fn native_rain_waits_for_contact_and_does_not_duplicate_volume_pigment() {
        let mut core = Core::new(101);
        let before = core.water.levels.clone();
        core.add_rain();
        core.water.update(Duration::ZERO);
        assert_eq!(
            core.water.levels, before,
            "airborne rain cannot disturb the surface"
        );
        assert!(
            !core.water.has_dye(),
            "native pigment must have one volume owner"
        );
    }

    #[test]
    fn ballistic_contact_disturbs_the_first_post_contact_surface_in_every_mode() {
        for mode in 0..3 {
            let mut hit = Core::new(0xAC1D2026);
            let mut calm = Core::new(0xAC1D2026);
            for core in [&mut hit, &mut calm] {
                core.command(3, 0.0);
                core.command(11, 0.0);
                core.command(12, 0.0);
                core.command(9, mode as f32);
                for _ in 0..12 {
                    core.step(0.1);
                }
            }
            assert!(hit.spawn_drop(hit.travel.center(), 1.5, 0.2));
            let mut found = false;
            for _ in 0..60 {
                hit.step(1.0 / 60.0);
                calm.step(1.0 / 60.0);
                if hit.drops[0].impact.is_some() {
                    let peak = hit
                        .water_pixels
                        .iter()
                        .zip(&calm.water_pixels)
                        .map(|(a, b)| (a[0] - b[0]).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        peak > 0.00001,
                        "mode {mode}: contact snapshot lost the accepted substep impact age; \
                         peak={peak}"
                    );
                    found = true;
                    break;
                }
                assert_eq!(hit.water_pixels, calm.water_pixels);
            }
            assert!(found);
        }
    }

    #[test]
    fn native_uniforms_match_the_c_metal_abi() {
        assert_eq!(std::mem::size_of::<ARUniforms>(), 208);
        assert_eq!(std::mem::align_of::<ARUniforms>(), 16);
        assert_eq!(std::mem::offset_of!(ARUniforms, illumination), 192);
        assert_eq!(std::mem::offset_of!(ARSnapshot, uniforms), 80);
        assert_eq!(std::mem::size_of::<ARDrop>(), 64);
        assert_eq!(std::mem::align_of::<ARDrop>(), 16);
        assert_eq!(std::mem::offset_of!(ARDrop, identity), 48);
        assert_eq!(std::mem::offset_of!(ARSnapshot, world_origin_x), 352);
        assert_eq!(std::mem::offset_of!(ARSnapshot, drops), 376);
        assert_eq!(std::mem::offset_of!(ARSnapshot, flow), 400);
        assert_eq!(std::mem::size_of::<ARSnapshot>(), 416);
    }

    #[test]
    fn core_defaults_and_exports_preserve_favorite_settings() {
        let core = Core::new(42);
        let frame = core.snapshot();
        assert_eq!([frame.water_width, frame.water_height], [192, 128]);
        assert_eq!([frame.bed_width, frame.bed_height], [256, 256]);
        assert_eq!([frame.sky_width, frame.sky_height], [128, 64]);
        assert_eq!([frame.light_width, frame.light_height], [64, 64]);
        assert_eq!([frame.speed, frame.rain, frame.drop_size, frame.bleach], [
            2.4, 32.0, 1.5, 1.0
        ]);
        assert_eq!(
            [
                frame.mode,
                frame.scene,
                frame.travel,
                frame.current,
                frame.paused,
                frame.quality,
                frame.effects
            ],
            [0, 0, 1, 1, 0, 2, 1]
        );
        assert_eq!([frame.uniforms.time.0[2], frame.uniforms.time.0[3]], [
            4.0, 2.0
        ]);
        for texture in [
            &core.water_pixels,
            &core.dye_pixels,
            &core.bed_pixels,
            &core.sky_pixels,
            &core.light_pixels,
            &core.flow_pixels,
        ] {
            assert!(!texture.is_empty());
            assert!(texture.iter().flatten().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn pause_holds_physics_rain_and_world_without_catchup() {
        let mut core = Core::new(73);
        core.step(0.04);
        core.command(1, 0.0);
        let pixels = core.water_pixels.clone();
        let dye = core.dye_pixels.clone();
        let center = core.travel.center();
        let elapsed = core.elapsed;
        let frame = core.frame;
        let drops = core.drop_pixels.clone();
        let physical_time = core.snapshot().physical_time;
        core.step(0.1);
        assert_eq!(core.water_pixels, pixels);
        assert_eq!(core.dye_pixels, dye);
        assert_eq!(core.travel.center(), center);
        assert_eq!(core.elapsed, elapsed);
        assert_eq!(core.frame, frame);
        assert_eq!(core.drop_pixels, drops);
        assert_eq!(core.snapshot().physical_time, physical_time);
        core.command(1, 0.0);
        core.step(0.01);
        assert!((core.elapsed - elapsed - 0.024).abs() < 1e-6);
    }

    #[test]
    fn control_validation_and_travel_are_independent_of_current() {
        let mut core = Core::new(91);
        assert!(!core.command(2, f32::NAN));
        assert!(!core.command(8, 1.5));
        assert!(!core.command(99, 0.0));
        assert!(core.command(2, 100.0));
        assert_eq!(core.speed, 3.0);
        assert!(core.command(3, -100.0));
        assert_eq!(core.rain, 0.0);
        core.command(11, 0.0);
        let held = core.travel.center();
        core.step(0.04);
        assert_eq!(core.travel.center(), held);
        assert!(core.current);
        core.command(11, 0.0);
        core.command(12, 0.0);
        core.step(0.04);
        assert_ne!(core.travel.center(), held);
        assert!(!core.current);
    }

    #[test]
    fn exported_surface_has_correct_row_major_axes_and_metric_slopes() {
        let mut core = Core::new(11);
        for x in 0..WATER_WIDTH {
            for y in 0..WATER_HEIGHT {
                core.water.levels[[x, y]] = x as f32 * (2.0 / WATER_WIDTH as f32) * 0.15
                    + y as f32 * (2.0 / WATER_HEIGHT as f32) * -0.08;
            }
        }
        core.pack_water();
        for (x, y) in [(0, 0), (25, 61), (191, 127)] {
            let p = core.water_pixels[y * WATER_WIDTH + x];
            assert_eq!(p[0], core.water.levels[[x, y]]);
            assert!((p[1] - 0.15).abs() < 1e-5);
            assert!((p[2] + 0.08).abs() < 1e-5);
        }
    }

    #[test]
    fn invalid_ffi_inputs_fail_without_unwinding() {
        unsafe {
            assert_eq!(ar_core_step(std::ptr::null_mut(), 0.01), 0);
            assert_eq!(ar_core_command(std::ptr::null_mut(), 1, 0.0), 0);
            assert_eq!(ar_core_snapshot(std::ptr::null(), std::ptr::null_mut()), 0);
            let core = ar_core_create(91);
            assert!(!core.is_null());
            assert_eq!(ar_core_step(core, f64::NAN), 0);
            assert_eq!(ar_core_step(core, -0.1), 0);
            assert_eq!(ar_core_step(core, 0.01), 1);
            ar_core_destroy(core);
            ar_core_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn seeded_rain_and_exports_repeat_without_changing_the_source_model() {
        let mut a = Core::new(179);
        let mut b = Core::new(179);
        for dt in [0.04, 0.08, 0.01, 0.1] {
            a.step(dt);
            b.step(dt);
        }
        assert_eq!(a.water_pixels, b.water_pixels);
        assert_eq!(a.dye_pixels, b.dye_pixels);
        assert_eq!(a.bed_pixels, b.bed_pixels);
        assert_eq!(a.drop_pixels, b.drop_pixels);
        assert!(!a.drop_pixels.is_empty());
        assert!(a.dye_pixels.iter().all(|p| p[0] + p[1] + p[2] == 0.0));
        assert!(a.dye_pixels.iter().all(|p| p[3] > 0.0));
    }

    #[test]
    fn numerical_modes_hold_travel_and_export_finite_water() {
        let mut core = Core::new(183);
        core.step(0.04);
        let center = core.travel.center();
        for mode in [1, 2] {
            assert!(core.command(9, mode as f32));
            for _ in 0..6 {
                core.step(0.05);
            }
            assert_eq!(core.travel.center(), center);
            assert_eq!(core.snapshot().mode, mode);
            assert!(core.water_pixels.iter().flatten().all(|v| v.is_finite()));
            assert!(
                core.dye_pixels
                    .iter()
                    .all(|p| p.iter().all(|v| v.is_finite() && *v >= 0.0))
            );
        }
        core.command(9, 0.0);
        for _ in 0..12 {
            core.step(0.1);
        }
        assert!(core.travel.center()[0] > center[0]);
    }

    #[test]
    fn bed_upload_changes_only_on_rebase_and_zero_trip_light_is_neutral() {
        let mut core = Core::new(189);
        let revision = core.bed_revision;
        core.command(11, 0.0);
        core.step(0.1);
        assert_eq!(core.bed_revision, revision);
        core.command(11, 0.0);
        for _ in 0..5 {
            core.step(0.1);
        }
        assert!(core.bed_revision > revision);
        core.command(6, 0.0);
        assert!(core.light_pixels.iter().all(|p| *p == [1.0; 4]));
    }

    #[test]
    fn one_visible_drop_creates_one_retained_impact_without_column_dye() {
        let mut core = Core::new(197);
        core.command(3, 0.0);
        core.command(11, 0.0);
        assert!(core.command(15, 0.0));
        let id = core.drop_pixels[0].identity;
        assert_eq!(core.drop_pixels[0].identity[2], 0);
        for _ in 0..8 {
            core.step(0.1);
        }
        assert_eq!(core.drop_pixels.len(), 1);
        let impact = core.drop_pixels[0];
        assert_eq!(impact.identity[..2], id[..2]);
        assert_eq!(impact.identity[2], 1);
        assert!(impact.dynamics.0[2] > 0.0);
        assert!(f64::from(impact.dynamics.0[2]) <= core.snapshot().physical_time);
        assert!(!core.water.has_dye());
        for _ in 0..12 {
            core.step(0.1);
        }
        assert_eq!(
            core.drop_pixels[0], impact,
            "retained event must not change or reinject"
        );
        for _ in 0..60 {
            core.step(0.1);
        }
        assert!(
            core.drop_pixels.is_empty(),
            "old journal entries must retire"
        );
        assert!(!core.water.has_dye());
    }

    #[test]
    fn source_cap_clear_and_world_coordinates_preserve_event_identity() {
        let mut core = Core::new(199);
        core.command(3, 0.0);
        for _ in 0..drops::MAX_DROPS {
            assert!(core.spawn_drop([0.0, 0.0], 1.0, 0.2));
        }
        assert!(!core.spawn_drop([0.0, 0.0], 1.0, 0.2));
        let last = core.next_drop_id - 1;
        let revision = core.cloud_revision;
        core.command(16, 0.0);
        assert_eq!(core.cloud_revision, revision + 1);
        assert!(core.drop_pixels.is_empty());
        assert!(core.spawn_drop([0.2, -0.1], 1.0, 0.2));
        assert!(core.drops[0].id > last);
        let world = core.drops[0].world;
        core.water.set_world_center([0.125, -0.125]);
        core.refresh();
        let exported = core.drop_pixels[0];
        let origin = core.water.world_origin();
        for axis in 0..2 {
            assert!(
                (f64::from(exported.position.0[axis]) + origin[axis] - world[axis]).abs() < 1e-6
            );
        }
        assert_eq!(core.drops[0].world, world);
    }

    #[test]
    fn hydraulic_mode_changes_clear_clouds_once_without_reusing_source_ids() {
        let mut core = Core::new(199);
        core.command(3, 0.0);
        core.command(11, 0.0);
        assert!(core.spawn_drop([0.0; 2], 1.0, 0.2));
        for _ in 0..5 {
            core.step(0.1);
        }
        assert!(core.drops[0].impact.is_some());
        assert!(core.spawn_drop([0.0; 2], 1.0, 0.2));
        assert!(core.drops[1].impact.is_none());
        let revision = core.cloud_revision;
        let next_id = core.next_drop_id;
        assert!(core.command(9, 0.0));
        assert_eq!(core.cloud_revision, revision);
        assert_eq!(core.drop_pixels.len(), 2);
        assert!(core.command(9, 1.0));
        assert_eq!(core.cloud_revision, revision + 1);
        assert!(core.drops.is_empty() && core.drop_pixels.is_empty());
        assert_eq!(core.next_drop_id, next_id);
        assert!(core.spawn_drop([0.0; 2], 1.0, 0.2));
        assert_eq!(core.drops[0].id, next_id);
        assert!(core.command(9, 1.0));
        assert_eq!(core.cloud_revision, revision + 1);
        assert_eq!(core.drop_pixels.len(), 1);
        assert!(core.command(9, 2.0));
        assert_eq!(core.cloud_revision, revision + 2);
        assert!(core.drop_pixels.is_empty());
    }

    #[test]
    fn explicit_drop_ffi_validates_coordinates_and_publishes_without_advancing_time() {
        unsafe {
            assert_eq!(ar_core_drop(std::ptr::null_mut(), 0.0, 0.0, 1.0), 0);
            let core = ar_core_create(211);
            assert!(!core.is_null());
            assert_eq!(ar_core_drop(core, f64::NAN, 0.0, 1.0), 0);
            assert_eq!(ar_core_drop(core, 0.0, f64::INFINITY, 1.0), 0);
            assert_eq!(ar_core_drop(core, 0.0, 0.0, 0.1), 0);
            assert_eq!(ar_core_drop(core, 100.0, 0.0, 1.0), 0);
            assert_eq!(ar_core_drop(core, 0.0, 0.0, 1.5), 1);
            let state = (&*core).snapshot();
            assert_eq!(state.physical_time, 0.0);
            assert_eq!(state.drop_count, 1);
            assert_eq!((&*core).drop_pixels[0].identity[2], 0);
            ar_core_destroy(core);
        }
    }

    #[test]
    fn exported_flow_preserves_axes_units_and_actual_depth() {
        let mut core = Core::new(223);
        for x in 0..WATER_WIDTH {
            for y in 0..WATER_HEIGHT {
                let source = x * WATER_HEIGHT + y;
                core.water.surface_velocity[source] = [x as f32 * 0.01, y as f32 * -0.02];
                core.water.depths[source] = 0.3 + y as f32 * 0.001;
            }
        }
        core.pack_water();
        for (x, y) in [(0, 0), (25, 61), (191, 127)] {
            assert_eq!(core.flow_pixels[y * WATER_WIDTH + x], [
                x as f32 * 0.01,
                y as f32 * -0.02,
                0.0,
                0.3 + y as f32 * 0.001,
            ]);
        }
    }

    /// Paired native-core evidence: the only difference is one ballistic source.
    /// Run explicitly with --ignored --nocapture; output can be versioned with
    /// AR_IMPACT_DIAGNOSTIC so before/after measurements use identical sampling.
    #[test]
    #[ignore]
    fn paired_ballistic_impact_diagnostic() {
        let mut records = Vec::new();
        println!(
            "mode size age_s peak_mm rms_mm peak_slope rms_slope radial_rms_m baseline_rms_mm \
             baseline_slope_rms"
        );
        for mode in 0..3 {
            for size in [1.5_f32, 3.0] {
                let mut hit = Core::new(0xAC1D2026);
                let mut calm = Core::new(0xAC1D2026);
                for core in [&mut hit, &mut calm] {
                    core.command(3, 0.0);
                    core.command(11, 0.0);
                    core.command(12, 0.0);
                    core.command(9, mode as f32);
                }
                for _ in 0..34 {
                    hit.step(0.1);
                    calm.step(0.1);
                }
                let center = hit.travel.center();
                assert!(hit.spawn_drop(center, size, 0.2));
                let mut next = 0;
                let target_ages = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0];
                for _ in 0..350 {
                    let wall_dt = 0.005 / (f64::from(hit.speed) * 0.25);
                    hit.step(wall_dt);
                    calm.step(wall_dt);
                    let Some(impact) = hit.drops[0].impact else {
                        assert_eq!(hit.water_pixels, calm.water_pixels);
                        continue;
                    };
                    let age = hit.water.elapsed().as_secs_f64() - impact;
                    if age < target_ages[next] {
                        continue;
                    }
                    let origin = hit.water.world_origin();
                    let local = [center[0] - origin[0], center[1] - origin[1]];
                    let mut peak = 0.0_f64;
                    let mut sum2 = 0.0_f64;
                    let mut slope_peak = 0.0_f64;
                    let mut slope2 = 0.0_f64;
                    let mut radius2_weighted = 0.0_f64;
                    let mut base_sum2 = 0.0_f64;
                    let mut base_slope2 = 0.0_f64;
                    for (i, (a, b)) in hit.water_pixels.iter().zip(&calm.water_pixels).enumerate() {
                        let delta = f64::from(a[0] - b[0]);
                        let sx = f64::from(a[1] - b[1]);
                        let sy = f64::from(a[2] - b[2]);
                        let s2 = sx * sx + sy * sy;
                        let dx = (i % WATER_WIDTH) as f64 * 2.0 / WATER_WIDTH as f64 - local[0];
                        let dy = (i / WATER_WIDTH) as f64 * 2.0 / WATER_HEIGHT as f64 - local[1];
                        peak = peak.max(delta.abs());
                        sum2 += delta * delta;
                        slope_peak = slope_peak.max(s2.sqrt());
                        slope2 += s2;
                        radius2_weighted += delta * delta * (dx * dx + dy * dy);
                        base_sum2 += f64::from(b[0]).powi(2);
                        base_slope2 += f64::from(b[1]).powi(2) + f64::from(b[2]).powi(2);
                    }
                    let count = hit.water_pixels.len() as f64;
                    let rms = (sum2 / count).sqrt();
                    let slope_rms = (slope2 / count).sqrt();
                    let radius_rms = (radius2_weighted / sum2.max(1e-30)).sqrt();
                    let baseline_rms = (base_sum2 / count).sqrt();
                    let baseline_slope_rms = (base_slope2 / count).sqrt();
                    println!(
                        "{mode} {size:.1} {age:.4} {:.5} {:.5} {slope_peak:.6} {slope_rms:.6} \
                         {radius_rms:.5} {:.5} {baseline_slope_rms:.6}",
                        peak * 1000.0,
                        rms * 1000.0,
                        baseline_rms * 1000.0
                    );
                    records.push(format!(
                        "{{\"mode\":{mode},\"size\":{size},\"age_s\":{age},\"peak_m\":{peak},\"\
                         rms_m\":{rms},\"peak_slope\":{slope_peak},\"rms_slope\":{slope_rms},\"\
                         radial_rms_m\":{radius_rms},\"baseline_rms_m\":{baseline_rms},\"\
                         baseline_slope_rms\":{baseline_slope_rms},\"impact_time\":{impact}}}"
                    ));
                    next += 1;
                    if next == target_ages.len() {
                        break;
                    }
                }
                assert_eq!(next, target_ages.len());
            }
        }
        let output = std::env::var("AR_IMPACT_DIAGNOSTIC").unwrap_or_else(|_| {
            format!(
                "{}/../artifacts/impact-fix/diagnosis.json",
                env!("CARGO_MANIFEST_DIR")
            )
        });
        std::fs::write(
            output,
            format!(
                "{{\"seed\":{},\"step_physical_s\":0.005,\"samples\":[{}]}}\n",
                0xAC1D2026_u64,
                records.join(",")
            ),
        )
        .unwrap();
    }
}
