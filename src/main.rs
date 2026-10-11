use std::{
    io::{
        IsTerminal,
        stdout,
    },
    time::{
        Duration,
        Instant,
    },
};

use anyhow::{
    Context,
    Result,
    ensure,
};
use clap::Parser;
use cli::{
    Options,
    QualityMode,
    WaterMode,
};
use crossterm::{
    cursor,
    event,
    execute,
    style,
    terminal,
};
use pacing::{
    FrameBudget,
    next_frame_deadline,
    simulation_size,
};
use rand::{
    Rng,
    SeedableRng,
    rngs::StdRng,
};
use scene::SceneKind;
use term::{
    RenderQuality,
    Renderer,
};
use user_input::{
    Action,
    key_action,
};
use water::Water;

mod bed;
mod benchmark;
mod cli;
mod color_pipeline;
mod colors;
mod drain;
mod dye;
mod help;
mod motion;
mod optics;
mod output;
mod pacing;
mod physics;
mod pigment_optics;
mod presentation;
mod rapids;
mod retina;
mod ripples;
mod scene;
mod settings;
mod stirring;
mod term;
mod user_input;
mod vortex;
mod water;
mod waves;
mod world;
mod world_flow;

/// Raw mode and the alternate screen must be restored on every return path.
struct TerminalSession;

impl TerminalSession {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        let session = Self;
        execute!(
            stdout(),
            terminal::EnterAlternateScreen,
            cursor::Hide,
            terminal::DisableLineWrap
        )?;
        Ok(session)
    }
}

fn restore_terminal() {
    let _ = terminal::disable_raw_mode();
    let _ = execute!(
        stdout(),
        terminal::EndSynchronizedUpdate,
        style::ResetColor,
        terminal::EnableLineWrap,
        cursor::Show,
        terminal::LeaveAlternateScreen
    );
}

impl Drop for TerminalSession {
    fn drop(&mut self) { restore_terminal(); }
}

fn selected_quality(
    mode: QualityMode,
    adaptive: &FrameBudget,
) -> RenderQuality {
    match mode {
        QualityMode::Auto => adaptive.quality(),
        QualityMode::High => RenderQuality::High,
        QualityMode::Balanced => RenderQuality::Balanced,
        QualityMode::Low => RenderQuality::Low,
    }
}

fn add_rain(
    water: &mut Water,
    rng: &mut StdRng,
    size: f32,
) {
    let strength = 0.08 + rng.r#gen::<f32>().powi(2) * 0.28;
    let (x, y) = (rng.r#gen(), rng.r#gen());
    if size == 1.0 {
        water.add_drop(x, y, strength);
    } else {
        water.add_drop_sized(x, y, strength, size);
    }
}

fn validate_screen_size(
    width: u16,
    height: u16,
) -> Result<()> {
    ensure!(
        usize::from(width) * usize::from(height) <= 500_000,
        "terminal exceeds 500,000 cells; increase the font size or use a smaller window"
    );
    Ok(())
}

fn configure_water(
    water: &mut Water,
    mode: WaterMode,
) {
    match mode {
        WaterMode::Drain => water.set_drain(true),
        WaterMode::Rapids => water.set_rapids(true),
        WaterMode::Classic => {
            water.set_drain(false);
            water.set_rapids(false);
        },
    }
}

fn poll_input(wait: Duration) -> std::io::Result<bool> {
    // The level-triggered Unix backend preserves concurrent resize/key readiness,
    // but crossterm 0.27 skips zero-timeout polls. Keep even drain polls positive
    // without adding a millisecond wait to each rendered frame.
    event::poll(wait.max(Duration::from_micros(100)))
}

fn run(mut options: Options) -> Result<()> {
    ensure!(
        stdout().is_terminal() && std::io::stdin().is_terminal(),
        "interactive mode needs a terminal; use --benchmark or --snapshot for headless rendering"
    );
    let (mut width, mut height) = terminal::size()?;
    validate_screen_size(width, height)?;
    let (sw, sh) = simulation_size(width, height);
    let mut water = Water::new(sw, sh);
    configure_water(&mut water, options.water_mode());
    water.set_bleach(options.bleach);
    water.configure_world(options.seed, true);
    water.set_current(!options.no_current);
    let mut travel = world::WorldTravel::default();
    if options.no_travel {
        travel.toggle();
    }
    let mut renderer = Renderer::new(width, height);
    renderer.set_scene(options.scene);
    renderer.set_intensity(options.intensity);
    renderer.set_rgb_shift(options.rgb_shift);
    renderer.set_subjective_intensity(options.subjective_intensity);
    renderer.set_world_view(travel.center(), water.world_origin(), options.seed);
    let mut adaptive = FrameBudget::new();
    let mut rng = StdRng::seed_from_u64(options.seed);
    let interval = Duration::from_secs_f64(1.0 / f64::from(options.fps));
    let mut last_frame = Instant::now();
    let mut deadline = last_frame;
    let mut elapsed = 0.0_f32;
    let mut rain_accum = 0.0_f32;
    let mut paused = false;
    let mut dirty = true;
    let mut status_until = None;
    let mut help_open = false;
    let mut help = help::HelpScreen::default();
    let mut trails = true;
    let mut interpolation = true;

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        previous_hook(info);
    }));
    let _session = TerminalSession::enter()?;
    // Drop the presenter (and join its writer) before restoring the terminal.
    let presenter = presentation::FramePresenter::new(width, height, options.fps)?;

    loop {
        presenter.check_error()?;
        let frame_interval = if options.quality == QualityMode::Auto {
            adaptive.frame_interval(interval)
        } else {
            interval
        };
        let wait = if (paused || help_open) && !dirty {
            Duration::from_millis(250)
        } else {
            deadline.saturating_duration_since(Instant::now())
        };
        if poll_input(wait)? {
            let mut pending_size = None;
            // Bound input draining so resize or key-repeat floods cannot starve drawing.
            for _ in 0..128 {
                match event::read()? {
                    event::Event::Resize(w, h) => pending_size = Some((w, h)),
                    event::Event::Key(key) if help_open => {
                        match key_action(key) {
                            Action::Help => help_open = false,
                            Action::Quit if key.code == event::KeyCode::Esc => help_open = false,
                            Action::Quit => return Ok(()),
                            _ if key.kind != event::KeyEventKind::Release => {
                                dirty |= help.navigate(key.code, height)
                            },
                            _ => {},
                        }
                        if !help_open {
                            last_frame = Instant::now();
                            deadline = last_frame;
                            renderer.invalidate();
                            dirty = true;
                        }
                    },
                    event::Event::Key(key) => match key_action(key) {
                        Action::Quit => return Ok(()),
                        Action::Help => {
                            help_open = true;
                            help = help::HelpScreen::default();
                            renderer.set_status(None);
                            status_until = None;
                            dirty = true;
                        },
                        Action::ToggleTrails => {
                            trails = !trails;
                            renderer.set_trails(trails);
                            renderer.set_status(Some(if trails {
                                "RGB echoes: on | e: toggle"
                            } else {
                                "RGB echoes: off | e: toggle"
                            }));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::ToggleInterpolation => {
                            interpolation = !interpolation;
                            presenter.set_enabled(interpolation)?;
                            renderer.set_status(Some(if interpolation {
                                "Frame blending: on | i: toggle"
                            } else {
                                "Frame blending: off | i: toggle"
                            }));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::ToggleTravel => {
                            travel.toggle();
                            options.no_travel = !travel.enabled();
                            renderer.set_status(Some(if travel.enabled() {
                                "Travel: on | c: toggle (Classic) | rotation continues"
                            } else {
                                "Travel: off | c: toggle (Classic) | rotation continues"
                            }));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::ToggleCurrent => {
                            options.no_current = !options.no_current;
                            water.set_current(!options.no_current);
                            renderer.set_status(Some(if options.no_current {
                                "Current: off | w: toggle (Classic) | local vortices remain"
                            } else {
                                "Current: on | w: toggle (Classic) | local vortices remain"
                            }));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action @ (Action::ToggleDrain | Action::ToggleRapids) => {
                            let target = if action == Action::ToggleDrain {
                                WaterMode::Drain
                            } else {
                                WaterMode::Rapids
                            };
                            options.toggle_water(target);
                            configure_water(&mut water, options.water_mode());
                            renderer.invalidate_geometry();
                            renderer.set_status(Some(options.water_mode().status()));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::NextScene => {
                            options.scene = options.scene.next();
                            renderer.set_scene(options.scene);
                            dirty = true;
                        },
                        Action::Journey => {
                            options.scene = SceneKind::Journey;
                            renderer.set_scene(options.scene);
                            dirty = true;
                        },
                        action @ (Action::NextQuality | Action::Quality(_)) => {
                            options.quality = match action {
                                Action::Quality(mode) => mode,
                                _ => options.quality.next(),
                            };
                            if options.quality == QualityMode::Auto {
                                adaptive = FrameBudget::new();
                            }
                            renderer.set_quality(selected_quality(options.quality, &adaptive));
                            renderer.set_status(Some(options.quality.status()));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::Pause => {
                            paused = !paused;
                            last_frame = Instant::now();
                            deadline = last_frame;
                            dirty = true;
                        },
                        Action::Intensity(delta) => {
                            options.intensity =
                                (options.intensity + delta).clamp(0.0, scene::MAX_INTENSITY);
                            renderer.set_intensity(options.intensity);
                            renderer.set_status(Some(&format!(
                                "Intensity: {:.1}x | +/-: adjust (0-4) | e: RGB trails",
                                options.intensity
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action @ (Action::Speed(_) | Action::ResetSpeed) => {
                            options.speed = match action {
                                Action::Speed(delta) => (options.speed + delta).clamp(0.1, 3.0),
                                _ => motion::DEFAULT_SPEED,
                            };
                            renderer.set_status(Some(&format!(
                                "Speed: {:.2}x | Left/Right or [/]: adjust | s: reset",
                                options.speed
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action
                        @ (Action::RainRate(_) | Action::DropSize(_) | Action::ResetRain) => {
                            match action {
                                Action::RainRate(delta) => options.adjust_rain(delta),
                                Action::DropSize(delta) => options.adjust_drop_size(delta),
                                _ => options.reset_rain(),
                            }
                            if options.rain == 0.0 {
                                rain_accum = 0.0;
                            }
                            renderer.set_status(Some(&format!(
                                "Rain: {:.1}/s | Drop size: {:.2}x | Up/Down: rate | ,/.: size | \
                                 R: reset",
                                options.rain, options.drop_size
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action @ (Action::Bleach(_) | Action::ResetBleach) => {
                            match action {
                                Action::Bleach(delta) => options.adjust_bleach(delta),
                                _ => options.reset_bleach(),
                            }
                            water.set_bleach(options.bleach);
                            renderer.set_status(Some(&format!(
                                "Bleach: {:.2}x | v/b: less/more | B: reset",
                                options.bleach
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action @ (Action::RgbShift(_) | Action::ResetRgbShift) => {
                            match action {
                                Action::RgbShift(delta) => options.adjust_rgb_shift(delta),
                                _ => options.reset_rgb_shift(),
                            }
                            renderer.set_rgb_shift(options.rgb_shift);
                            renderer.set_status(Some(&format!(
                                "RGB shift: {:.2}px | j/l: less/more | J: reset",
                                options.rgb_shift
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        action @ (Action::SubjectiveIntensity(_)
                        | Action::ResetSubjectiveIntensity) => {
                            match action {
                                Action::SubjectiveIntensity(delta) => {
                                    options.adjust_subjective_intensity(delta)
                                },
                                _ => options.reset_subjective_intensity(),
                            }
                            renderer.set_subjective_intensity(options.subjective_intensity);
                            renderer.set_status(Some(&format!(
                                "LSD: {:.1}x | n/m: less/more (0-2) | N: reset",
                                options.subjective_intensity
                            )));
                            status_until = Some(Instant::now() + Duration::from_secs(3));
                            dirty = true;
                        },
                        Action::Burst => {
                            for _ in 0..12 {
                                add_rain(&mut water, &mut rng, options.drop_size);
                            }
                            renderer.invalidate_geometry();
                            dirty = true;
                        },
                        Action::None => {},
                    },
                    _ => {},
                }
                if !poll_input(Duration::ZERO)? {
                    break;
                }
            }
            if let Some((w, h)) = pending_size {
                validate_screen_size(w, h)?;
                width = w;
                height = h;
                let (sw, sh) = simulation_size(w, h);
                if (water.width(), water.height()) != (sw, sh) {
                    water.resize(sw, sh);
                }
                renderer.resize(w, h);
                // Discard frames in the old viewport, including a minimized one.
                presenter.show_raw(Vec::new())?;
                dirty = true;
            }
        }
        if help_open {
            if dirty {
                presenter.show_raw(
                    help.encode(width, height, &options, paused, trails, interpolation)
                        .into_bytes(),
                )?;
                dirty = false;
            }
            continue;
        }
        if paused && !dirty {
            continue;
        }
        let now = Instant::now();
        if now < deadline && !dirty {
            continue;
        }
        if !paused && status_until.is_some_and(|until| now >= until) {
            renderer.set_status(None);
            status_until = None;
        }
        // Limit catch-up after a terminal stall. Numerical water and pigment
        // consume the same bounded physical interval; analytic waves sample its end.
        let render_dt = if paused {
            Duration::ZERO
        } else {
            now.duration_since(last_frame)
        };
        let wall_dt = render_dt.min(Duration::from_millis(100));
        let scene_dt = wall_dt.mul_f32(options.speed);
        let physics_dt = motion::surface_delta(scene_dt);
        last_frame = now;
        elapsed += scene_dt.as_secs_f32();
        let work_start = Instant::now();
        travel.advance(
            scene_dt.as_secs_f32(),
            options.water_mode() == WaterMode::Classic && !water.has_numerical_flow(),
        );
        water.set_world_center(travel.center());
        rain_accum += physics_dt.as_secs_f32() * options.rain;
        for _ in 0..(rain_accum as u32).min(32) {
            add_rain(&mut water, &mut rng, options.drop_size);
        }
        rain_accum = rain_accum.fract();
        // A minimized terminal still receives rain. Advance persistent water and
        // pigment together so restoring the window cannot reveal a frozen burst.
        water.update(physics_dt);
        if width > 0 && height > 0 {
            renderer.set_world_view(travel.center(), water.world_origin(), options.seed);
            renderer.set_quality(selected_quality(options.quality, &adaptive));
            renderer.render_at(&water, elapsed, render_dt.as_secs_f32())?;
            presenter.submit(
                now,
                width,
                height,
                renderer.snapshot_cells(),
                renderer.status(),
                !paused,
            )?;
            if options.quality == QualityMode::Auto && !paused {
                adaptive.observe(work_start.elapsed(), frame_interval);
            }
        }
        dirty = false;
        deadline = next_frame_deadline(deadline, now, Instant::now(), frame_interval);
    }
}

fn main() -> Result<()> {
    let options = Options::parse();
    if options.benchmark || options.snapshot.is_some() {
        return benchmark::run(&options).context("headless render failed");
    }
    run(options).context("terminal animation failed")
}

#[cfg(test)]
mod tests {
    use super::validate_screen_size;

    #[test]
    fn malformed_terminal_sizes_cannot_trigger_unbounded_allocations() {
        assert!(validate_screen_size(u16::MAX, u16::MAX).is_err());
        assert!(validate_screen_size(1000, 500).is_ok());
        assert!(validate_screen_size(0, 0).is_ok());
    }
}
