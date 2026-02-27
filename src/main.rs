use std::io::stdout;
use std::time::Duration;
use crossterm::{cursor, execute, terminal};

use term::Renderer;
use user_input::user_input;
use water::Water;

use crate::settings::SETTINGS;

mod colors;
mod settings;
mod term;
mod user_input;
mod water;
mod waves;

fn main() -> anyhow::Result<()> {
    let mut stdout = stdout();
    let (w, h) = terminal::size()?;

    let mut water = Water::new(w, h);
    let mut renderer = Renderer::new(w, h);

    let mut is_running = true;
    let mut last_iter_time = std::time::Instant::now();

    terminal::enable_raw_mode()?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    let frame_duration = Duration::from_millis(33); // ~30 fps
    let sim_dt = Duration::from_millis(16);          // fixed 60 Hz physics step
    let drop_interval = 1.0 / SETTINGS.drops_frequency;
    let mut sim_accum = Duration::ZERO;
    let mut drop_accum: f32 = 0.0;

    while is_running {
        let frame_start = std::time::Instant::now();

        is_running = user_input(&mut stdout, &mut water, &mut renderer)?;

        let now = std::time::Instant::now();
        let delta = now - last_iter_time;
        last_iter_time = now;

        // Fixed-step simulation: drain accumulator in sim_dt chunks
        sim_accum += delta;
        while sim_accum >= sim_dt {
            sim_accum -= sim_dt;
            water.update(sim_dt);

            drop_accum += sim_dt.as_secs_f32();
            while drop_accum >= drop_interval {
                drop_accum -= drop_interval;
                water.add_random_drop();
            }
        }

        renderer.draw(&mut stdout, &water)?;

        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }

    execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;

    Ok(())
}
