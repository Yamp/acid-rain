use std::{
    fmt::Write as FmtWrite,
    fs,
    io::{
        self,
        Write,
    },
    time::{
        Duration,
        Instant,
    },
};

use anyhow::{
    Context,
    Result,
};

use crate::{
    cli::{
        Options,
        QualityMode,
    },
    term::{
        RenderQuality,
        Renderer,
    },
    water::Water,
};

const WARMUP: u32 = 30;

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(
        &mut self,
        buf: &[u8],
    ) -> io::Result<usize> {
        self.bytes += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

fn drop_parameters(index: u32) -> (f32, f32, f32) {
    (
        ((index * 37 + 17) % 997) as f32 / 997.0,
        ((index * 71 + 53) % 991) as f32 / 991.0,
        0.10 + ((index * 13 + 7) % 101) as f32 / 101.0 * 0.20,
    )
}

pub fn run(options: &Options) -> Result<()> {
    let (width, height) = options.size;
    // Deliberately uncapped: matches the baseline grid, including the largest case.
    let mut water = Water::new(width, height);
    crate::configure_water(&mut water, options.water_mode());
    water.set_bleach(options.bleach);
    water.configure_world(options.seed, true);
    water.set_current(!options.no_current);
    let mut travel = crate::world::WorldTravel::default();
    if options.no_travel {
        travel.toggle();
    }
    // Pigment has history in every mode: replay sources chronologically.
    for age in (0..10).rev() {
        water.update(Duration::from_secs(1));
        let (x, y, strength) = drop_parameters(age);
        water.add_drop_sized(x, y, strength, options.drop_size);
    }
    let mut renderer = Renderer::new(width, height);
    renderer.set_scene(options.scene);
    renderer.set_intensity(options.intensity);
    renderer.set_rgb_shift(options.rgb_shift);
    renderer.set_subjective_intensity(options.subjective_intensity);
    // Fixed quality keeps repeated headless runs deterministic; auto uses Balanced.
    let quality = match options.quality {
        QualityMode::High => RenderQuality::High,
        QualityMode::Low => RenderQuality::Low,
        _ => RenderQuality::Balanced,
    };
    renderer.set_quality(quality);
    let dt = Duration::from_secs_f64(1.0 / 60.0);
    let scene_dt = dt.mul_f32(options.speed);
    let mut drop_accum = Duration::ZERO;
    let mut next_drop = 10;
    let mut writer = CountingWriter::default();
    let mut samples = Vec::with_capacity(options.frames as usize);
    let mut raw = String::from("frame,elapsed_seconds,work_ms,bytes\n");
    let mut bytes = 0_u64;
    for frame in 0..WARMUP + options.frames {
        writer.bytes = 0;
        let started = Instant::now();
        let physics_dt = crate::motion::surface_delta(scene_dt);
        travel.advance(
            scene_dt.as_secs_f32(),
            options.water_mode() == crate::cli::WaterMode::Classic && !water.has_numerical_flow(),
        );
        water.set_world_center(travel.center());
        water.update(physics_dt);
        drop_accum += physics_dt;
        while drop_accum >= Duration::from_millis(125) {
            drop_accum -= Duration::from_millis(125);
            let (x, y, strength) = drop_parameters(next_drop);
            water.add_drop_sized(x, y, strength, options.drop_size);
            next_drop += 1;
        }
        let elapsed = 10.0 + (frame + 1) as f32 * scene_dt.as_secs_f32();
        renderer.set_world_view(travel.center(), water.world_origin(), options.seed);
        renderer.draw_at(&mut writer, &water, elapsed, dt.as_secs_f32())?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        if frame >= WARMUP {
            samples.push(ms);
            bytes += writer.bytes as u64;
            writeln!(
                raw,
                "{},{elapsed:.6},{ms:.6},{}",
                frame - WARMUP,
                writer.bytes
            )?;
        }
    }
    if let Some(path) = &options.snapshot {
        let rgb = renderer.snapshot_rgb();
        let mut file =
            fs::File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        write!(file, "P6\n{} {}\n255\n", width, usize::from(height) * 2)?;
        file.write_all(&rgb)?;
        eprintln!(
            "Saved {} ({}×{} virtual pixels)",
            path.display(),
            width,
            usize::from(height) * 2
        );
    }
    if options.benchmark {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        samples.sort_by(f64::total_cmp);
        let p50 = samples[(samples.len() - 1) * 50 / 100];
        let p95 = samples[(samples.len() - 1) * 95 / 100];
        let report = format!(
            "width,height,quality,scene,flow,warmup_frames,measured_frames,mean_ms,p50_ms,p95_ms,\
             bytes_per_frame\n{width},{height},{},{},{},{WARMUP},{},{mean:.6},{p50:.6},{p95:.6},{:\
             .1}\n",
            quality.name(),
            options.scene.name(),
            options.water_mode().name(),
            options.frames,
            bytes as f64 / samples.len() as f64,
        );
        print!("{report}");
        if let Some(path) = &options.report {
            fs::write(path, &report).with_context(|| format!("cannot write {}", path.display()))?;
            let samples_path = path.with_extension("samples.csv");
            fs::write(&samples_path, raw)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{
        SystemTime,
        UNIX_EPOCH,
    };

    use clap::Parser;

    use super::*;

    #[test]
    fn headless_measurement_uses_selected_speed() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("acid-rain-speed-{}-{nonce}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let report = directory.join("speed.csv");
        let options = Options::parse_from([
            "acid-rain",
            "--benchmark",
            "--size",
            "1x1",
            "--frames",
            "1",
            "--speed",
            "0.5",
            "--water",
            "classic",
            "--report",
            report.to_str().unwrap(),
        ]);
        run(&options).unwrap();
        let samples = fs::read_to_string(report.with_extension("samples.csv")).unwrap();
        fs::remove_dir_all(&directory).unwrap();
        let elapsed: f32 = samples
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let expected = 10.0 + (WARMUP + 1) as f32 * 0.5 / 60.0;
        assert!(
            (elapsed - expected).abs() < 2e-6,
            "selected 0.5x: expected {expected}, got {elapsed}"
        );
    }
}
