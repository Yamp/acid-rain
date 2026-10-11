#![recursion_limit = "256"]
#![cfg_attr(all(target_os = "windows", not(test)), windows_subsystem = "windows")]
mod app;
mod cli;
mod ui;

use std::{
    path::Path,
    time::{
        Duration,
        Instant,
    },
};

use acid_rain_desktop::gpu::{
    GpuConfig,
    GpuRenderer,
    RenderInput,
};
use clap::Parser;
use cli::Options;

fn main() -> anyhow::Result<()> {
    #[cfg(target_os = "windows")]
    windows_launch::attach_existing_console();
    acid_rain_desktop::gpu::init_diagnostics_logging();
    let options = Options::parse();
    #[cfg(target_os = "windows")]
    let interactive = !options.headless;
    let result = if options.headless {
        headless(&options)
    } else {
        app::run(options)
    };
    #[cfg(target_os = "windows")]
    if interactive && let Err(error) = &result {
        windows_launch::show_error(error);
    }
    result
}

#[cfg(target_os = "windows")]
mod windows_launch {
    use windows_sys::Win32::{
        Foundation::INVALID_HANDLE_VALUE,
        System::Console::{
            ATTACH_PARENT_PROCESS,
            AttachConsole,
            GetStdHandle,
            STD_ERROR_HANDLE,
            STD_OUTPUT_HANDLE,
        },
        UI::WindowsAndMessaging::{
            MB_ICONERROR,
            MB_OK,
            MessageBoxW,
        },
    };

    pub fn attach_existing_console() {
        // A GUI-subsystem executable does not allocate a blank terminal on
        // double-click. Shells/CI may already supply redirected pipes or files:
        // leave either inherited output handle untouched. Otherwise attach the
        // parent's existing console so --help and headless output remain useful.
        // Explorer has no console, in which case AttachConsole simply fails.
        unsafe {
            let stdout = GetStdHandle(STD_OUTPUT_HANDLE);
            let stderr = GetStdHandle(STD_ERROR_HANDLE);
            let valid = |handle: windows_sys::Win32::Foundation::HANDLE| {
                !handle.is_null() && handle != INVALID_HANDLE_VALUE
            };
            if !valid(stdout) && !valid(stderr) {
                let _ = AttachConsole(ATTACH_PARENT_PROCESS);
            }
        }
    }

    pub fn show_error(error: &anyhow::Error) {
        let title: Vec<u16> = "Acid Rain".encode_utf16().chain(Some(0)).collect();
        let message: Vec<u16> = format!(
            "Acid Rain could not continue.\n\n{error:#}\n\nUse --help from a terminal to see \
             launch options."
        )
        .replace('\0', " ")
        .encode_utf16()
        .chain(Some(0))
        .collect();
        // Both UTF-16 strings are NUL-terminated and live through this synchronous
        // call. Headless/CI failures never display a modal dialog.
        unsafe {
            let _ = MessageBoxW(
                std::ptr::null_mut(),
                message.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR,
            );
        }
    }
}
fn create_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
fn headless(o: &Options) -> anyhow::Result<()> {
    let mut sim = o.simulation()?;
    let mut gpu = pollster::block_on(GpuRenderer::new(None, GpuConfig {
        width: o.size[0],
        height: o.size[1],
        backends: o.backend.backends(),
        force_fallback_adapter: o.software,
    }))?;
    let total_frames = o.warmup + o.frames;
    let mut measured = Vec::with_capacity(o.frames as usize);
    let mut simulation_ms = Vec::with_capacity(o.frames as usize);
    let mut gpu_timings = Vec::with_capacity(o.frames as usize);
    let mut steady_start = None;
    let start = Instant::now();
    let mut outcome = None;
    let step = Duration::from_secs_f64(1. / 60.);
    for frame in 0..total_frames {
        if frame == o.warmup {
            // Warm-up work must finish before timing the steady run. Otherwise
            // its queued GPU work is charged to the first measured calls.
            gpu.wait_idle()?;
            gpu.take_timings();
            steady_start = Some(Instant::now());
        }
        let simulation_start = Instant::now();
        sim.advance(step)?;
        let snapshot = sim.snapshot();
        if frame >= o.warmup {
            simulation_ms.push(simulation_start.elapsed().as_secs_f64() * 1000.);
        }
        let input = RenderInput {
            delta: step.as_secs_f64(),
            now: f64::from(frame + 1) / 60.,
            sample_age: 0.,
            predict_camera: false,
            held: false,
            quality: o.quality.render(),
            interpolation: !o.no_interpolation,
            cloud_study: o.cloud_study || o.drop_demo,
            show_falling_drops: o.show_drops,
            subjective_intensity: o.lsd,
            backing_scale: 1.,
        };
        let frame_start = Instant::now();
        let result = gpu.render(&snapshot, &input, |_, _, _, _, _, _| {})?;
        anyhow::ensure!(result.submitted, "headless frame {frame} was not submitted");
        if frame >= o.warmup {
            measured.push(frame_start.elapsed().as_secs_f64() * 1000.);
            gpu_timings.extend(gpu.take_timings());
        }
        outcome = Some(result);
    }
    // Explicit offline synchronization only. The interactive path never reads pixels back.
    gpu.wait_idle()?;
    let steady = steady_start.expect("validated frame count is positive");
    let steady_completed_ms = steady.elapsed().as_secs_f64() * 1000.;
    let warmup_completed_ms = steady.duration_since(start).as_secs_f64() * 1000.;
    gpu_timings.extend(gpu.take_timings());
    gpu_timings.sort_by_key(|timing| timing.sequence);
    let readback_start = Instant::now();
    let pixels = gpu.capture_rgba()?;
    let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.;
    let completed_ms = start.elapsed().as_secs_f64() * 1000.;
    let outcome = outcome.expect("validated frame count is positive");
    let width = o.size[0];
    let height = o.size[1];
    anyhow::ensure!(
        pixels.len() == width as usize * height as usize * 4,
        "capture dimensions do not match the requested size"
    );
    let image = image_metrics(&pixels);
    anyhow::ensure!(
        image["nonblack_pixels"].as_u64().unwrap_or(0) > 0,
        "rendered image is entirely black"
    );
    if let Some(path) = &o.output {
        create_parent(path)?;
        image::save_buffer_with_format(
            path,
            &pixels,
            width,
            height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
    }
    let (mean, p95) = timing_stats(&measured).expect("measured frames exist");
    let gpu_ms: Vec<_> = gpu_timings
        .iter()
        .filter_map(|timing| timing.gpu_ms)
        .collect();
    let gpu_stats = timing_stats(&gpu_ms);
    let completion_ms: Vec<_> = gpu_timings
        .iter()
        .map(|timing| timing.completion_ms)
        .collect();
    let completion_stats = timing_stats(&completion_ms);
    let simulation_stats = timing_stats(&simulation_ms).expect("measured frames exist");
    let timing_records: Vec<_> = gpu_timings
        .iter()
        .map(|timing| {
            serde_json::json!({
                "sequence": timing.sequence, "gpu_ms": timing.gpu_ms,
                "submit_to_completion_ms": timing.completion_ms, "queue_depth": timing.queue_depth,
            })
        })
        .collect();
    let final_state = sim.state();
    let report = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"), "adapter": gpu.adapter_info.name,
        "backend": format!("{:?}", gpu.adapter_info.backend), "device_type": format!("{:?}", gpu.adapter_info.device_type),
        "software_requested": o.software, "hardware_filtering": gpu.hardware_filtering,
        "width": width, "height": height, "render_width": outcome.width, "render_height": outcome.height,
        "frames": o.frames, "warmup": o.warmup, "seed": o.seed, "quality": o.quality.render().name(),
        "speed": o.speed, "trip": o.trip, "rgb_shift": o.rgb_shift, "lsd": o.lsd,
        "scene": ui::scene_name(final_state.scene), "mode": ui::mode_name(final_state.mode),
        "rain": final_state.rain, "drop_size": final_state.drop_size, "bleach": final_state.bleach,
        "effects": final_state.effects, "travel": final_state.travel, "current": final_state.current,
        "cloud_study": o.cloud_study || o.drop_demo, "physical_time": final_state.physical_time,
        "show_drops": o.show_drops, "fixed_step_seconds": step.as_secs_f64(),
        "render_call_mean_ms": mean, "render_call_p95_ms": p95,
        "simulation_and_snapshot_mean_ms": simulation_stats.0, "simulation_and_snapshot_p95_ms": simulation_stats.1,
        "timestamp_queries": gpu.timestamp_queries, "gpu_timed_frames": gpu_ms.len(),
        "gpu_mean_ms": gpu_stats.map(|stats| stats.0), "gpu_p95_ms": gpu_stats.map(|stats| stats.1),
        "submit_to_completion_mean_ms": completion_stats.map(|stats| stats.0),
        "submit_to_completion_p95_ms": completion_stats.map(|stats| stats.1),
        "warmup_completed_ms": warmup_completed_ms,
        "steady_completed_ms": steady_completed_ms,
        "steady_completed_mean_frame_ms": steady_completed_ms / f64::from(o.frames),
        "capture_readback_ms": readback_ms,
        "total_completed_ms_including_warmup_and_readback": completed_ms,
        "timing_note": "Steady wall time includes simulation, snapshot, render calls and final GPU drain; warm-up is fully drained first. Render-call durations include queue backpressure and exclude simulation. GPU intervals bracket the dependent scene passes through final post-processing and exclude the UI overlay, using actual timestamp queries when available. Completion latency includes queued work and callback service. Capture readback is excluded from steady timing.",
        "gpu_timings": timing_records,
        "image": image,
    });
    if let Some(path) = &o.report {
        create_parent(path)?;
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    let gpu_mean =
        gpu_stats.map_or_else(|| "unavailable".into(), |stats| format!("{:.3}", stats.0));
    let gpu_p95 = gpu_stats.map_or_else(|| "unavailable".into(), |stats| format!("{:.3}", stats.1));
    println!(
        "| Adapter | Backend | Size | Frames | Steady wall ms/frame | Render call mean ms | \
         Render call p95 ms | GPU mean ms | GPU p95 ms \
         |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n| {} | {:?} | {}×{} | {} | {:.3} | \
         {:.3} | {:.3} | {} | {} |",
        gpu.adapter_info.name,
        gpu.adapter_info.backend,
        width,
        height,
        o.frames,
        steady_completed_ms / f64::from(o.frames),
        mean,
        p95,
        gpu_mean,
        gpu_p95
    );
    if let Some(path) = &o.output {
        println!("Image: {}", path.display());
    }
    if let Some(path) = &o.report {
        println!("Report: {}", path.display());
    }
    Ok(())
}
fn timing_stats(samples: &[f64]) -> Option<(f64, f64)> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p95 = sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize];
    Some((mean, p95))
}
fn image_metrics(rgba: &[u8]) -> serde_json::Value {
    let mut nonblack = 0u64;
    let mut sum = 0.;
    let mut squared = 0.;
    let mut chroma = 0.;
    for p in rgba.as_chunks::<4>().0 {
        if p[0] > 0 || p[1] > 0 || p[2] > 0 {
            nonblack += 1;
        }
        let l = 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]);
        sum += l;
        squared += l * l;
        chroma += f64::from(p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]));
    }
    let n = (rgba.len() / 4).max(1) as f64;
    let mean = sum / n;
    serde_json::json!({"nonblack_pixels": nonblack, "mean_luma_255": mean,
        "luma_stddev_255": (squared / n - mean * mean).max(0.).sqrt(), "mean_chroma_255": chroma / n})
}
#[cfg(test)]
mod tests {
    #[test]
    fn missing_gpu_samples_stay_missing_and_stats_use_all_samples() {
        assert_eq!(super::timing_stats(&[]), None);
        assert_eq!(super::timing_stats(&[3., 1., 2.]), Some((2., 3.)));
        assert_eq!(super::timing_stats(&[7.]), Some((7., 7.)));
    }
    #[test]
    fn image_evidence_distinguishes_black_and_colour() {
        let black = super::image_metrics(&[0, 0, 0, 255]);
        assert_eq!(black["nonblack_pixels"], 0);
        let mixed = super::image_metrics(&[255, 0, 0, 255, 0, 255, 0, 255]);
        assert_eq!(mixed["nonblack_pixels"], 2);
        assert_eq!(mixed["mean_chroma_255"], 255.);
        assert!(mixed["luma_stddev_255"].as_f64().unwrap() > 0.);
    }
}
