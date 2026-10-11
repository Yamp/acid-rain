#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Capture deterministic before/after animation and indicative frame-change metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DRIVER = r'''
#![allow(dead_code)]
mod colors; mod output; mod scene; mod settings; mod term; mod water; mod waves;
use std::{fs, io::Write, path::PathBuf, time::{Duration, Instant}};
use term::{RenderQuality, Renderer};
use water::{Water, WaveCenter};

fn drop(water: &mut Water, i: u32, age: u64) {
    let x = ((i*37+17)%997) as f32/997.0;
    let y = ((i*71+53)%991) as f32/991.0;
    let strength = 0.10+((i*13+7)%101) as f32/101.0*0.20;
    // CURRENT_DROP
    water.wave_centers.push(WaveCenter::new(
        x, y, strength,
        water.time-Duration::from_secs(age),
    ));
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let destination = PathBuf::from(&args[1]);
    let surface_speed: f32 = args[2].parse()?;
    let scene_speed: f32 = args[3].parse()?;
    let frame_count: u32 = args[4].parse()?;
    let dt = Duration::from_secs_f64(1.0/30.0);
    let scene_dt = dt.mul_f32(scene_speed);
    let physics_dt = scene_dt.mul_f32(surface_speed);
    fs::create_dir_all(&destination)?;
    for quality_name in args[5].split(',') {
        let quality = match quality_name {
            "high" => RenderQuality::High,
            "balanced" => RenderQuality::Balanced,
            "low" => RenderQuality::Low,
            other => anyhow::bail!("unknown quality: {other}"),
        };
        let name = quality.name();
        let folder = destination.join(name);
        fs::create_dir_all(&folder)?;
        let mut stats = fs::File::create(destination.join(format!("{name}.csv")))?;
        writeln!(stats, "frame,elapsed_seconds,mean_abs_channel_delta,fraction_channels_gt32,fraction_pixels_gt32,render_ms,mean_chroma,fraction_saturated_red,fraction_saturated_green")?;
        let mut water = Water::new(160, 50);
        // WATER_INITIALIZATION
        let mut rain = Duration::ZERO;
        let mut next_drop = 10;
        let mut renderer = Renderer::new(160, 50);
        renderer.set_quality(quality);
        // RENDERER_SETTINGS
        let mut previous: Option<Vec<u8>> = None;
        for frame in 0..frame_count {
            let elapsed = if args[6] == "fixed" { 10.0 }
                else { 10.0+frame as f32*scene_dt.as_secs_f32() };
            let begin = Instant::now();
            water.update(physics_dt);
            rain += physics_dt;
            while rain >= Duration::from_millis(125) {
                rain -= Duration::from_millis(125);
                drop(&mut water, next_drop, 0);
                next_drop += 1;
            }
            renderer.draw_at(&mut std::io::sink(), &water, elapsed, dt.as_secs_f32())?;
            let render_ms = begin.elapsed().as_secs_f64()*1000.0;
            let rgb = renderer.snapshot_rgb();
            assert_eq!(rgb.len(), 160*100*3);
            if let Some(prev) = previous.as_ref() {
                let sum: u64 = rgb.iter().zip(prev).map(|(a,b)| a.abs_diff(*b) as u64).sum();
                let channels = rgb.iter().zip(prev).filter(|(a,b)| a.abs_diff(**b)>32).count();
                let pixels = rgb.chunks_exact(3).zip(prev.chunks_exact(3))
                    .filter(|(a,b)| a.iter().zip(*b).any(|(a,b)| a.abs_diff(*b)>32)).count();
                let mut chroma_sum = 0.0;
                let (mut reds, mut greens) = (0, 0);
                for color in rgb.chunks_exact(3) {
                    let (r,g,b) = (color[0] as f64,color[1] as f64,color[2] as f64);
                    let hi = r.max(g).max(b);
                    let chroma = hi-r.min(g).min(b);
                    chroma_sum += chroma;
                    if hi >= 64.0 && chroma/hi >= 0.65 {
                        let hue = if hi == r { ((g-b)/chroma).rem_euclid(6.0)*60.0 }
                            else if hi == g { ((b-r)/chroma+2.0)*60.0 }
                            else { ((r-g)/chroma+4.0)*60.0 };
                        if hue < 30.0 || hue >= 330.0 { reds += 1; }
                        if (75.0..=165.0).contains(&hue) { greens += 1; }
                    }
                }
                let pixel_count = (rgb.len()/3) as f64;
                writeln!(stats, "{frame},{elapsed:.6},{:.6},{:.6},{:.6},{render_ms:.6},{:.6},{:.6},{:.6}",
                    sum as f64/rgb.len() as f64, channels as f64/rgb.len() as f64,
                    pixels as f64/pixel_count, chroma_sum/pixel_count,
                    reds as f64/pixel_count, greens as f64/pixel_count)?;
            }
            if frame%2==0 {
                let mut ppm = fs::File::create(folder.join(format!("frame-{frame:03}.ppm")))?;
                write!(ppm, "P6\n160 100\n255\n")?;
                ppm.write_all(&rgb)?;
            }
            previous = Some(rgb);
        }
        println!("{name}: {frame_count} frames, {} RGB deltas, {} RGB captures, scene speed {scene_speed}, surface speed {surface_speed}", frame_count-1, frame_count.div_ceil(2));
    }
    Ok(())
}
'''


def capture(stage: str, output: Path, archive: Path, args: argparse.Namespace) -> None:
    destination = output / stage
    destination.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix=f"acid-rain-motion-{stage}-") as temporary:
        build = Path(temporary)
        source_archive = archive if stage == "before" else args.after_source
        if source_archive is not None:
            with tarfile.open(source_archive) as source:
                source.extractall(build, filter="data")
        else:
            shutil.copytree(ROOT / "src", build / "src")
            for name in ("Cargo.toml", "Cargo.lock"):
                shutil.copy2(ROOT / name, build / name)
        hashes = {str(p.relative_to(build)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(build.rglob("*.rs"))}
        if stage == "after":
            with tarfile.open(destination / "source.tar.gz", "w:gz") as snapshot:
                for name in ("src", "Cargo.toml", "Cargo.lock"):
                    snapshot.add(build / name, arcname=name)
        optional_modules = "".join(f"mod {name};\n" for name in ("motion", "vortex", "physics", "color_pipeline", "optics", "retina", "rapids", "dye", "stirring", "drain", "pigment_optics")
                                   if (build / f"src/{name}.rs").exists())
        rapids = getattr(args, f"{stage}_rapids")
        drain = getattr(args, f"{stage}_drain")
        if rapids and drain:
            raise ValueError(f"--{stage}-rapids and --{stage}-drain are mutually exclusive")
        intensity = getattr(args, f"{stage}_intensity")
        trails = not getattr(args, f"{stage}_no_trails")
        chronological = rapids or (build / "src/dye.rs").is_file()
        driver = DRIVER.replace("mod colors;", optional_modules+"mod colors;")
        settings = []
        if intensity is not None:
            settings.append(f"renderer.set_intensity({intensity!r}_f32);")
        if not trails:
            settings.append("renderer.set_trails(false);")
        driver = driver.replace("// RENDERER_SETTINGS", "\n        ".join(settings))
        if rapids and not (build / "src/rapids.rs").is_file():
            raise ValueError(f"--{stage}-rapids requires sources with the rapids module")
        if drain and not (build / "src/drain.rs").is_file():
            raise ValueError(f"--{stage}-drain requires sources with the drain module")
        if chronological:
            initialization = """for age in (0..10).rev() {
            water.update(Duration::from_secs(1));
            drop(&mut water, age, 0);
        }"""
            if rapids:
                initialization = "water.set_rapids(true);\n        " + initialization
            elif drain:
                initialization = "water.set_drain(true);\n        " + initialization
            driver = driver.replace("// CURRENT_DROP", "if age == 0 { water.add_drop(x, y, strength); return; }")
        else:
            initialization = """water.update(Duration::from_secs(10));
        for i in 0..10 { drop(&mut water, i, i as u64); }"""
        driver = driver.replace("// WATER_INITIALIZATION", initialization)
        (build / "src/main.rs").write_text(driver)
        manifest = build / "Cargo.toml"
        manifest.write_text(manifest.read_text().replace('edition = "2021"', 'edition = "2024"'))
        environment = os.environ.copy()
        target = output / ".build-target"
        environment["CARGO_TARGET_DIR"] = str(target)
        scene_speed = getattr(args, f"{stage}_scene_speed")
        surface_speed = getattr(args, f"{stage}_surface_speed")
        frame_count = round(args.seconds*30)
        for command in (["cargo", "check"], ["cargo", "build", "--release"],
                        [str(target / "release/acid-rain"), str(destination),
                         str(surface_speed), str(scene_speed), str(frame_count), ",".join(args.qualities),
                         "fixed" if args.fixed_camera else "moving"]):
            result = subprocess.run(command, cwd=build, env=environment, capture_output=True, text=True)
            commands.append({"argv": command, "status": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            (destination / "commands.json").write_text(json.dumps(commands, indent=2)+"\n")
            print(result.stdout or result.stderr[-1000:], flush=True)
            result.check_returncode()
        metadata = {"stage": stage, "cells": [160, 50], "rgb_pixels": [160, 100],
                    "frames": frame_count, "fps": 30, "capture_every_frames": 2,
                    "surface_speed": surface_speed, "scene_speed": scene_speed,
                    "flow": "rapids" if rapids else "drain" if drain else "classic", "fixed_camera": args.fixed_camera,
                    "requested_intensity": intensity,
                    "intensity_setting": "renderer default" if intensity is None else "explicit Renderer::set_intensity input",
                    "trails": trails,
                    "scene": "Journey", "qualities": args.qualities,
                    "source_sha256": hashes, "initial_simulation_seconds": 10,
                    "initial_drops": ("ten chronological one-second advances, each followed by add_drop; indices 9..0, final ages 9..0 seconds"
                                      if chronological else "10 deterministic drops inserted with ages 0..9 seconds after a single 10-second advance"),
                    "chronological_rain": chronological,
                    "ongoing_drop_injection": "Water::add_drop" if chronological else "WaveCenter::new pushed directly",
                    "dye_module": (build / "src/dye.rs").is_file(),
                    "rain_interval_physics_seconds": 0.125,
                    "render_elapsed_seconds": ("10.0 (camera and scene held; physical water evolves)" if args.fixed_camera
                                               else f"10.0+frame*({scene_speed}/30.0), frame=0..{frame_count-1}"),
                    "physics_dt_seconds": scene_speed*surface_speed/30.0,
                    "capture_arguments": {key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()},
                    "rustc": subprocess.check_output(["rustc", "-Vv"], text=True)}
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    # Generated binaries are large and reproducible; artifacts retain source hashes and logs.
    shutil.rmtree(output / ".build-target")


def compose(output: Path) -> None:
    metadata = {stage: json.loads((output / stage / "metadata.json").read_text())
                for stage in ("before", "after")}
    before, after = metadata["before"], metadata["after"]
    clocks_differ = any(before.get(key, 1.0) != after.get(key, 1.0)
                       for key in ("scene_speed", "surface_speed"))
    mode_comparison = (
        "The water modes differ: this compares the requested startup experiences, "
        "not a rendering-only change on identical geometry or pigment flow."
        if before.get("flow", "classic") != after.get("flow", "classic") else
        "Both captures use the same selected water mode."
    )
    clock_comparison = (
        "This is a requested-default comparison, not a matched-clock comparison. "
        "Different scene/physical rates also change camera progress and the number of rain impacts per wall second; "
        "the motion metrics do not isolate the renderer or mixing changes."
        if clocks_differ else
        "Scene and physical clock rates match between the two versions."
    )
    assert before["frames"] == after["frames"] and before["qualities"] == after["qualities"]
    assert before.get("fixed_camera", False) == after.get("fixed_camera", False), "match camera mode before comparing"
    frame_count, fps = before["frames"], before["fps"]
    assert fps == after["fps"] == 30
    captured_frames = list(range(0, frame_count, 2))
    table = ["| Quality | Version | Mean RGB delta (0–255) | Channels >32 | Pixels >32 |",
             "| --- | --- | ---: | ---: | ---: |"]
    colors = ["| Quality | Version | Mean chroma (0–255) | Saturated red pixels | Saturated green pixels |",
              "| --- | --- | ---: | ---: | ---: |"]
    summary = []
    validation = {}
    for quality in before["qualities"]:
        for stage in ("before", "after"):
            with (output / stage / f"{quality}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == frame_count-1
            assert [int(row["frame"]) for row in rows] == list(range(1, frame_count))
            assert all(math.isfinite(float(value)) for row in rows for value in row.values())
            hashes = [hashlib.sha256((output / stage / quality / f"frame-{frame:03}.ppm").read_bytes()).hexdigest()
                      for frame in captured_frames]
            assert len(set(hashes)) == len(hashes)
            validation[f"{stage}/{quality}"] = {"transitions": len(rows), "unique_frames": len(hashes)}
            # Exposure startup is visible in the GIF but excluded from this steady-state table.
            rows = rows[29:]
            keys = ("mean_abs_channel_delta", "fraction_channels_gt32", "fraction_pixels_gt32",
                    "mean_chroma", "fraction_saturated_red", "fraction_saturated_green")
            means = {key: sum(float(row[key]) for row in rows)/len(rows) for key in keys if key in rows[0]}
            summary.append({"quality": quality, "version": stage, "frames": len(rows), **means})
            table.append(f"| {quality} | {stage} | {means['mean_abs_channel_delta']:.3f} | "
                         f"{means['fraction_channels_gt32']:.2%} | {means['fraction_pixels_gt32']:.2%} |")
            if "mean_chroma" in means:
                colors.append(f"| {quality} | {stage} | {means['mean_chroma']:.3f} | "
                              f"{means['fraction_saturated_red']:.2%} | {means['fraction_saturated_green']:.2%} |")
        frames = []
        for frame in captured_frames:
            sheet = Image.new("RGB", (972, 348), (10, 15, 23))
            draw = ImageDraw.Draw(sheet)
            for column, stage in enumerate(("before", "after")):
                x = column*492
                draw.text((x+8, 8), f"{quality.upper()} / {stage.upper()} / {frame/fps:.2f}s", fill=(232, 240, 250))
                path = output / stage / quality / f"frame-{frame:03}.ppm"
                with Image.open(path) as image:
                    assert image.size == (160, 100)
                    # Nearest-neighbor enlargement preserves real half-cell spatial structure.
                    sheet.paste(image.resize((480, 300), Image.Resampling.NEAREST), (x, 36))
            frames.append(sheet)
        # One palette for the entire clip avoids independent quantization flicker.
        contact = Image.new("RGB", (972, 348*len(frames)))
        for i, frame in enumerate(frames):
            contact.paste(frame, (0, i*348))
        palette = contact.quantize(colors=256, dither=Image.Dither.NONE)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        indexed[0].save(output / f"{quality}-before-after.gif", save_all=True,
                        append_images=indexed[1:], duration=[[70, 60, 70][i % 3] for i in range(len(indexed))],
                        loop=0, optimize=False, disposal=2)
        frames[len(frames)//2].save(output / f"{quality}-before-after.png")
        contact = Image.new("RGB", (1304, 744), (10, 15, 23))
        draw = ImageDraw.Draw(contact)
        inspection_frames = [captured_frames[round(i*(len(captured_frames)-1)/5)] for i in range(6)]
        for i, frame in enumerate(inspection_frames):
            x, y = (i % 2)*652, (i // 2)*248
            draw.text((x+8, y+6), f"{quality.upper()} / time {frame/30:.2f}s", fill=(232, 240, 250))
            for column, stage in enumerate(("before", "after")):
                draw.text((x+column*326+8, y+23), stage.upper(), fill=(190, 208, 228))
                with Image.open(output / stage / quality / f"frame-{frame:03}.ppm") as image:
                    contact.paste(image.resize((320, 200), Image.Resampling.NEAREST),
                                  (x+column*326, y+42))
        contact.save(output / f"{quality}-motion-contact.png")
        with Image.open(output / f"{quality}-before-after.gif") as clip:
            assert clip.n_frames == len(captured_frames)
            gif_ms = sum(clip.seek(i) or clip.info["duration"] for i in range(clip.n_frames))
            assert abs(gif_ms-frame_count/fps*1000) <= 70
            validation[f"{quality}_gif"] = {"frames": clip.n_frames, "duration_ms": gif_ms}
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    report = ["# Motion and palette comparison", "", *table, "", *colors, "",
              f"The tables average frames 30–{frame_count-1}, excluding exposure startup. "
              f"All {frame_count-1} changes remain in each raw CSV. A pixel exceeds the motion threshold when any RGB channel changes by more than 32. "
              "Chroma is max(R,G,B) minus min(R,G,B). Saturated means HSV saturation at least 0.65 and value at least 64/255; "
              "red hue is [330,360) or [0,30), green hue is [75,165]. Pixel fractions use all image pixels as the denominator.", "",
              "These metrics are indicative: they combine real movement, water dynamics, camera changes, "
              "temporal reconstruction and aliasing. A smaller value alone does not prove smoother animation or better image quality. "
              "Palette measurements apply to this Journey clip beginning at renderer time 10 seconds, not to every world or future scene transition.", "",
              clock_comparison, "", mode_comparison, "",
              f"The GIFs show {frame_count/fps:g} seconds at 15 captured frames/second; RGB captures are 160×100 virtual pixels "
              "for a 160×50 terminal. Enlargement is nearest-neighbor, with one shared GIF palette. "
              "PPMs preserve the unquantized RGB output. Terminal transport and emulator display are excluded.", "",
              "Both versions start at simulation time 10 seconds. "
              f"Before mode: {before.get('flow', 'classic')}; {before['initial_drops']}. "
              f"After mode: {after.get('flow', 'classic')}; {after['initial_drops']}. "
              f"Intensity input: before {before.get('requested_intensity') if before.get('requested_intensity') is not None else 'renderer default'}, "
              f"after {after.get('requested_intensity') if after.get('requested_intensity') is not None else 'renderer default'}. "
              f"RGB trails: before {'on' if before.get('trails', True) else 'off'}, "
              f"after {'on' if after.get('trails', True) else 'off'}. "
              f"Scene speed: before {before.get('scene_speed', 1):g}, after {after.get('scene_speed', 1):g}. "
              f"Surface scale relative to scene time: before {before['surface_speed']:g}, after {after['surface_speed']:g}. "
              "Physics delta is wall delta × scene speed × surface scale. "
              f"Renderer time: {after['render_elapsed_seconds']}; "
              "temporal-filter delta remains wall delta. New drops arrive each 0.125 physics seconds. "
              "Wall-clock times align; the rendered pictures differ. "
              "Source hashes, exact commands, compiler output and full captures are saved in before/ and after/. "
              "Six time samples in each motion-contact PNG support frame-by-frame visual inspection."]
    (output / "comparison.md").write_text("\n".join(report)+"\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print("\n".join(table+[""]+colors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("before", "after", "compose"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/refinement-motion")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/refinement-before/source.tar.gz")
    parser.add_argument("--after-source", type=Path, help="Reproduce an archived after version instead of current src")
    parser.add_argument("--before-rapids", action="store_true", help="Enable numerical river rapids in the before capture")
    parser.add_argument("--after-rapids", action="store_true", help="Enable numerical river rapids in the after capture")
    parser.add_argument("--before-drain", action="store_true", help="Enable the draining whirlpool in the before capture")
    parser.add_argument("--after-drain", action="store_true", help="Enable the draining whirlpool in the after capture")
    parser.add_argument("--before-intensity", type=float, help="Explicit before intensity input, 0..4; omitted uses that renderer's default")
    parser.add_argument("--after-intensity", type=float, help="Explicit after intensity input, 0..4; omitted uses that renderer's default")
    parser.add_argument("--before-no-trails", action="store_true", help="Disable RGB trails in the before capture")
    parser.add_argument("--after-no-trails", action="store_true", help="Disable RGB trails in the after capture")
    parser.add_argument("--fixed-camera", action="store_true", help="Hold renderer/camera/scene time at 10s while physical water evolves")
    parser.add_argument("--before-scene-speed", type=float, default=1.0)
    parser.add_argument("--before-surface-speed", type=float, default=1.0)
    parser.add_argument("--after-scene-speed", type=float, default=1.0)
    parser.add_argument("--after-surface-speed", type=float, default=0.35)
    parser.add_argument("--qualities", nargs="+", choices=("high", "balanced", "low"), default=["balanced", "low"])
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 4)
           for value in (args.before_intensity, args.after_intensity)):
        parser.error("intensity inputs must be finite numbers in [0, 4]")
    if not math.isfinite(args.seconds) or args.seconds < 2 or any(not 0 < getattr(args, key) <= 3 for key in
                              ("before_scene_speed", "before_surface_speed", "after_scene_speed", "after_surface_speed")):
        parser.error("seconds must be at least 2 and speed scales must be in (0, 3]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "compose":
        compose(output)
    else:
        capture(args.stage, output, args.before_source, args)


if __name__ == "__main__":
    main()
