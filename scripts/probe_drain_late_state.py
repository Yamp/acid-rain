#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pillow>=10"]
# ///
"""Capture real Drain rendering at late physical times from a frozen source archive.

One chronological simulation supplies every checkpoint. Camera/light time and
display exposure are held fixed only in the isolated capture build; production
sources are never edited. This is a state/image diagnostic, not a CPU benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''
#![allow(dead_code, unused_variables)]
// MODULES
use std::{fs, io::Write, path::Path, time::Duration};
use rand::{Rng, SeedableRng, rngs::StdRng};
use scene::SceneKind;
use term::{RenderQuality, Renderer};
use water::Water;

fn checkpoint(water: &Water, output: &Path, number: usize, drops: u64,
              width: u16, height: u16, settle: u32, summary: &mut fs::File) -> anyhow::Result<()> {
    let name = format!("checkpoint-{number:02}");
    let mut fields = fs::File::create(output.join(format!("{name}-fields.csv")))?;
    writeln!(fields, "x,y,u,v,height_m,depth_m,foam,velocity_x_mps,velocity_y_mps,cyan_P_per_m2,magenta_P_per_m2,amber_P_per_m2")?;
    let (mut low, mut high, mut min_depth, mut max_speed, mut foam_sum) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, 0.0_f32, 0.0_f64);
    let mut pigment = [0.0_f64; 3];
    for x in 0..usize::from(width) { for y in 0..usize::from(height) {
        let i = x * usize::from(height) + y;
        let z = water.levels[(x,y)]; let depth = water.depths[i];
        let foam = water.foam[i]; let [u,v] = water.surface_velocity[i];
        let dye = water.dye[i];
        anyhow::ensure!([z,depth,foam,u,v,dye[0],dye[1],dye[2]].iter().all(|v| v.is_finite()), "nonfinite field at {x},{y}");
        anyhow::ensure!(depth > 0.0 && (0.0..=1.0).contains(&foam) && dye.iter().all(|v| *v >= 0.0), "invalid material field at {x},{y}");
        low=low.min(z); high=high.max(z); min_depth=min_depth.min(depth);
        max_speed=max_speed.max(u.hypot(v)); foam_sum+=f64::from(foam);
        for c in 0..3 {pigment[c]+=f64::from(dye[c]);}
        writeln!(fields,"{x},{y},{:.9e},{:.9e},{z:.9e},{depth:.9e},{foam:.9e},{u:.9e},{v:.9e},{:.9e},{:.9e},{:.9e}",
            x as f64/f64::from(width), y as f64/f64::from(height), dye[0],dye[1],dye[2])?;
    }}
    let cells=f64::from(width)*f64::from(height);
    let area=f64::from(crate::physics::PATCH_LENGTH).powi(2)/cells;
    let sampled_mass: f64=pigment.into_iter().sum::<f64>()*area;
    let mut renderer=Renderer::new(width,height);
    renderer.set_scene(SceneKind::Acid);
    renderer.set_quality(RenderQuality::Balanced);
    renderer.set_intensity(2.0);
    // Only display histories settle: no physics, rain or camera advances here.
    for _ in 0..settle {renderer.draw_at(&mut std::io::sink(),water,12.0,1.0/60.0)?;}
    let rgb=renderer.snapshot_rgb();
    anyhow::ensure!(rgb.len()==usize::from(width)*usize::from(height)*2*3, "unexpected RGB shape");
    let mut ppm=fs::File::create(output.join(format!("{name}.ppm")))?;
    write!(ppm,"P6\n{width} {}\n255\n",height*2)?;ppm.write_all(&rgb)?;
    let seconds=water.elapsed().as_secs_f64();
    writeln!(summary,"{number},{seconds:.9},{:.9},{drops},{low:.9e},{high:.9e},{min_depth:.9e},{max_speed:.9e},{:.9e},{sampled_mass:.12e}",
        seconds/0.75,foam_sum/cells)?;
    summary.flush()?;
    println!("checkpoint {number}: {seconds:.3} physical s, {drops} drops, height [{low:.6},{high:.6}] m, min depth {min_depth:.6} m");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_>=std::env::args().collect();
    let output=Path::new(&args[1]);
    let width: u16=args[2].parse()?; let height: u16=args[3].parse()?;
    let settle: u32=args[4].parse()?; let seed: u64=args[5].parse()?;
    let checkpoints: Vec<Duration>=args[6].split(',').map(|v| v.parse::<f64>().map(Duration::from_secs_f64)).collect::<Result<_,_>>()?;
    fs::create_dir_all(output)?;
    let mut water=Water::new(width,height); water.set_drain(true); water.set_bleach(1.0);
    let mut rng=StdRng::seed_from_u64(seed);
    let mut now=Duration::ZERO; let mut drops=0_u64;
    let mut summary=fs::File::create(output.join("states.csv"))?;
    writeln!(summary,"checkpoint,physical_seconds,equivalent_wall_seconds_at_speed3,drops,height_min_m,height_max_m,depth_min_m,maximum_sampled_speed_mps,mean_foam,sampled_pigment_mass_P")?;
    for (number,target) in checkpoints.into_iter().enumerate() {
        loop {
            // Exact 6/s schedule rounded to nanoseconds; no display-cadence quantization.
            let event=Duration::from_nanos(((u128::from(drops)+1)*1_000_000_000/6) as u64);
            if event>target {break;}
            water.update(event-now);now=event;
            // Same distribution and RNG order as interactive add_rain, at size 1.
            let strength=0.08+rng.r#gen::<f32>().powi(2)*0.28;
            let (x,y)=(rng.r#gen(),rng.r#gen());
            water.add_drop_sized(x,y,strength,1.0);drops+=1;
        }
        water.update(target-now);now=target;
        // Zero-time update also resolves a deposit exactly on the checkpoint.
        water.update(Duration::ZERO);
        checkpoint(&water,output,number,drops,width,height,settle,&mut summary)?;
    }
    Ok(())
}
'''


def dimensions(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.lower().split("x"))
        if 1 <= width <= 1000 and 1 <= height <= 1000:
            return width, height
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, each within 1..1000")


def compose(output: Path) -> None:
    with (output / "states.csv").open() as handle:
        states = list(csv.DictReader(handle))
    decode = lambda value: value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    linear = [decode(value / 255.0) for value in range(256)]
    metrics = []
    sheet = Image.new("RGB", (976, 344 * math.ceil(len(states) / 2)), (12, 15, 23))
    draw = ImageDraw.Draw(sheet)
    table = ["| Physical / equivalent wall seconds | Mean linear Y | Dark Y < 0.01 | Min depth | Height range |",
             "|---|---:|---:|---:|---:|"]
    for number, state in enumerate(states):
        with Image.open(output / f"checkpoint-{number:02}.ppm") as picture:
            pixels = list(picture.convert("RGB").getdata())
            luminance = [0.2126 * linear[r] + 0.7152 * linear[g] + 0.0722 * linear[b]
                         for r, g, b in pixels]
            mean = sum(luminance) / len(luminance)
            dark = sum(value < 0.01 for value in luminance) / len(luminance)
            metrics.append({**state, "mean_linear_Y": mean, "dark_Y_lt_0_01_fraction": dark})
            seconds = float(state["physical_seconds"])
            wall = float(state["equivalent_wall_seconds_at_speed3"])
            low, high = float(state["height_min_m"]), float(state["height_max_m"])
            table.append(f"| {seconds:g} / {wall:g} | {mean:.6f} | {dark:.2%} | "
                         f"{float(state['depth_min_m']):.5f} m | {low:.5f}…{high:.5f} m |")
            x, y = number % 2 * 488, number // 2 * 344
            draw.text((x + 8, y + 8), f"Drain / {seconds:g} physical s / camera12 / exposure1", fill="white")
            scale = min(480 / picture.width, 300 / picture.height)
            enlarged = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (x + (480 - enlarged.width) // 2, y + 34))
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    sheet.save(output / "late-state-contact.png")
    report = ["# Actual Drain rendering at late physical times", "", *table, "",
              "One chronological Drain simulation supplies all checkpoints. Rain is deterministic at "
              "6 impacts per physical second with the interactive strength distribution, diameter 1, bleach 1 "
              "and seed recorded in metadata. At speed 3 and surface scale 0.25, one physical second equals "
              "4/3 wall seconds. Events occur at exact scheduled timestamps rather than display updates.", "",
              "Each frozen Water state is rendered independently with a fresh Balanced renderer, Acid camera/light "
              "time 12 and intensity 2. Display histories settle without advancing Water. The isolated capture copy "
              "holds exposure at exactly 1; this prevents exposure or camera changes from hiding changes in depth "
              "and color. Production sources are not edited. These diagnostic stills do not demonstrate animation.", "",
              "The PPMs are direct Renderer::snapshot_rgb output. The contact sheet only labels and enlarges those pixels. "
              "Mean Y decodes sRGB then applies Rec.709 luminance weights. Sampled field CSVs retain height, actual "
              "column depth, foam, velocity and all three pigment species; their integrated pigment mass is a sampled "
              "display-grid estimate, not the internal solver's conservation ledger. Source archives, original and "
              "compiled hashes, driver, commands and raw metrics are retained. This is not a timing benchmark."]
    (output / "comparison.md").write_text("\n".join(report) + "\n")
    print("\n".join(table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen final source.tar.gz")
    parser.add_argument("--expect-gates", type=Path, help="Require source hashes to match this gates.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/drain-late-state")
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[1, 10, 60, 180])
    parser.add_argument("--size", type=dimensions, default=(160, 50))
    parser.add_argument("--settle-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if (not args.checkpoints or any(not math.isfinite(t) or not 0 < t <= 3600 for t in args.checkpoints)
            or sorted(set(args.checkpoints)) != args.checkpoints):
        parser.error("checkpoints must be increasing, unique finite physical seconds in (0, 3600]")
    if not 1 <= args.settle_frames <= 600 or not 0 <= args.seed < 2**64:
        parser.error("settle-frames must be 1..600 and seed an unsigned 64-bit integer")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix="acid-rain-drain-late-") as temporary:
        build = Path(temporary)
        with tarfile.open(source) as archive:
            archive.extractall(build, filter="data")
        manifest = tomllib.loads((build / "Cargo.toml").read_text())
        assert manifest["package"]["edition"] == "2024", "diagnostic requires Rust edition 2024"
        assert (build / "src/drain.rs").is_file(), "archive has no Drain model"
        hashes = {str(path.relative_to(build)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((build / "src").glob("*.rs"))}
        cargo_hashes = {name: hashlib.sha256((build / name).read_bytes()).hexdigest()
                        for name in ("Cargo.toml", "Cargo.lock")}
        if args.expect_gates:
            gates = json.loads(args.expect_gates.read_text())
            assert hashes == gates["source_sha256"], "archive differs from requested gate sources"
            assert cargo_hashes == gates["cargo_files_sha256"], "archive Cargo files differ from gates"
        shutil.copy2(source, output / "source.tar.gz")
        term = build / "src/term.rs"
        original = "self.adapted_exposure = adapt_exposure(self.adapted_exposure, target, dt);"
        text = term.read_text()
        assert text.count(original) == 1, "exposure capture hook changed; inspect before running"
        term.write_text(text.replace(original, "self.adapted_exposure = 1.0;"))
        # The production benchmark calls the production entry point's UI mode
        # configurator; this independent Water/Renderer driver does not use it.
        modules = "\n".join(module for module in re.findall(
            r"^mod \w+;", (build / "src/main.rs").read_text(), re.MULTILINE
        ) if module != "mod benchmark;")
        driver = DRIVER.replace("// MODULES", modules)
        (build / "src/main.rs").write_text(driver)
        (output / "driver.rs").write_text(driver)
        compiled = {str(path.relative_to(build)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((build / "src").glob("*.rs"))}
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(build / "target")
        metadata = {"source_archive": str(source), "source_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "source_sha256": hashes, "cargo_files_sha256": cargo_hashes, "compiled_source_sha256": compiled,
                    "capture_only_changes": ["replace main with retained driver", "hold display exposure at 1 in term.rs"],
                    "checkpoints_physical_seconds": args.checkpoints, "scene_speed": 3, "surface_scale": 0.25,
                    "camera_light_elapsed": 12, "scene": "Acid", "mode": "Drain", "quality": "Balanced",
                    "rain_per_physical_second": 6, "drop_size": 1, "bleach": 1, "intensity": 2,
                    "seed": args.seed, "size": args.size, "settle_frames": args.settle_frames,
                    "display_dt": 1 / 60, "exposure": 1, "chronological_single_run": True,
                    "rustflags": environment.get("RUSTFLAGS", ""),
                    "rustc": subprocess.check_output(["rustc", "-Vv"], text=True)}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for command in (["cargo", "check", "--locked"], ["cargo", "build", "--locked", "--release"],
                        [str(build / "target/release/acid-rain"), str(output), str(args.size[0]), str(args.size[1]),
                         str(args.settle_frames), str(args.seed), ",".join(map(str, args.checkpoints))]):
            result = subprocess.run(command, cwd=build, env=environment, capture_output=True, text=True)
            commands.append({"argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            print(result.stdout or result.stderr[-2000:], flush=True)
            result.check_returncode()
    compose(output)


if __name__ == "__main__":
    main()
