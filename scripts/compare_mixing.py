#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pillow>=10"]
# ///
"""Compare actual before/after pigment fields, without the terminal renderer.

The generated Rust driver evolves Water in identical 8 ms steps. It exports
Water.dye directly; Python shows its neutral Beer-Lambert transmission. The
diagnostic images are deliberately separate from the real rendered appearance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODES = ("classic", "rapids")
CHECKPOINTS = ((0, 0.0), (63, 0.5), (125, 1.0), (250, 2.0), (500, 4.0), (1000, 8.0))
FRAME_TICKS = tuple(range(0, 1001, 8))
EXTINCTION = ((2.0, 0.30, 0.25), (0.65, 1.80, 0.65), (0.30, 0.45, 1.80))
DRIVER = r'''
#![allow(dead_code)]
mod dye; mod physics; mod rapids; mod settings; mod vortex; mod water; mod waves;
// OPTIONAL_MODULES
use std::{fs, io::Write, path::Path, time::Duration};
use water::Water;

const CHECKPOINTS: [(u32, f64); 6] = [(0,0.0),(63,0.5),(125,1.0),(250,2.0),(500,4.0),(1000,8.0)];

fn export(water: &Water, folder: &Path, tick: u32) -> anyhow::Result<()> {
    let mut file = fs::File::create(folder.join(format!("field-{:06}.f32", tick*8)))?;
    for cell in &water.dye {
        for species in cell {
            anyhow::ensure!(species.is_finite() && *species >= 0.0, "invalid pigment: {species}");
            file.write_all(&species.to_le_bytes())?;
        }
    }
    Ok(())
}

fn checkpoint(water: &Water, report: &mut fs::File, tick: u32, requested: f64, rapids: bool) -> anyhow::Result<()> {
    let (width,height) = (usize::from(water.width()),usize::from(water.height()));
    let cells = width*height;
    let (dx,dy) = (2.0/width as f64,2.0/height as f64);
    let mut mass = [0.0_f64;3];
    let mut peak_total = 0.0_f64;
    let mut gradient_squared = 0.0_f64;
    for x in 0..width {
        for y in 0..height {
            let cell = water.dye[x*height+y];
            let nx = (x+1)%width;
            let ny = (y+1)%height;
            peak_total = peak_total.max(cell.iter().map(|v| f64::from(*v)).sum());
            for species in 0..3 {
                let value = f64::from(cell[species]);
                mass[species] += value*dx*dy;
                let gx = if rapids && x+1 == width {0.0}
                    else {(f64::from(water.dye[nx*height+y][species])-value)/dx};
                let gy = (f64::from(water.dye[x*height+ny][species])-value)/dy;
                gradient_squared += gx*gx+gy*gy;
            }
        }
    }
    let total: f64 = mass.iter().sum();
    let gradient_rms = (gradient_squared/cells as f64).sqrt();
    writeln!(report,"{requested:.3},{:.3},{:.12},{:.12},{:.12},{total:.12},{peak_total:.9},{gradient_rms:.9},{:.9}",
        f64::from(tick)*0.008,mass[0],mass[1],mass[2],gradient_rms/total.max(1e-30))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let destination = Path::new(&args[1]);
    let width: u16 = args[2].parse()?;
    let height: u16 = args[3].parse()?;
    let size: f32 = args[4].parse()?;
    let dt = Duration::from_millis(8);
    for rapids in [false,true] {
        let mode = if rapids {"rapids"} else {"classic"};
        let folder = destination.join(mode);
        fs::create_dir_all(&folder)?;
        let mut water = Water::new(width,height);
        water.set_rapids(rapids);
        // Mature the underlying water without pigment, using the same integration steps.
        for _ in 0..1250 { water.update(dt); }
        for [x,y] in [[0.43,0.46],[0.58,0.50],[0.55,0.62]] {
            water.add_drop_sized(x,y,0.30,size);
        }
        water.update(Duration::ZERO); // Resolve the newly deposited public material field.
        let mut report = fs::File::create(folder.join("checkpoints.csv"))?;
        writeln!(report,"requested_seconds,physical_seconds,sampled_mass_cyan,sampled_mass_magenta,sampled_mass_amber,sampled_total_mass,peak_total_concentration,gradient_rms,gradient_per_mass")?;
        for tick in 0..=1000 {
            if tick>0 { water.update(dt); }
            let saved_checkpoint = CHECKPOINTS.iter().find(|(at,_)| *at==tick);
            if tick%8==0 || saved_checkpoint.is_some() { export(&water,&folder,tick)?; }
            if let Some((_,requested)) = saved_checkpoint {
                checkpoint(&water,&mut report,tick,*requested,rapids)?;
            }
        }
        println!("{mode}: 10 s warmup + 8 s pigment evolution, 8 ms steps; no ongoing rain");
    }
    Ok(())
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pigment_loss_model(source: str) -> dict:
    legacy = re.search(r"const DILUTION_TIME: f32 = ([0-9.]+);", source)
    if legacy:
        return {"kind": "legacy visible-film exponential sink", "e_folding_seconds": float(legacy.group(1))}
    default = re.search(r"const DEFAULT_BLEACH: f32 = ([0-9.]+);", source)
    half_life = re.search(r"LN_2 / ([0-9.]+) \* self.bleach", source)
    if default and half_life:
        return {"kind": "first-order chromophore bleaching", "default_multiplier": float(default.group(1)),
                "half_life_seconds_at_one": float(half_life.group(1))}
    raise ValueError("unrecognized pigment loss model; record its actual law before comparing")


def describe_loss(model: dict) -> str:
    if model["kind"] == "legacy visible-film exponential sink":
        tau = model["e_folding_seconds"]
        return (f"legacy visible-film sink exp(-t/{tau:g}), an {tau:g}-second e-folding time, "
                f"leaving {math.exp(-8 / tau):.1%} after eight seconds in a periodic domain")
    half_life = model["half_life_seconds_at_one"]
    rate = model["default_multiplier"]
    return (f"separate first-order chromophore bleaching exp(-ln(2)*{rate:g}*t/{half_life:g}), "
            f"default multiplier {rate:g}, half-life {half_life:g} seconds at multiplier 1, "
            f"leaving {2 ** (-rate * 8 / half_life):.1%} after eight seconds in a periodic domain")


def read_field(path: Path, width: int, height: int) -> list[tuple[float, float, float]]:
    raw = path.read_bytes()
    assert len(raw) == width * height * 3 * 4, f"wrong field size: {path}"
    values = list(struct.iter_unpack("<fff", raw))
    assert all(math.isfinite(value) and value >= 0 for cell in values for value in cell), path
    return values


def linear_to_srgb(value: float) -> int:
    value = min(1.0, max(0.0, value))
    encoded = value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
    return round(encoded * 255)


def neutral_transmission(field: list[tuple[float, float, float]], width: int, height: int) -> Image.Image:
    # Fixed reference optics for comparing fields across renderer revisions.
    # These are the pre-whirlpool coefficients, with unit path/depth.
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            cell = field[x * height + y]  # Public Water buffers are x-major.
            for channel, coefficients in enumerate(EXTINCTION):
                sigma = sum(c * k for c, k in zip(cell, coefficients))
                pixels[(y * width + x) * 3 + channel] = linear_to_srgb(math.exp(-sigma))
    return Image.frombytes("RGB", (width, height), bytes(pixels))


def capture(stage: str, args: argparse.Namespace) -> None:
    destination = args.output / stage
    destination.mkdir(parents=True, exist_ok=True)
    archive = args.before_source if stage == "before" else args.after_source
    commands = []
    metadata = {
        "stage": stage, "utc_started": datetime.now(timezone.utc).isoformat(),
        "water_grid": [args.width, args.height], "patch_metres": [2.0, 2.0],
        "physics_step_seconds": 0.008, "warmup_seconds": 10.0, "evolution_seconds": 8.0,
        "capture_interval_seconds": 0.064,
        "drops": [{"point": point, "strength": 0.30, "diameter_multiplier": args.drop_size}
                  for point in ([0.43, 0.46], [0.58, 0.50], [0.55, 0.62])],
        "ongoing_rain": False, "source_archive": str(archive) if archive else "current workspace",
        "field_format": "little-endian float32, x-major, [cyan,magenta,amber] at every Water.dye sample",
        "diagnostic_optics": {"white_linear_illumination": [1.0, 1.0, 1.0], "path_over_depth": 1.0,
                              "extinction_matrix": EXTINCTION, "transfer": "standard sRGB",
                              "scope": "fixed pre-whirlpool reference optics, not the current scene shader"},
        "checkpoints": [{"requested_seconds": requested, "physical_seconds": tick * 0.008}
                        for tick, requested in CHECKPOINTS],
        "mass_note": "Area integral of public resampled Water.dye, not an internal finite-volume mass ledger. Open-boundary interpolation may affect quadrature.",
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "commands": commands,
    }
    with tempfile.TemporaryDirectory(prefix=f"acid-mixing-{stage}-") as temporary:
        build = Path(temporary) / "source"
        build.mkdir()
        if archive:
            metadata["input_archive_sha256"] = sha256(archive)
            with tarfile.open(archive) as saved:
                saved.extractall(build, filter="data")
        else:
            shutil.copytree(ROOT / "src", build / "src")
            for name in ("Cargo.toml", "Cargo.lock"):
                shutil.copy2(ROOT / name, build / name)
        assert all((build / "src" / f"{name}.rs").is_file() for name in ("water", "dye", "rapids")), \
            "mixing comparison requires both dye and rapids modules"
        metadata["source_sha256"] = {str(path.relative_to(build)): sha256(path)
                                      for path in sorted((build / "src").rglob("*.rs"))}
        metadata["cargo_sha256"] = {name: sha256(build / name) for name in ("Cargo.toml", "Cargo.lock")}
        metadata["pigment_loss_model"] = pigment_loss_model((build / "src/dye.rs").read_text())
        with tarfile.open(destination / "source.tar.gz", "w:gz") as saved:
            for name in ("src", "Cargo.toml", "Cargo.lock"):
                saved.add(build / name, arcname=name)
        metadata["saved_source_archive_sha256"] = sha256(destination / "source.tar.gz")
        optional_modules = [name for name in ("stirring", "drain")
                            if (build / "src" / f"{name}.rs").is_file()]
        optional = "\n".join(f"mod {name};" for name in optional_modules)
        driver = DRIVER.replace("// OPTIONAL_MODULES", optional)
        (destination / "driver.rs").write_text(driver)
        (build / "src/main.rs").write_text(driver)
        metadata["driver_sha256"] = sha256(destination / "driver.rs")
        metadata["stirring_module"] = "stirring" in optional_modules
        metadata["optional_modules"] = optional_modules
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(Path(temporary) / "target")
        metadata["rustflags"] = environment.get("RUSTFLAGS", "")
        executable = Path(environment["CARGO_TARGET_DIR"]) / "release/acid-rain"
        for command in (["cargo", "check", "--locked"], ["cargo", "build", "--locked", "--release"],
                        [str(executable), str(destination), str(args.width), str(args.height), str(args.drop_size)]):
            result = subprocess.run(command, cwd=build, env=environment, capture_output=True, text=True)
            commands.append({"argv": command, "status": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(result.stdout or result.stderr[-1200:], flush=True)
            result.check_returncode()
        metadata["driver_binary_sha256"] = sha256(executable)
    fields = {}
    for mode in MODES:
        for path in sorted((destination / mode).glob("field-*.f32")):
            field = read_field(path, args.width, args.height)
            neutral_transmission(field, args.width, args.height).save(path.with_suffix(".ppm"))
            fields[str(path.relative_to(destination))] = sha256(path)
    metadata["field_sha256"] = fields
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def comparison_frame(output: Path, mode: str, milliseconds: int, side: int = 320) -> Image.Image:
    sheet = Image.new("RGB", (side * 2 + 24, side + 64), (15, 17, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), f"{mode.upper()} / physical time {milliseconds / 1000:.3f}s / neutral dye diagnostic", fill=(234, 220, 238))
    for column, stage in enumerate(("before", "after")):
        x = 8 + column * (side + 8)
        draw.text((x, 27), stage.upper(), fill=(208, 216, 230))
        with Image.open(output / stage / mode / f"field-{milliseconds:06}.ppm") as frame:
            # The patch is 2m square even though its sampling grid is rectangular.
            sheet.paste(frame.resize((side, side), Image.Resampling.NEAREST), (x, 48))
    return sheet


def compose(args: argparse.Namespace) -> None:
    output = args.output
    metadata = {stage: json.loads((output / stage / "metadata.json").read_text()) for stage in ("before", "after")}
    before, after = metadata["before"], metadata["after"]
    for stage, entry in metadata.items():
        if "pigment_loss_model" not in entry:
            with tarfile.open(output / stage / "source.tar.gz") as archive:
                entry["pigment_loss_model"] = pigment_loss_model(archive.extractfile("src/dye.rs").read().decode())
    for key in ("water_grid", "physics_step_seconds", "warmup_seconds", "evolution_seconds", "drops", "ongoing_rain", "diagnostic_optics"):
        assert before[key] == after[key], f"unmatched mixing fixture: {key}"
    width, height = before["water_grid"]
    rows = []
    validation = {"pigment_loss_models": {stage: entry["pigment_loss_model"] for stage, entry in metadata.items()}}
    tables = ["| Mode | Time s | Version | Sampled mass | Retained | Peak concentration | Gradient RMS | Gradient / mass |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        checkpoints = {}
        for stage in ("before", "after"):
            with (output / stage / mode / "checkpoints.csv").open() as file:
                checkpoints[stage] = list(csv.DictReader(file))
            assert len(checkpoints[stage]) == len(CHECKPOINTS)
            assert all(math.isfinite(float(value)) for row in checkpoints[stage] for value in row.values())
        initial_fields = [read_field(output / stage / mode / "field-000000.f32", width, height) for stage in ("before", "after")]
        initial_l1 = sum(abs(a - b) for ca, cb in zip(*initial_fields) for a, b in zip(ca, cb))
        initial_sum = sum(sum(cell) for cell in initial_fields[0])
        assert initial_l1 / max(initial_sum, 1e-30) < 1e-6, "initial deposits differ between versions"
        validation[mode] = {"initial_relative_l1_difference": initial_l1 / max(initial_sum, 1e-30), "checkpoints": []}
        for index, (tick, requested) in enumerate(CHECKPOINTS):
            millis = tick * 8
            fields = [read_field(output / stage / mode / f"field-{millis:06}.f32", width, height) for stage in ("before", "after")]
            l1 = sum(abs(a - b) for ca, cb in zip(*fields) for a, b in zip(ca, cb)) * 4 / (width * height)
            reference_mass = float(checkpoints["before"][index]["sampled_total_mass"])
            sums = [sum(sum(cell) for cell in field) for field in fields]
            normalized_l1 = sum(abs(a / max(sums[0], 1e-30) - b / max(sums[1], 1e-30))
                                for ca, cb in zip(*fields) for a, b in zip(ca, cb))
            validation[mode]["checkpoints"].append({"physical_seconds": millis / 1000, "field_l1_difference": l1,
                                                    "relative_to_before_sampled_mass": l1 / max(reference_mass, 1e-30),
                                                    "mass_normalized_l1_difference": normalized_l1})
            for stage in ("before", "after"):
                row = checkpoints[stage][index]
                assert float(row["physical_seconds"]) == millis / 1000
                initial_mass = float(checkpoints[stage][0]["sampled_total_mass"])
                retention = float(row["sampled_total_mass"]) / initial_mass
                rows.append({"mode": mode, "version": stage, **row, "retained_fraction": retention})
                tables.append(f"| {mode} | {millis / 1000:.3f} | {stage} | {float(row['sampled_total_mass']):.6f} | "
                              f"{retention:.2%} | {float(row['peak_total_concentration']):.3f} | "
                              f"{float(row['gradient_rms']):.3f} | {float(row['gradient_per_mass']):.3f} |")
        cells = [comparison_frame(output, mode, tick * 8, 240) for tick, _ in CHECKPOINTS]
        contact = Image.new("RGB", (cells[0].width * 3, cells[0].height * 2), (15, 17, 25))
        for index, cell in enumerate(cells):
            contact.paste(cell, ((index % 3) * cell.width, (index // 3) * cell.height))
        contact.save(output / f"{mode}-concentration-contact.png")
        frames = [comparison_frame(output, mode, tick * 8) for tick in FRAME_TICKS]
        palette_samples = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
        for index, frame in enumerate(frames):
            palette_samples.paste(frame, (0, index * frame.height))
        palette = palette_samples.quantize(colors=256, dither=Image.Dither.NONE)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        durations = [round((index + 1) * 6.4) * 10 - round(index * 6.4) * 10 for index in range(len(frames) - 1)] + [500]
        indexed[0].save(output / f"{mode}-concentration-before-after.gif", save_all=True,
                        append_images=indexed[1:], duration=durations, loop=0, optimize=False, disposal=2)
        with Image.open(output / f"{mode}-concentration-before-after.gif") as clip:
            assert clip.n_frames == len(frames)
            elapsed = sum(clip.seek(index) or clip.info["duration"] for index in range(clip.n_frames))
            assert elapsed == 8500
            validation[mode]["gif"] = {"frames": clip.n_frames, "duration_ms": elapsed, "final_hold_ms": 500}
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Pigment redistribution: direct concentration diagnostic", "",
              "These are top-down diagnostic views of actual Water.dye samples, not screenshots of the terminal renderer. "
              "A neutral white beam passes through the measured pigment with the renderer's fixed extinction coefficients "
              "and unit path/depth ratio; standard sRGB encoding is the only display transfer. "
              "There are no reflections, camera changes, bloom, RGB trails, contrast enhancement or per-frame normalization.", "",
              "Both versions warm up classic/rapids water for 10 physical seconds with 8 ms steps, then receive exactly "
              "three simultaneous drops at (0.43,0.46), (0.58,0.50), (0.55,0.62), strength 0.30. "
              f"Drop diameter multiplier: {before['drops'][0]['diameter_multiplier']:g}. No more rain enters the fixture. "
              "The same 8 ms integration steps continue for eight seconds. Initial public pigment fields are checked for equality.", "",
              "Requested checkpoints are 0, 0.5, 1, 2, 4 and 8 seconds. Because 0.5 is not an integer number of 8 ms steps, "
              "its actual sample is 0.504 s; labels and CSV use the actual time. GIFs run at physical speed with a final 0.5 s hold. "
              "The 2m by 2m patch is shown square; nearest-neighbor enlargement preserves the captured samples.", "",
              *tables, "",
              "Mass here means the area integral of the resampled public Water.dye field, not the solver's internal "
              "finite-volume mass ledger. "
              f"Before: {describe_loss(before['pigment_loss_model'])}. "
              f"After: {describe_loss(after['pigment_loss_model'])}. "
              "The applicable loss laws are read from the respective archived sources. "
              "Rapids can additionally carry pigment out of the open channel. "
              "Resampling near open boundaries affects area quadrature. Exact conservation is verified by solver tests separately.", "",
              "Gradient RMS is sqrt(mean over sampled cells of the sum, over species, of squared forward x/y concentration "
              "derivatives in metres). The x derivative excludes the last open-channel edge; y is periodic. Gradient/mass "
              "helps separate field shape from overall pigment loss. Neither a larger gradient nor a lower peak alone "
              "proves more realistic mixing. Validation JSON includes both the raw concentration-field difference and "
              "the L1 difference after each field is divided by its own total sampled pigment mass; the latter isolates "
              "spatial/species distribution from uniform loss amplitude. Raw differences include the changed loss law.", "",
              "Each version retains original sources, SHA256 hashes, the generated driver, exact commands/compiler output, "
              "all raw little-endian concentration fields, neutral-transmission PPMs and checkpoint CSVs. The diagnostic driver "
              "replaces main.rs only in a temporary copy; production water and pigment modules are unchanged."]
    (output / "comparison.md").write_text("\n".join(report) + "\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("\n".join(tables))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("before", "after", "compose"))
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/mixing-before/source.tar.gz")
    parser.add_argument("--after-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/mixing-fields")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--drop-size", type=float, default=1.5)
    args = parser.parse_args()
    if not 8 <= args.width <= 512 or not 8 <= args.height <= 512:
        parser.error("diagnostic dimensions must each be between 8 and 512")
    if not math.isfinite(args.drop_size) or not 0.25 <= args.drop_size <= 3:
        parser.error("drop size must be finite and between 0.25 and 3")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "compose":
        compose(args)
    else:
        capture(args.stage, args)


if __name__ == "__main__":
    main()
