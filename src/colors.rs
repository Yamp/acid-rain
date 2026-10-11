use crate::scene::{
    SceneFrame,
    SceneKind,
};

/// Related linear-light pigments. The worlds differ in structure and depth,
/// while their reflected and transmitted layers can mix without rainbow bands.
const ROSE: [f32; 3] = [0.53, 0.31, 0.43];
const LILAC: [f32; 3] = [0.29, 0.24, 0.43];
const PEARL: [f32; 3] = [0.61, 0.54, 0.65];
const MINT: [f32; 3] = [0.29, 0.43, 0.39];

/// Directional, linear-HDR environment. Evaluated into the small environment map,
/// never for individual water samples. Cartesian textures have no azimuth seam.
#[inline]
pub fn sky_color(
    _elapsed: f32,
    dir: [f32; 3],
    scene: &SceneFrame,
) -> (f32, f32, f32) {
    let [cosine, sine] = scene.sky_rotation;
    let direction = [
        dir[0] * cosine - dir[1] * sine,
        dir[0] * sine + dir[1] * cosine,
        dir[2],
    ];
    let first = environment(scene.sky_a, direction, scene);
    let color = if scene.sky_mix > 0.0 {
        mix(
            first,
            environment(scene.sky_b, direction, scene),
            scene.sky_mix,
        )
    } else {
        first
    };
    let color = if scene.light_complexity > 0.0 {
        // Carve self-similar contours into each world's own light instead of
        // blending in a separate luminous grey field. The scalar mask preserves
        // chromaticity and cannot fill dark pockets or increase source radiance;
        // display adaptation still handles the resulting scene exposure.
        let folded = folded_light(direction, scene);
        let structure = crate::color_pipeline::luminance(folded) / scene.sky_energy;
        let mask = smoothstep(0.025, 0.26, structure);
        let gain = 1.0 - 0.84 * scene.light_complexity.sqrt() * (1.0 - mask);
        color.map(|channel| channel * gain)
    } else {
        color
    };
    let base_spectrum = (0.58 + scene.sky_energy * 0.10).min(0.80);
    let storm_mix = if scene.sky_a == SceneKind::Storm {
        1.0 - scene.sky_mix
    } else {
        0.0
    } + if scene.sky_b == SceneKind::Storm {
        scene.sky_mix
    } else {
        0.0
    };
    // Storm's broad cloud light otherwise dilutes the directional spectrum into
    // a neutral veil. Use the same hue field as adjacent worlds at greater
    // source purity, preserving luminance and the smooth Journey crossfade.
    // This changes cloud illumination, not global display saturation.
    let spectrum = base_spectrum + (0.90 - base_spectrum) * storm_mix;
    let color =
        crate::optics::spectral_emitter(color, direction, scene.interference_phase, spectrum);
    (color[0], color[1], color[2])
}

/// Three smooth folds repeat the same broad light contours at smaller scales.
/// This finite artistic emitter is cached by the renderer; it is not a recursive
/// ray march or part of the water equations. All folds have bounded derivatives.
fn folded_light(
    mut point: [f32; 3],
    scene: &SceneFrame,
) -> [f32; 3] {
    let dose = scene.light_complexity;
    let mut sum = [0.0; 3];
    let mut total = 0.0;
    let offset = 0.37 + dose * 0.13;
    // Upper bound on the first contour's directional phase derivative, including
    // its sinusoidal bend and the first 1.55x fold. Later folds scale this bound.
    let mut angular_frequency = 10.0;
    for (octave, weight) in [0.56, 0.29, 0.15].into_iter().enumerate() {
        // Rotate, gently fold, then magnify: self-similarity without hard abs()
        // cusps. z stays unfolded, providing layers behind the repeated arches.
        let rotated = [
            0.8 * point[0] - 0.6 * point[2],
            point[1],
            0.6 * point[0] + 0.8 * point[2],
        ];
        point = [
            ((rotated[0] * rotated[0] + 0.0225).sqrt() - offset) * 1.55,
            ((rotated[1] * rotated[1] + 0.0225).sqrt() - offset) * 1.55,
            rotated[2] * 1.55,
        ];
        let phase = scene.sky_phase[octave] + octave as f32 * 1.7;
        let bend = (point[1] * 2.1 - point[2] * 1.7 + phase * 0.55).sin();
        let contour =
            (point[0] * 3.7 + point[1] * 2.9 - point[2] * 2.3 + bend * 0.45 + phase).sin();
        // The environment is a 64x64 octahedral map. A conservative 0.08-radian
        // footprint widens fine contours instead of sharpening them into texel
        // noise; high octaves also carry less energy. This is analytic low-pass
        // regularization of the emitter, not full per-pixel texture filtering.
        let width = 0.30 + angular_frequency * 0.08 * 0.5;
        let membrane = width * width / (width * width + contour * contour);
        let depth = 0.5 + 0.5 * (point[2] * 2.2 - point[0] * 1.3 + phase * 0.37).sin();
        let pigment = mix(
            mix(LILAC, ROSE, depth),
            mix(MINT, PEARL, depth),
            0.25 + 0.25 * bend,
        );
        let illumination = 0.018 + membrane * (0.28 + 0.72 * depth) * scene.sky_energy;
        let weight = weight * (1.0 + octave as f32 * dose * 0.20);
        for channel in 0..3 {
            sum[channel] += pigment[channel] * illumination * weight;
        }
        total += weight;
        angular_frequency *= 1.55;
    }
    sum.map(|value| value / total)
}

#[inline(always)]
fn mix(
    a: [f32; 3],
    b: [f32; 3],
    t: f32,
) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

#[inline(always)]
fn glow(
    base: [f32; 3],
    color: [f32; 3],
    strength: f32,
) -> [f32; 3] {
    std::array::from_fn(|i| base[i] + color[i] * strength)
}

#[inline(always)]
fn smoothstep(
    low: f32,
    high: f32,
    value: f32,
) -> f32 {
    let x = ((value - low) / (high - low)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[inline]
fn environment(
    kind: SceneKind,
    [x, y, z]: [f32; 3],
    scene: &SceneFrame,
) -> [f32; 3] {
    let [p, q, r, s] = scene.sky_phase;
    // A rational rotation twists the environment into vortices without atan2,
    // angular seams, or extra per-texel trigonometric calls.
    let twist = match kind {
        SceneKind::Journey | SceneKind::Acid => 1.25,
        SceneKind::Prism => 1.8,
        SceneKind::Abyss => 0.65,
        SceneKind::Lava => 0.9,
        _ => 0.0,
    };
    let k = (z + x * y * 0.55) * twist;
    let inverse = 1.0 / (1.0 + k * k);
    let cos_twist = (1.0 - k * k) * inverse;
    let sin_twist = 2.0 * k * inverse;
    let (x, y) = (x * cos_twist - y * sin_twist, x * sin_twist + y * cos_twist);
    let elevation = (z + 1.0) * 0.5;
    let horizon = 1.0 / (1.0 + 14.0 * z * z);
    let energy = scene.sky_energy;
    match kind {
        SceneKind::Journey | SceneKind::Acid => {
            // Related pigments overlap like translucent rose and lilac glass;
            // broad shadow pockets retain depth without hard spectral edges.
            let warp = (x * 7.0 - y * 5.0 + z * 3.8 + p).sin();
            let folds = (y * 11.5 + z * 7.3 + warp * 1.7 + q).sin();
            let plasma = (x * 6.2 + y * 4.0 - z * 6.4 + folds * 1.2 + r).sin();
            let t = smoothstep(-0.8, 0.8, plasma);
            let cold = mix(LILAC, MINT, t * 0.24);
            let warm = mix(ROSE, PEARL, t * 0.45);
            let palette = mix(cold, warm, smoothstep(-0.8, 0.8, folds));
            let shading = 0.035 + 0.57 * smoothstep(-0.7, 0.8, warp * folds);
            let base = palette.map(|channel| channel * shading);
            let membrane = 1.0 / (1.0 + 24.0 * plasma * plasma);
            let glow_color = mix(ROSE, PEARL, smoothstep(-0.8, 0.8, folds));
            glow(base, glow_color, membrane * 0.95 * energy)
        },
        SceneKind::Aurora => {
            let bend = (y * 4.4 - z * 1.8 + p * 0.65).sin();
            let ribbon = (x * 5.8 + y * 3.7 + bend * 1.4 + q * 0.65).sin();
            let curtain = 1.0 / (1.0 + 18.0 * ribbon * ribbon);
            let filament = 0.58 + 0.42 * (x * 15.2 + y * 10.4 + bend * 2.0 + r).sin();
            let sky = mix([0.12, 0.11, 0.17], [0.035, 0.033, 0.07], elevation);
            let sky = glow(sky, ROSE, horizon * 0.18);
            let aurora = mix(
                mix(PEARL, MINT, 0.25),
                mix(ROSE, LILAC, 0.4),
                smoothstep(0.25, 0.95, z),
            );
            let altitude = smoothstep(-0.5, 0.6, z);
            glow(
                sky,
                aurora,
                curtain * filament * (0.22 + altitude) * energy * 0.70,
            )
        },
        SceneKind::Storm => {
            // Rounded cloud banks and cool diffuse light keep this world restrained.
            let billow = (x * 3.2 + y * 2.4 + p * 0.32).sin();
            let cloud = (x * 6.5 - y * 3.0 + z * 4.0 + billow * 1.8 + q * 0.35).sin();
            let veil = smoothstep(-0.8, 0.9, cloud * 0.68 + billow * 0.32);
            let sky = mix([0.16, 0.14, 0.21], [0.055, 0.05, 0.09], elevation);
            let clouds = mix(sky, [0.36, 0.32, 0.41], veil * 0.65);
            let light = (x * 0.4 + y * -0.3 + z * 0.866).max(0.0);
            let halo = light * light * light * light;
            glow(
                clouds,
                PEARL,
                halo * (0.12 + scene.storm_glow) + horizon * 0.06,
            )
        },
        SceneKind::Abyss => {
            let current = (x * 6.6 + y * 3.3 - z * 4.5 + p * 0.6).sin();
            let tissue = (x * -5.0 + y * 9.4 + z * 5.9 + current * 2.2 + q * 0.7).sin();
            let caustic = 1.0 / (1.0 + 30.0 * tissue * tissue);
            let base = mix([0.023, 0.02, 0.049], [0.09, 0.075, 0.14], elevation);
            let nebula = smoothstep(-0.3, 1.0, current * tissue);
            let base = glow(base, LILAC, nebula * 0.36);
            let dust = ((x * 22.4 + z * 13.6 + s).sin() * (y * 24.8 - z * 15.2 - s).sin()).max(0.0);
            let dust = dust * dust * dust * dust;
            glow(
                base,
                mix(PEARL, MINT, 0.12),
                (caustic * 0.55 + dust * dust * 0.045) * energy,
            )
        },
        SceneKind::Lava => {
            let flow = (x * 4.8 + y * -3.5 + z * 3.3 + p * 0.45).sin();
            let vein = (x * 8.9 + y * 6.2 - z * 4.6 + flow * 2.0 + q * 0.5).sin();
            let molten = 1.0 / (1.0 + 14.0 * vein * vein);
            let hot_core = molten * molten * molten;
            let rock = mix(
                [0.045, 0.028, 0.057],
                [0.15, 0.085, 0.125],
                flow * 0.5 + 0.5,
            );
            let rock = glow(rock, ROSE, molten * energy * 0.65 + horizon * 0.10);
            glow(rock, PEARL, hot_core * 0.30 * energy)
        },
        SceneKind::Prism => {
            // Mirrored directional interference creates a breathing crystal lattice.
            let a = (x.abs() * 10.4 - y.abs() * 7.2 + z * 3.6 + p * 0.6).sin();
            let b = ((x * y).abs() * 15.2 + z * 6.4 - q * 0.7).sin();
            let c = ((x * x - y * y) * 8.0 + z * 4.8 + a * 1.3 + r).sin();
            let mask = 0.035 + 0.6 * smoothstep(-0.5, 0.8, a * b * 0.65 + c * 0.35);
            let pigment = mix(LILAC, ROSE, smoothstep(-0.8, 0.8, a * 0.6 + c * 0.4));
            let pigment = mix(pigment, PEARL, (b * 0.5 + 0.5) * 0.35);
            let spectral = pigment.map(|channel| channel * mask);
            let line = 1.0 / (1.0 + 24.0 * (a + b) * (a + b));
            let spectrum = mix(PEARL, ROSE, smoothstep(-0.8, 0.8, c) * 0.4);
            glow(spectral, spectrum, line * energy * 0.85)
        },
    }
}

#[cfg(test)]
mod scene_tests {
    use clap::ValueEnum;

    use super::*;
    use crate::scene::{
        TRANSITION_DURATION,
        WORLD_DURATION,
    };

    fn difference(
        a: (f32, f32, f32),
        b: (f32, f32, f32),
    ) -> f32 {
        (a.0 - b.0).abs() + (a.1 - b.1).abs() + (a.2 - b.2).abs()
    }

    #[test]
    fn storm_emitter_keeps_cloud_luminance_and_distinct_spectral_colors() {
        let frame = SceneFrame::new(SceneKind::Storm, 186.0, 2.0);
        let mut unmasked = frame;
        unmasked.light_complexity = 0.0;
        let mut chroma = 0.0;
        let mut count = 0;
        for row in 0..8 {
            let z = (row as f32 + 0.5) / 8.0;
            let radius = (1.0 - z * z).sqrt();
            for col in 0..32 {
                let (s, c) = (col as f32 * std::f32::consts::TAU / 32.0).sin_cos();
                let dir = [radius * c, radius * s, z];
                let [rotation_c, rotation_s] = frame.sky_rotation;
                let source_dir = [
                    dir[0] * rotation_c - dir[1] * rotation_s,
                    dir[0] * rotation_s + dir[1] * rotation_c,
                    dir[2],
                ];
                let source = environment(SceneKind::Storm, source_dir, &frame);
                let result = sky_color(186.0, dir, &unmasked);
                let result = [result.0, result.1, result.2];
                assert!(
                    (crate::color_pipeline::luminance(source)
                        - crate::color_pipeline::luminance(result))
                    .abs()
                        < 1e-6
                );
                let high = result.iter().copied().fold(0.0_f32, f32::max);
                let low = result.iter().copied().fold(f32::INFINITY, f32::min);
                chroma += (high - low) / high;
                count += 1;
            }
        }
        let average = chroma / count as f32;
        assert!(
            average > 0.45,
            "Storm's spectral clouds became a low-chroma veil: {average}"
        );
    }

    #[test]
    fn overdrive_adds_spatial_structure_beyond_a_brightness_change() {
        for kind in [SceneKind::Acid, SceneKind::Aurora, SceneKind::Prism] {
            let normal = SceneFrame::new(kind, 17.0, 2.0);
            let intense = SceneFrame::new(kind, 17.0, 4.0);
            let mut values = Vec::new();
            let mut aa = 0.0;
            let mut ab = 0.0;
            for row in 0..16 {
                let z = (row as f32 + 0.5) / 8.0 - 1.0;
                let radius = (1.0 - z * z).sqrt();
                for col in 0..32 {
                    let phase = col as f32 * std::f32::consts::TAU / 32.0;
                    let (s, c) = phase.sin_cos();
                    let dir = [radius * c, radius * s, z];
                    let a = sky_color(17.0, dir, &normal);
                    let b = sky_color(17.0, dir, &intense);
                    for (a, b) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                        values.push((a, b));
                        aa += a * a;
                        ab += a * b;
                    }
                }
            }
            let brightness = ab / aa;
            let shape_change = values
                .iter()
                .map(|(a, b)| (b - a * brightness).powi(2))
                .sum::<f32>();
            assert!(
                (shape_change / aa).sqrt() > 0.08,
                "{kind:?}: increasing dose must change structure, not only brightness"
            );
        }
    }

    #[test]
    fn zero_dose_keeps_the_original_environment_exactly() {
        let frame = SceneFrame::new(SceneKind::Acid, 0.0, 0.0);
        for dir in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.36, 0.48, 0.8]] {
            let original = environment(frame.sky_a, dir, &frame);
            let expected = crate::optics::spectral_emitter(
                original,
                dir,
                frame.interference_phase,
                (0.58 + frame.sky_energy * 0.10).min(0.80),
            );
            let actual = sky_color(0.0, dir, &frame);
            assert_eq!([actual.0, actual.1, actual.2], expected);
        }
    }

    #[test]
    fn layered_contours_preserve_the_worlds_original_dark_pockets() {
        for kind in [SceneKind::Acid, SceneKind::Abyss, SceneKind::Prism] {
            let frame = SceneFrame::new(kind, 17.0, 4.0);
            let mut original = frame;
            original.light_complexity = 0.0;
            let mut dark_samples = 0;
            for row in 0..16 {
                let z = (row as f32 + 0.5) / 8.0 - 1.0;
                let radius = (1.0 - z * z).sqrt();
                for col in 0..32 {
                    let (s, c) = (col as f32 * std::f32::consts::TAU / 32.0).sin_cos();
                    let dir = [radius * c, radius * s, z];
                    let base = sky_color(17.0, dir, &original);
                    let actual = sky_color(17.0, dir, &frame);
                    let base_y = base.0 * 0.2126 + base.1 * 0.7152 + base.2 * 0.0722;
                    if base_y < 0.07 {
                        dark_samples += 1;
                        let y = actual.0 * 0.2126 + actual.1 * 0.7152 + actual.2 * 0.0722;
                        assert!(
                            y <= base_y + 1e-7,
                            "{kind:?}: light detail filled a shadow {base_y} -> {y}"
                        );
                    }
                }
            }
            assert!(
                dark_samples > 20,
                "regression must exercise original shadow pockets"
            );
        }
    }

    #[test]
    fn pastel_skies_keep_depth_between_shadow_and_translucent_layers() {
        for scene in [SceneKind::Acid, SceneKind::Prism] {
            let frame = SceneFrame::new(scene, 13.0, 1.0);
            let mut darkest = f32::MAX;
            let mut brightest = 0.0_f32;
            for row in 0..24 {
                let z = (row as f32 + 0.5) / 24.0;
                let radius = (1.0 - z * z).sqrt();
                for column in 0..48 {
                    let angle = column as f32 * std::f32::consts::TAU / 48.0;
                    let (sine, cosine) = angle.sin_cos();
                    let color = sky_color(13.0, [radius * cosine, radius * sine, z], &frame);
                    let luminance = color.0 * 0.2126 + color.1 * 0.7152 + color.2 * 0.0722;
                    darkest = darkest.min(luminance);
                    brightest = brightest.max(luminance);
                }
            }
            assert!(darkest < 0.075, "{scene:?} has no dark pockets: {darkest}");
            assert!(
                brightest > darkest * 6.0 && brightest > 0.25,
                "{scene:?} has no luminous structures: {brightest}"
            );
        }
    }

    #[test]
    fn related_pastel_worlds_avoid_hot_red_and_isolated_spectral_primaries() {
        for scene in SceneKind::value_variants() {
            for time in [0.0, 17.0, 80.0] {
                let frame = SceneFrame::new(*scene, time, 4.0);
                for row in 0..12 {
                    let z = (row as f32 + 0.5) / 6.0 - 1.0;
                    let radius = (1.0 - z * z).sqrt();
                    for column in 0..24 {
                        let angle = column as f32 * std::f32::consts::TAU / 24.0;
                        let (sine, cosine) = angle.sin_cos();
                        let (red, green, blue) =
                            sky_color(time, [radius * cosine, radius * sine, z], &frame);
                        let maximum = red.max(green).max(blue);
                        let minimum = red.min(green).min(blue);
                        assert!(
                            minimum >= maximum * 0.35,
                            "{scene:?} has a harsh spectral primary: {red}, {green}, {blue}"
                        );
                        assert!(
                            blue >= red * 0.60,
                            "{scene:?} has a hot red/orange region: {red}, {green}, {blue}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn each_world_has_its_own_environment() {
        let direction = [0.36, 0.48, 0.80];
        let worlds = &SceneKind::value_variants()[1..];
        for (index, scene) in worlds.iter().enumerate() {
            let first = sky_color(13.0, direction, &SceneFrame::new(*scene, 13.0, 1.0));
            for other in &worlds[index + 1..] {
                let second = sky_color(13.0, direction, &SceneFrame::new(*other, 13.0, 1.0));
                assert!(
                    difference(first, second) > 0.02,
                    "{scene:?} and {other:?} look identical"
                );
            }
        }
    }

    #[test]
    fn sky_stays_finite_and_nonnegative_in_every_direction() {
        for scene in SceneKind::value_variants() {
            for time in [
                0.0,
                WORLD_DURATION - TRANSITION_DURATION,
                WORLD_DURATION - 0.001,
                WORLD_DURATION,
                WORLD_DURATION * 6.0 - 0.001,
                WORLD_DURATION * 6.0,
                100_000.0,
            ] {
                let frame = SceneFrame::new(*scene, time, 4.0);
                for direction in [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [0.36, 0.48, 0.8],
                ] {
                    let rgb = sky_color(time, direction, &frame);
                    for channel in [rgb.0, rgb.1, rgb.2] {
                        assert!(channel.is_finite() && (0.0..=16.0).contains(&channel));
                    }
                }
            }
        }
    }

    #[test]
    fn worlds_crossfade_without_flashes_or_an_azimuth_seam() {
        for world in SceneKind::value_variants() {
            for time in [
                0.0,
                WORLD_DURATION - TRANSITION_DURATION,
                WORLD_DURATION,
                WORLD_DURATION * 2.0 - TRANSITION_DURATION,
                WORLD_DURATION * 2.0,
                WORLD_DURATION * 6.0 - TRANSITION_DURATION,
                WORLD_DURATION * 6.0,
            ] {
                let before = SceneFrame::new(*world, time - 0.001, 4.0);
                let after = SceneFrame::new(*world, time + 0.001, 4.0);
                let rgb_a = sky_color(time - 0.001, [0.36, 0.48, 0.8], &before);
                let rgb_b = sky_color(time + 0.001, [0.36, 0.48, 0.8], &after);
                assert!(
                    difference(rgb_a, rgb_b) < 0.02,
                    "{world:?} flashes at {time}"
                );
                let seam_a = sky_color(time, [-1.0, -0.000001, 0.0], &before);
                let seam_b = sky_color(time, [-1.0, 0.000001, 0.0], &before);
                assert!(
                    difference(seam_a, seam_b) < 0.001,
                    "{world:?} has an azimuth seam"
                );
            }
        }
    }
}
