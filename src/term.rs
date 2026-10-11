use std::{
    f32::consts::PI,
    io::Write,
};

use anyhow::Result;
use ndarray::Array2;

use crate::{
    color_pipeline::{
        DisplayTransform,
        decode_byte,
        encode_srgb,
    },
    colors::sky_color,
    output::{
        Cell,
        TerminalOutput,
    },
    physics::{
        DEPTH,
        PATCH_LENGTH,
    },
    scene::{
        SceneFrame,
        SceneKind,
    },
    water::Water,
};


const CAMERA_DIST: f32 = 1.0;
const CHAR_ASPECT: f32 = 2.0;
const WATER_SCALE: f32 = PATCH_LENGTH;
const DRIFT_SPEED: f32 = 0.15;
const DRIFT_TURN_PERIOD: f32 = 180.0;
const TAA_KEEP: f32 = 0.3;
const BLOOM_THRESHOLD: f32 = 0.35;
const BLOOM_INTENSITY: f32 = 0.28;
// A half-resolution five-tap Gaussian replaces the full-resolution 22-tap pass.
const BLOOM_KERNEL: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
const NDF_FILTER_K: f32 = 0.5;

// Effective unresolved microfacet roughness; resolved normal variance still
// broadens the lobe below to keep subpixel glints filtered.
const GGX_ROUGHNESS: f32 = 0.05;

// Water IOR varies weakly with wavelength (RGB spectral approximation).
const IOR_RGB: [f32; 3] = [1.331, 1.333, 1.337];

// Water transparency — physically-based volume transport
const WATER_DEPTH: f32 = DEPTH;
const IOR_WATER: f32 = 1.333; // refractive index
// Clear rose glass: a little more green absorption leaves a gentle rose/lilac tint.
const ABSORB: [f32; 3] = [1.2, 1.6, 1.2];
// Chosen clear-water scattering coefficient in m^-1. ABSORB remains an authored
// rose-tinted absorption spectrum; these are not calibrated natural-water data.
const WATER_SCATTER: f32 = 0.015;
const FOAM_ALBEDO: f32 = 0.86;
#[cfg(test)]
const BED_ALBEDO: f32 = 0.90; // passive mineral-albedo upper bound
const DIRECT_RADIANCE: f32 = 0.18;
const BED_LIGHT_SIZE: usize = 64;

#[cfg(test)]
mod sampling_tests {
    use super::*;

    #[test]
    fn unlit_clear_water_cannot_create_absorption_colored_light() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 24.0, 4.0);
        scene.light_dir = [0.0, 0.0, -1.0];
        let camera = Camera::new(24.0, 80, 24, &scene);
        let levels = Array2::zeros((8, 8));
        let normals = vec![[0.0, 0.0, 1.0, 0.0]; 64];
        let sky = SkyEnvironment::constant((0.0, 0.0, 0.0));
        let actual = shade_water(
            &levels,
            MaterialGrid::clear(),
            &normals,
            &[1.0; 64],
            &sky,
            8,
            8,
            scene.light_dir,
            &camera,
            [0.0, 0.0, -1.0],
            [0.5, 0.5],
            &scene,
            0.0,
            [0.0; 2],
            0.0,
            0,
        );
        assert_eq!(
            actual, [0.0; 3],
            "absorption cannot turn into a light source"
        );
    }

    #[test]
    fn colored_incident_light_creates_resolved_hues_without_emitting() {
        let scene = SceneFrame::new(SceneKind::Acid, 12.0, 2.0);
        let mut sky = SkyEnvironment::new();
        sky.update(32, 12.0, &scene);
        sky.bed_emission_strength = 0.0;
        let samples: Vec<_> = (0..25)
            .map(|i| {
                let point = [0.12 + (i % 5) as f32 * 0.19, 0.12 + (i / 5) as f32 * 0.19];
                submerged_bed_radiance(
                    MaterialGrid::clear(),
                    &sky,
                    32,
                    32,
                    point,
                    0.16,
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0],
                    &scene,
                )
            })
            .collect();
        let mean_chroma = samples
            .iter()
            .map(|p| {
                let high = p.iter().copied().fold(0.0_f32, f32::max);
                let low = p.iter().copied().fold(f32::INFINITY, f32::min);
                (high - low) / high.max(1e-6)
            })
            .sum::<f32>()
            / samples.len() as f32;
        assert!(
            mean_chroma > 0.45,
            "incident colors dissolved into gray: {mean_chroma}"
        );
    }

    #[test]
    fn submerged_diffuse_bed_does_not_show_the_directional_sky_through_flat_water() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 12.0, 2.0);
        scene.absorption_scale = 0.0;
        // Isolate the unfiltered passive boundary; colored projected lighting
        // and emissive material legitimately vary between different bed points.
        scene.light_complexity = 0.0;
        scene.interference_strength = 0.0;
        scene.light_dir = [0.0, 0.0, -1.0];
        let camera = Camera::new(12.0, 80, 24, &scene);
        let levels = Array2::zeros((8, 8));
        let normals = vec![[0.0, 0.0, 1.0, 0.0]; 64];
        let caustics = vec![1.0; 64];
        let mut sky = SkyEnvironment::new();
        sky.update(64, 12.0, &scene);
        sky.volume_incident = [0.0; 3]; // isolate the diffuse lower boundary
        let diffuse = |dir: [f32; 3]| {
            let rgb = shade_water(
                &levels,
                MaterialGrid::clear(),
                &normals,
                &caustics,
                &sky,
                8,
                8,
                scene.light_dir,
                &camera,
                dir,
                [0.5, 0.5],
                &scene,
                0.0,
                [0.0; 2],
                0.0,
                0,
            );
            let reflected = sample_sky(&sky, [dir[0], dir[1], -dir[2]]);
            let f = fresnel_rgb(-dir[2]);
            std::array::from_fn::<_, 3, _>(|c| {
                let bed = refracted_bed_point([0.5, 0.5], [0.0, 0.0, 1.0], dir, DEPTH, IOR_RGB[c])
                    .unwrap()
                    .0;
                (rgb[c] - f[c] * [reflected.0, reflected.1, reflected.2][c]) / bed_albedo(bed)
            })
        };
        let a = diffuse(norm3([0.4, 0.0, -1.0]));
        let b = diffuse(norm3([-0.4, 0.0, -1.0]));
        for c in 0..3 {
            assert!(
                (a[c] - b[c]).abs() < 2e-6,
                "diffuse irradiance per unit bed albedo cannot carry a directional sky image: \
                 {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn atmospheric_airlight_is_continuous_around_a_downward_view() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 12.0, 2.0);
        scene.fog_density = 100.0;
        let mut camera = Camera::new(12.0, 80, 24, &scene);
        camera.pos = [0.0, 0.0, 1.0];
        camera.ground = [0.0; 2];
        camera.patch_center = [0.0; 2];
        let levels = Array2::zeros((8, 8));
        let normals = vec![[0.0, 0.0, 1.0, 0.0]; 64];
        let caustics = vec![1.0; 64];
        let mut sky = SkyEnvironment::new();
        sky.update(64, 12.0, &scene);
        let mut samples = Vec::new();
        for x in [-0.001, 0.0, 0.001] {
            camera.fwd = norm3([x, 0.0, -1.0]);
            samples.push(
                shade_ray(
                    &camera,
                    &levels,
                    MaterialGrid::clear(),
                    &normals,
                    &caustics,
                    &sky,
                    8,
                    8,
                    scene.light_dir,
                    0.5,
                    0.5,
                    &scene,
                    0.0,
                    [0.0; 2],
                    0.0,
                    0,
                )
                .0,
            );
        }
        for sample in &samples {
            for c in 0..3 {
                assert!(
                    (sample[c] - samples[1][c]).abs() < 0.001,
                    "airlight must not unfold the horizon into radial spokes: {samples:?}"
                );
            }
        }
    }

    #[test]
    fn diffuse_bed_is_passive_and_cannot_emit_without_incident_light() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 0.0, 2.0);
        scene.light_dir = [0.0, 0.0, -1.0];
        scene.absorption_scale = 0.0;
        for input in [[0.0; 3], [0.3, 0.6, 0.9]] {
            let sky = SkyEnvironment::constant((input[0], input[1], input[2]));
            let result = submerged_bed_radiance(
                MaterialGrid::clear(),
                &sky,
                8,
                8,
                [0.5, 0.5],
                0.16,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                scene.light_dir,
                &scene,
            );
            for c in 0..3 {
                assert!(result[c] >= 0.0 && result[c] <= BED_ALBEDO * input[c]);
                let attenuation = (-WATER_SCATTER * 0.16 * (1.0 + sky.bed_illumination_path)).exp();
                assert!(
                    (result[c] - bed_albedo([0.5, 0.5]) * sky.bed_irradiance[c] * attenuation)
                        .abs()
                        < 1e-7
                );
            }
            if input == [0.0; 3] {
                assert_eq!(result, [0.0; 3]);
            }
        }
    }

    #[test]
    fn bed_beer_attenuates_both_legs_and_actual_local_pigment() {
        let near = SampleFootprint::new(8, 8, 0.2, 0.5);
        let far = SampleFootprint::new(8, 8, 0.8, 0.5);
        let pigment: Vec<_> = (0..64)
            .map(|i| {
                if i / 8 >= 4 {
                    [0.4, 0.1, 0.0]
                } else {
                    [0.0; 3]
                }
            })
            .collect();
        let material = MaterialGrid {
            dye: Some(&pigment),
            ..MaterialGrid::clear()
        };
        let depth = 0.16;
        let clear = bed_path_transmittance(material, near, near, depth, 1.0, 1.0, 0.2);
        let one_leg = WaterTransport::new(depth, 0.2, WATER_SCATTER).transmission;
        let dyed = bed_path_transmittance(material, far, far, depth, 1.0, 1.0, 0.2);
        let longer = bed_path_transmittance(material, far, far, depth, 1.0, 1.5, 0.2);
        let extinction = [
            0.4 * 2.0 + 0.1 * 0.3,
            0.4 * 0.65 + 0.1 * 1.8,
            0.4 * 0.3 + 0.1 * 0.45,
        ];
        for c in 0..3 {
            assert!(
                (clear[c] - (-(ABSORB[c] * 0.2 + WATER_SCATTER) * depth * 2.0).exp()).abs() < 2e-7
            );
            assert!(clear[c] < one_leg[c]);
            let expected = (-(ABSORB[c] * 0.2 + WATER_SCATTER) * depth * 2.0
                - extinction[c] * 0.85 * 2.0)
                .exp();
            assert!((dyed[c] - expected).abs() < 2e-7);
            assert!(0.0 <= longer[c] && longer[c] < dyed[c] && dyed[c] < clear[c]);
        }
        let without_dye =
            bed_path_transmittance(MaterialGrid::clear(), near, near, depth, 1.0, 1.0, 0.2);
        // The constant no-material path may fold exp at compile time, while
        // a sampled zero-pigment path evaluates it at runtime. Those agree
        // physically but can differ by one f32 ULP in an optimized build.
        for (local_zero, no_material) in clear.into_iter().zip(without_dye) {
            assert!(
                (local_zero - no_material).abs() < 1e-7,
                "remote pigment cannot tint an unoccupied local path: {local_zero} vs \
                 {no_material}"
            );
        }
    }

    #[test]
    fn snell_bed_projection_uses_metric_depth_and_stays_inside_the_streaming_halo() {
        let dir = norm3([0.4, -0.2, -1.0]);
        let (point, path) =
            refracted_bed_point([0.5, 0.5], [0.0, 0.0, 1.0], dir, 0.16, IOR_WATER).unwrap();
        let cos_t = (1.0 - (1.0 - dir[2] * dir[2]) / IOR_WATER.powi(2)).sqrt();
        assert!((path - 1.0 / cos_t).abs() < 1e-6);
        assert!((point[0] - (0.5 + dir[0] / IOR_WATER * 0.16 / cos_t / PATCH_LENGTH)).abs() < 1e-6);
        for corner in [[0.0, 0.0], [1.0, 1.0]] {
            let (point, path) =
                refracted_bed_point(corner, [0.0, 0.0, 1.0], dir, 10.0, IOR_WATER).unwrap();
            assert!(
                point
                    .into_iter()
                    .zip(corner)
                    .all(|(p, c)| (p - c).abs() <= crate::bed::REFRACTION_HALO / PATCH_LENGTH)
            );
            assert!(path.is_finite() && path >= 1.0);
        }
    }

    #[test]
    fn bed_material_stays_at_its_world_position_when_the_patch_moves() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 0.0, 0.0);
        scene.absorption_scale = 0.0;
        let mut sky = SkyEnvironment::constant((1.0, 1.0, 1.0));
        let sample = |sky: &SkyEnvironment, point| {
            submerged_bed_radiance(
                MaterialGrid::clear(),
                sky,
                8,
                8,
                point,
                0.16,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                &scene,
            )
        };
        let before = sample(&sky, [0.5, 0.5]);
        sky.prepare_world_bed([-0.75, -1.375], 7);
        assert_eq!(before, sample(&sky, [0.375, 0.6875]));
    }

    #[test]
    fn world_bed_filter_stays_fixed_during_patch_translation() {
        let mut sky = SkyEnvironment::constant((1.0, 1.0, 1.0));
        let scene = SceneFrame::new(SceneKind::Acid, 12.0, 2.0);
        sky.update_bed_light_filter(&scene);
        let before: [f32; 3] = std::array::from_fn(|c| sky.bed_light_transmission([0.5, 0.5], c));
        sky.prepare_world_bed([-0.75, -1.375], 7);
        sky.update_bed_light_filter(&scene);
        let after: [f32; 3] =
            std::array::from_fn(|c| sky.bed_light_transmission([0.375, 0.6875], c));
        assert!(
            before
                .into_iter()
                .zip(after)
                .all(|(a, b)| (a - b).abs() < 2e-6),
            "{before:?} vs {after:?}"
        );
    }

    #[test]
    fn world_bed_refraction_continues_beyond_water_patch() {
        let direction = norm3([0.4, -0.2, -1.0]);
        let (point, _) =
            refracted_bed_point([0.99, 0.01], [0.0, 0.0, 1.0], direction, 0.7, IOR_WATER).unwrap();
        assert!(point[0] > 1.0 && point[1] < 0.0, "{point:?}");
    }

    #[test]
    fn neutral_bed_has_resolved_world_space_structure_without_colored_emission() {
        let mut scene = SceneFrame::new(SceneKind::Acid, 0.0, 2.0);
        scene.light_dir = [0.0, 0.0, -1.0];
        scene.absorption_scale = 0.0;
        let sky = SkyEnvironment::constant((1.0, 1.0, 1.0));
        let samples: Vec<_> = (0..32)
            .map(|x| {
                submerged_bed_radiance(
                    MaterialGrid::clear(),
                    &sky,
                    64,
                    64,
                    [(x as f32 + 0.5) / 32.0, 0.41],
                    0.16,
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    scene.light_dir,
                    &scene,
                )
            })
            .collect();
        let low = samples.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let high = samples.iter().map(|p| p[1]).fold(0.0, f32::max);
        assert!(
            high - low > 0.08,
            "the resolved refracted bed cannot be a featureless plate: {low}..{high}"
        );
        for p in samples {
            // With vertical Snell rays every wavelength hits the same neutral
            // material; only wavelength-dependent surface Fresnel may differ.
            let albedo = std::array::from_fn::<_, 3, _>(|c| p[c] / sky.bed_irradiance[c]);
            assert!((albedo[0] - albedo[1]).abs() < 1e-6);
            assert!((albedo[1] - albedo[2]).abs() < 1e-6);
            assert!((0.10..=0.90).contains(&albedo[1]));
        }
    }

    #[test]
    fn bed_filter_is_passive_continuous_and_independent_of_camera() {
        let mut sky = SkyEnvironment::constant((1.0, 1.0, 1.0));
        let mut scene = SceneFrame::new(SceneKind::Acid, 12.0, 2.0);
        sky.update_bed_light_filter(&scene);
        for value in sky.bed_light_filter.iter().flatten() {
            assert!(value.is_finite() && (0.12..=1.0).contains(value));
        }
        let previous = sky.bed_light_filter.clone();
        scene.sky_phase[2] += 1e-4;
        sky.update_bed_light_filter(&scene);
        assert!(
            previous
                .iter()
                .flatten()
                .zip(sky.bed_light_filter.iter().flatten())
                .all(|(a, b)| (a - b).abs() < 0.001)
        );
        scene.light_complexity = 0.0;
        sky.update_bed_light_filter(&scene);
        assert_eq!(sky.bed_light_transmission([0.23, 0.75], 1), 1.0);
    }

    #[test]
    fn emissive_bed_is_bounded_and_pigment_attenuates_only_the_view_leg() {
        let mut point = [0.0; 2];
        let mut largest = 0.0;
        let mut active = 0;
        for x in 0..128 {
            for y in 0..128 {
                let p = [x as f32 / 127.0, y as f32 / 127.0];
                let m = bed_material(p);
                assert!((0.10..=0.90).contains(&m[0]));
                assert!(
                    m[1..]
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=0.32).contains(v))
                );
                active += usize::from(m[3] > 0.02);
                if m[3] > largest {
                    largest = m[3];
                    point = p;
                }
            }
        }
        assert!(
            largest > 0.15 && active > 100 && active < 5000,
            "vein coverage {active}, peak {largest}"
        );
        let mut sky = SkyEnvironment::constant((0.0, 0.0, 0.0));
        sky.bed_emission_strength = 1.0;
        let mut scene = SceneFrame::new(SceneKind::Acid, 0.0, 2.0);
        scene.absorption_scale = 0.0;
        let pigment = vec![[0.4, 0.2, 0.1]; 64];
        let material = MaterialGrid {
            dye: Some(&pigment),
            ..MaterialGrid::clear()
        };
        let shade = |m| {
            submerged_bed_radiance(
                m,
                &sky,
                8,
                8,
                point,
                0.16,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                &scene,
            )
        };
        let clear = shade(MaterialGrid::clear());
        let dyed = shade(material);
        let attenuation = crate::pigment_optics::transmittance(pigment[0], 1.0);
        for c in 0..3 {
            assert!(
                (clear[c] - bed_material(point)[c + 1] * (-WATER_SCATTER * 0.16).exp()).abs()
                    < 1e-6
            );
            assert!((dyed[c] - clear[c] * attenuation[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn moving_water_history_tracks_metres_and_stops_with_physical_time() {
        let point = [0.5, 0.5];
        let footprint = SampleFootprint::new(2, 2, point[0], point[1]);
        let velocity = [[1.0, -0.4]; 4];
        assert_eq!(
            backtrack_surface(point, footprint, &velocity, 0.0, 1.0),
            point
        );
        assert_eq!(
            backtrack_surface(point, footprint, &velocity, 0.5, 0.0),
            point
        );
        let previous = backtrack_surface(point, footprint, &velocity, 0.5, 1.0);
        assert!((previous[0] - 0.25).abs() < 1e-6);
        assert!((previous[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn water_reflectance_at_normal_incidence_matches_its_refractive_index() {
        let expected = ((IOR_WATER - 1.0) / (IOR_WATER + 1.0)).powi(2);
        assert!((fresnel_scalar(1.0, IOR_WATER) - expected).abs() < 1e-6);
        assert_eq!(fresnel_scalar(0.1, 1.0 / IOR_WATER), 1.0);
        assert_eq!(fresnel_scalar(0.0, IOR_WATER), 1.0);
        assert_eq!(smith_ggx(0.0, 0.5, 0.1), 0.0);
        assert!((smith_ggx(1.0, 1.0, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bottom_caustics_redistribute_flux_and_flat_water_stays_uniform() {
        let flat = Array2::zeros((64, 40));
        let mut intensity = Vec::new();
        precompute_caustics(&flat, None, &mut intensity);
        assert!(intensity.iter().all(|value| (*value - 1.0).abs() < 1e-4));
        let curved = Array2::from_shape_fn((64, 40), |(x, y)| {
            (x as f32 / 64.0 * 8.0).sin() * (y as f32 / 40.0 * 6.0).cos() * 0.015
        });
        precompute_caustics(&curved, None, &mut intensity);
        assert!(
            intensity
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(intensity.iter().any(|value| *value > 1.05));
        assert!(intensity.iter().any(|value| *value < 0.95));
        assert!((intensity.iter().sum::<f32>() / intensity.len() as f32 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shallow_columns_reduce_caustic_focusing_without_changing_total_flux() {
        let levels = Array2::from_shape_fn((64, 40), |(x, y)| {
            (x as f32 / 64.0 * 8.0).sin() * (y as f32 / 40.0 * 6.0).cos() * 0.001
        });
        let mut deep = Vec::new();
        let mut shallow = Vec::new();
        let depths = vec![0.04; levels.len()];
        precompute_caustics(&levels, None, &mut deep);
        precompute_caustics(&levels, Some(&depths), &mut shallow);
        let variance = |values: &[f32]| {
            values
                .iter()
                .map(|value| (value - 1.0).powi(2))
                .sum::<f32>()
        };
        assert!(variance(&shallow) < variance(&deep) * 0.02);
        assert!((shallow.iter().sum::<f32>() / shallow.len() as f32 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn primary_rays_intersect_the_displaced_surface() {
        let scene = SceneFrame::new(SceneKind::Acid, 12.0, 1.0);
        let camera = Camera::new(12.0, 120, 40, &scene);
        for (a, b, c) in [(0.0, 0.0, 0.05), (0.12, -0.08, -0.02)] {
            let levels = Array2::from_shape_fn((80, 40), |(x, y)| {
                a * x as f32 * PATCH_LENGTH / 80.0 + b * y as f32 * PATCH_LENGTH / 40.0 + c
            });
            let bounds = levels
                .iter()
                .fold([f32::INFINITY, f32::NEG_INFINITY], |z, h| {
                    [z[0].min(*h), z[1].max(*h)]
                });
            for (u, v) in [(0.3, 0.3), (0.5, 0.5), (0.7, 0.6)] {
                let dir = camera.ray(u, v);
                let (wp, t) = camera.hit_surface(dir, &levels, bounds).unwrap();
                let actual_z = camera.pos[2] + t * dir[2];
                assert!((actual_z - sample(&levels, wp[0], wp[1])).abs() < 1e-5);
                if c == 0.05 {
                    assert!((actual_z - c).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn camera_footprint_stays_inside_both_height_planes_through_rotation() {
        let worlds = [
            SceneKind::Journey,
            SceneKind::Acid,
            SceneKind::Aurora,
            SceneKind::Storm,
            SceneKind::Abyss,
            SceneKind::Lava,
            SceneKind::Prism,
        ];
        let sizes = [
            (200, 60),
            (80, 24),
            (80, 40),
            (80, 80),
            (160, 24),
            (1, 1),
            (1, 80),
            (400, 1),
        ];
        for world in worlds {
            for elapsed in (0..=540).step_by(3) {
                let scene = SceneFrame::new(world, elapsed as f32, 2.0);
                for (width, height) in sizes {
                    for bounds in [[0.0, 0.0], [-0.35, 0.20], [-0.9, 0.6]] {
                        let original = Camera::new(elapsed as f32, width, height, &scene);
                        let mut camera = original;
                        camera.fit_surface(bounds);
                        assert_eq!(camera.pos, original.pos);
                        assert_eq!(camera.ground, original.ground);
                        assert_eq!(camera.fwd, original.fwd);
                        assert_eq!(camera.right, original.right);
                        assert_eq!(camera.up, original.up);
                        assert!(
                            (camera.half_w / camera.half_h - original.half_w / original.half_h)
                                .abs()
                                < 0.0001 * (1.0 + original.half_w / original.half_h)
                        );
                        for level in bounds {
                            let plane = Array2::from_elem((2, 2), level);
                            for u in [0.0, 0.5, 1.0] {
                                for v in [0.0, 0.5, 1.0] {
                                    let hit = camera
                                        .hit_surface(camera.ray(u, v), &plane, [level, level]);
                                    let (point, _) = hit.unwrap_or_else(|| {
                                        panic!(
                                            "{world:?} t={elapsed} size={width}x{height} \
                                             bounds={bounds:?} uv=({u},{v}) exposes the patch edge"
                                        )
                                    });
                                    assert!(
                                        point
                                            .into_iter()
                                            .all(|coordinate| (0.02999..=0.97001)
                                                .contains(&coordinate)),
                                        "camera lost its edge guard: {point:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn translating_look_target_moves_over_a_stationary_surface() {
        let scene = SceneFrame::new(SceneKind::Journey, 0.0, 4.0);
        let first = Camera::new(0.0, 120, 40, &scene);
        let mut moved = first;
        moved.pos[0] += 0.2;
        moved.ground[0] += 0.2;
        let a = first.hit_water(first.ray(0.5, 0.5)).unwrap();
        let b = moved.hit_water(moved.ray(0.5, 0.5)).unwrap();
        assert!(
            (b[0] - a[0] - 0.2 / PATCH_LENGTH).abs() < 1e-5,
            "translation must move the view across water, not carry water with it"
        );
    }

    #[test]
    fn streaming_origin_shift_reprojects_the_same_world_point() {
        let scene = SceneFrame::new(SceneKind::Journey, 0.0, 4.0);
        for base in [0.0, -170.0, 1_000_000_000.0] {
            let mut previous = Camera::new(0.0, 120, 40, &scene);
            previous.place_in_world([base, base], [base - 1.0, base - 1.0]);
            let mut current = Camera::new(1.0, 120, 40, &scene);
            current.place_in_world([base + 0.2, base], [base - 0.75, base - 1.0]);
            // Same world point [base + .125, base + .25], represented in current UV.
            let actual = current
                .history_direction(&previous, [0.4375, 0.625], 0.04)
                .unwrap();
            let expected = [
                0.125 - previous.pos[0],
                0.25 - previous.pos[1],
                0.04 - previous.pos[2],
            ];
            for axis in 0..3 {
                assert!((actual[axis] - expected[axis]).abs() < 1e-6);
            }
            let center = current.hit_water(current.ray(0.5, 0.5)).unwrap();
            assert!((center[0] - 0.475).abs() < 1e-6);
        }
    }

    #[test]
    fn newly_entered_water_has_no_previous_window_history() {
        let scene = SceneFrame::new(SceneKind::Journey, 0.0, 4.0);
        let mut previous = Camera::new(0.0, 120, 40, &scene);
        previous.place_in_world([0.0; 2], [-1.0; 2]);
        let mut current = previous;
        current.place_in_world([0.5, 0.0], [-0.5, -1.0]);
        assert!(
            current
                .history_direction(&previous, [0.9, 0.5], 0.0)
                .is_none()
        );
        assert!(
            current
                .history_direction(&previous, [0.5, 0.5], 0.0)
                .is_some()
        );
    }

    #[test]
    fn streamed_camera_keeps_edges_hidden_with_cell_sized_residuals() {
        for time in (0..540).step_by(13) {
            let scene = SceneFrame::new(SceneKind::Journey, time as f32, 4.0);
            for (width, height) in [(200, 60), (60, 100), (320, 100), (1, 1)] {
                let mut camera = Camera::new(time as f32, width, height, &scene);
                camera.place_in_world([1000.018, -499.974], [999.0, -501.0]);
                camera.fit_surface([-0.15, 0.10]);
                for z in [-0.15, 0.10] {
                    let plane = Array2::from_elem((16, 16), z);
                    for u in [0.0, 0.5, 1.0] {
                        for v in [0.0, 0.5, 1.0] {
                            let (point, _) = camera
                                .hit_surface(camera.ray(u, v), &plane, [z, z])
                                .unwrap();
                            assert!(point.into_iter().all(|p| (0.029..=0.971).contains(&p)));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn renderer_fits_the_live_deep_bowl_before_shading() {
        let mut water = Water::new(80, 80);
        let shape = water.levels.raw_dim();
        water.levels = Array2::from_shape_fn(shape, |(x, y)| {
            let px = x as f32 / shape[0] as f32 - 0.5;
            let py = y as f32 / shape[1] as f32 - 0.5;
            -0.45 * (-(px * px + py * py) / 0.04).exp()
        });
        let bounds = water
            .levels
            .iter()
            .fold([f32::INFINITY, f32::NEG_INFINITY], |b, &z| {
                [b[0].min(z), b[1].max(z)]
            });
        let mut renderer = Renderer::new(80, 80);
        renderer.set_scene(SceneKind::Prism);
        renderer.set_quality(RenderQuality::Balanced);
        renderer
            .draw_at(&mut std::io::sink(), &water, 228.0, 1.0 / 60.0)
            .unwrap();
        let camera = renderer
            .prev_cam
            .expect("the rendered camera must enter history");
        for y in 0..=20 {
            for x in 0..=20 {
                let dir = camera.ray(x as f32 / 20.0, y as f32 / 20.0);
                assert!(
                    camera.hit_surface(dir, &water.levels, bounds).is_some(),
                    "rendered deep bowl exposes the finite patch at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn rising_and_horizontal_secondary_rays_hit_a_steeper_surface() {
        let scene = SceneFrame::new(SceneKind::Acid, 0.0, 1.0);
        let mut camera = Camera::new(0.0, 80, 24, &scene);
        camera.ground = [0.0; 2];
        camera.patch_center = [0.0; 2];
        camera.pos = [-0.5, 0.0, -0.04];
        let levels = Array2::from_shape_fn((80, 40), |(x, _)| {
            0.1 * (x as f32 * PATCH_LENGTH / 80.0 - PATCH_LENGTH * 0.5)
        });
        for dz in [0.0, 0.05] {
            let dir = norm3([1.0, 0.0, dz]);
            let (wp, t) = camera.hit_surface(dir, &levels, [-0.10, 0.10]).unwrap();
            assert!((camera.pos[2] + t * dir[2] - sample(&levels, wp[0], wp[1])).abs() < 1e-5);
            assert!(
                (camera.pos[0] + t * dir[0] - if dz == 0.0 { -0.4 } else { -0.3 }).abs() < 1e-4
            );
        }
    }

    #[test]
    fn reflection_budgets_preserve_the_sky_tail_and_bound_the_passive_bed() {
        let mut scene = SceneFrame::new(SceneKind::Prism, 0.0, 1.0);
        scene.absorption_scale = 0.0;
        let mut camera = Camera::new(0.0, 80, 24, &scene);
        camera.ground = [0.0; 2];
        camera.patch_center = [0.0; 2];
        let levels = Array2::from_shape_fn((80, 40), |(x, _)| {
            0.1 * (x as f32 * PATCH_LENGTH / 80.0 - PATCH_LENGTH * 0.5)
        });
        let mut normals = Vec::new();
        precompute_normals(&levels, &mut normals);
        let expected = (0.3, 0.6, 0.9);
        let sky = SkyEnvironment::constant(expected);
        for direction in [norm3([1.0, 0.0, 0.05]), [0.0, 0.0, 1.0]] {
            for limit in 0..=3 {
                let result = trace_reflections(
                    &camera,
                    &levels,
                    MaterialGrid::clear(),
                    &normals,
                    &sky,
                    [0.25, 0.5],
                    -0.04,
                    [0.0, 0.0, 1.0],
                    direction,
                    &scene,
                    [-0.10, 0.10],
                    surface_slope_bound(&levels),
                    limit,
                );
                if limit == 0 || direction[2] > 0.5 {
                    // An escaping/unresolved reflection still returns the exact sky.
                    assert!((result.0 - expected.0).abs() < 1e-6);
                    assert!((result.1 - expected.1).abs() < 1e-6);
                    assert!((result.2 - expected.2).abs() < 1e-6);
                } else {
                    // A resolved hit now sees a passive diffuse bed (albedo < 1),
                    // so a uniform sky is no longer an equal-radiance lower boundary.
                    for (actual, source) in [result.0, result.1, result.2]
                        .into_iter()
                        .zip([expected.0, expected.1, expected.2])
                    {
                        assert!(actual > 0.0 && actual < source);
                    }
                }
            }
        }
        // This path must actually encounter the ramp: absorbing transport then
        // differs from a zero-bounce tail, rather than vacuously passing above.
        scene.absorption_scale = 1.0;
        let absorbed = trace_reflections(
            &camera,
            &levels,
            MaterialGrid::clear(),
            &normals,
            &sky,
            [0.25, 0.5],
            -0.04,
            [0.0, 0.0, 1.0],
            norm3([1.0, 0.0, 0.05]),
            &scene,
            [-0.10, 0.10],
            surface_slope_bound(&levels),
            3,
        );
        assert!(absorbed.2 < expected.2 * 0.99, "absorbed={absorbed:?}");
        let depths = vec![0.02; levels.len()];
        let shallow = trace_reflections(
            &camera,
            &levels,
            MaterialGrid {
                depths: Some(&depths),
                ..MaterialGrid::clear()
            },
            &normals,
            &sky,
            [0.25, 0.5],
            -0.04,
            [0.0, 0.0, 1.0],
            norm3([1.0, 0.0, 0.05]),
            &scene,
            [-0.10, 0.10],
            surface_slope_bound(&levels),
            3,
        );
        assert!(shallow.2 > absorbed.2 && shallow.2 < expected.2);
    }

    #[test]
    fn secondary_foam_replaces_water_transport_and_zero_coverage_is_exact() {
        let scene = SceneFrame::new(SceneKind::Prism, 0.0, 1.0);
        let mut camera = Camera::new(0.0, 80, 24, &scene);
        camera.ground = [0.0; 2];
        camera.patch_center = [0.0; 2];
        let levels = Array2::from_shape_fn((80, 40), |(x, _)| {
            0.1 * (x as f32 * PATCH_LENGTH / 80.0 - PATCH_LENGTH * 0.5)
        });
        let mut normals = Vec::new();
        precompute_normals(&levels, &mut normals);
        let sky = SkyEnvironment::constant((0.3, 0.6, 0.9));
        let lighting = FoamLighting::from_sky(&sky);
        let foam_fields = [0.0, 0.35, 1.0].map(|coverage| vec![coverage; levels.len()]);
        let trace = |material, limit| {
            let result = trace_reflections(
                &camera,
                &levels,
                material,
                &normals,
                &sky,
                [0.25, 0.5],
                -0.04,
                [0.0, 0.0, 1.0],
                norm3([1.0, 0.0, 0.05]),
                &scene,
                [-0.10, 0.10],
                surface_slope_bound(&levels),
                limit,
            );
            [result.0, result.1, result.2]
        };
        let normal = norm3([-0.1, 0.0, 1.0]);
        let opaque = lighting.radiance(normal, dot3(normal, scene.light_dir));
        let zero_dye = vec![[0.0; 3]; levels.len()];
        let colored_dye = vec![[0.2, 0.8, 0.1]; levels.len()];
        for limit in 1..=3 {
            let clear = trace(MaterialGrid::clear(), limit);
            assert_eq!(
                clear,
                trace(
                    MaterialGrid {
                        dye: Some(&zero_dye),
                        ..MaterialGrid::clear()
                    },
                    limit
                )
            );
            let tinted = trace(
                MaterialGrid {
                    dye: Some(&colored_dye),
                    ..MaterialGrid::clear()
                },
                limit,
            );
            for channel in 0..3 {
                assert!(tinted[channel] >= 0.0 && tinted[channel] < clear[channel]);
            }
            let dyed_foam = trace(
                MaterialGrid {
                    foam: Some(&foam_fields[2]),
                    dye: Some(&colored_dye),
                    foam_light: Some(&lighting),
                    ..MaterialGrid::clear()
                },
                limit,
            );
            for channel in 0..3 {
                assert!(dyed_foam[channel] >= 0.0 && dyed_foam[channel] < opaque[channel]);
            }
            assert!((clear[2] - opaque[2]).abs() > 0.01);
            for (coverage, foam) in [0.0, 0.35, 1.0].into_iter().zip(&foam_fields) {
                let actual = trace(
                    MaterialGrid {
                        foam: Some(foam),
                        foam_light: Some(&lighting),
                        ..MaterialGrid::clear()
                    },
                    limit,
                );
                if coverage == 0.0 {
                    assert_eq!(actual, clear);
                }
                for channel in 0..3 {
                    let expected = clear[channel] * (1.0 - coverage) + opaque[channel] * coverage;
                    assert!((actual[channel] - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn scene_crossfades_mix_linear_light() {
        assert_eq!(
            mix_display_color((0, 0, 0), (255, 255, 255), 0.5),
            (188, 188, 188)
        );
    }

    #[test]
    fn foam_is_a_bounded_coverage_mixture_and_clear_water_is_identical() {
        let water = [0.08, 1.7, 0.31];
        let foam = [0.6, 0.9, 0.8];
        assert_eq!(mix_foam(water, foam, 0.0), water);
        assert_eq!(mix_foam(water, foam, 1.0), foam);
        for coverage in [0.01, 0.25, 0.5, 0.9] {
            let mixed = mix_foam(water, foam, coverage);
            for channel in 0..3 {
                assert!(mixed[channel] >= water[channel].min(foam[channel]));
                assert!(mixed[channel] <= water[channel].max(foam[channel]));
                assert!(
                    (mixed[channel]
                        - (water[channel] * (1.0 - coverage) + foam[channel] * coverage))
                        .abs()
                        < 2e-7
                );
            }
        }
    }

    #[test]
    fn diffuse_foam_preserves_uniform_environment_and_is_dark_without_light() {
        for incident in [[0.0; 3], [0.3, 0.6, 0.9]] {
            let sky = SkyEnvironment::constant((incident[0], incident[1], incident[2]));
            let foam = FoamLighting::from_sky(&sky);
            for normal in [[0.0, 0.0, 1.0], norm3([0.8, -0.5, 1.0]), [-1.0, 0.0, 0.0]] {
                let outgoing = foam.radiance(normal, 0.0);
                for channel in 0..3 {
                    assert!((outgoing[channel] - incident[channel] * FOAM_ALBEDO).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn actual_column_depth_controls_optical_attenuation() {
        let footprint = SampleFootprint::new(2, 2, 0.25, 0.25);
        let depths = [0.04, 0.08, 0.12, 0.16];
        let material = MaterialGrid {
            depths: Some(&depths),
            ..MaterialGrid::clear()
        };
        let depth = material.depth(footprint, 0.3);
        assert!((depth - 0.1).abs() < 1e-7);
        let shallow = beer_lambert(depth);
        let classic = beer_lambert(MaterialGrid::clear().depth(footprint, 0.3));
        assert!(
            shallow
                .iter()
                .zip(classic)
                .all(|(shallow, deep)| *shallow > deep)
        );
        assert_eq!(
            MaterialGrid::clear().depth(footprint, 0.3),
            (DEPTH + 0.3).max(0.001)
        );
    }

    #[test]
    fn dissolved_pigments_are_passive_and_use_column_integrated_mass() {
        assert_eq!(dye_transmittance([0.0; 3], 1.0), [1.0; 3]);
        assert_eq!(dye_transmittance([0.2, 0.3, 0.4], 0.0), [1.0; 3]);
        let dye = [0.2, 0.3, 0.4];
        let normal = dye_transmittance(dye, 1.0);
        let slanted = dye_transmittance(dye, 1.4);
        let twice_the_mass = dye_transmittance(dye.map(|c| c * 2.0), 1.0);
        for channel in 0..3 {
            assert!(normal[channel] > 0.0 && normal[channel] < 1.0);
            assert!(slanted[channel] < normal[channel]);
            assert!((twice_the_mass[channel] - normal[channel].powi(2)).abs() < 1e-6);
        }
        // Column depth cancels when an areal pigment mass is uniformly mixed:
        // concentration_per_volume * path = areal_mass * path / depth.
        assert_eq!(
            dye_transmittance(dye, 0.10 / 0.05),
            dye_transmittance(dye, 1.0 / 0.5)
        );
        let cyan = dye_transmittance([1.0, 0.0, 0.0], 1.0);
        let magenta = dye_transmittance([0.0, 1.0, 0.0], 1.0);
        let amber = dye_transmittance([0.0, 0.0, 1.0], 1.0);
        assert!(cyan[0] < cyan[1] && cyan[0] < cyan[2]);
        assert!(magenta[1] < magenta[0] && magenta[1] < magenta[2]);
        assert!(amber[2] < amber[0] && amber[2] < amber[1]);
    }

    #[test]
    fn primary_transport_uses_column_depth_and_zero_foam_does_not_change_it() {
        let scene = SceneFrame::new(SceneKind::Prism, 0.0, 1.0);
        let camera = Camera::new(0.0, 80, 24, &scene);
        let levels = Array2::zeros((8, 8));
        let normals = vec![[0.0, 0.0, 1.0, 0.0]; 64];
        let caustics = vec![1.0; 64];
        let sky = SkyEnvironment::constant((0.3, 0.6, 0.9));
        let foam_light = FoamLighting::from_sky(&sky);
        let shade = |material| {
            shade_water(
                &levels,
                material,
                &normals,
                &caustics,
                &sky,
                8,
                8,
                scene.light_dir,
                &camera,
                [0.0, 0.0, -1.0],
                [0.5, 0.5],
                &scene,
                0.0,
                [0.0; 2],
                0.0,
                0,
            )
        };
        let clear = shade(MaterialGrid::clear());
        let foam = vec![0.0; 64];
        let depths = vec![DEPTH; 64];
        assert_eq!(
            clear,
            shade(MaterialGrid {
                depths: Some(&depths),
                foam: Some(&foam),
                foam_light: Some(&foam_light),
                ..MaterialGrid::clear()
            })
        );
        let shallow = vec![0.04; 64];
        let transmitted = shade(MaterialGrid {
            depths: Some(&shallow),
            ..MaterialGrid::clear()
        });
        for channel in 0..3 {
            assert!(transmitted[channel] > clear[channel]);
        }
        let zero_dye = vec![[0.0; 3]; 64];
        assert_eq!(
            clear,
            shade(MaterialGrid {
                dye: Some(&zero_dye),
                ..MaterialGrid::clear()
            })
        );
        let colored_dye = vec![[0.2, 0.8, 0.1]; 64];
        let tinted = shade(MaterialGrid {
            dye: Some(&colored_dye),
            ..MaterialGrid::clear()
        });
        for channel in 0..3 {
            assert!(tinted[channel] >= 0.0 && tinted[channel] < clear[channel]);
        }
        let opaque_foam = vec![1.0; 64];
        let foam_material = MaterialGrid {
            foam: Some(&opaque_foam),
            foam_light: Some(&foam_light),
            ..MaterialGrid::clear()
        };
        let white_foam = shade(foam_material);
        assert_eq!(
            white_foam,
            shade(MaterialGrid {
                dye: Some(&zero_dye),
                ..foam_material
            })
        );
        let dyed_foam = shade(MaterialGrid {
            dye: Some(&colored_dye),
            ..foam_material
        });
        for channel in 0..3 {
            assert!(dyed_foam[channel] >= 0.0 && dyed_foam[channel] < white_foam[channel]);
        }
    }

    #[test]
    fn surface_normals_match_metric_slopes_at_every_resolution() {
        for (width, height) in [(16, 16), (80, 40), (240, 120)] {
            let levels = Array2::from_shape_fn((width, height), |(x, y)| {
                0.1 * x as f32 * WATER_SCALE / width as f32
                    - 0.2 * y as f32 * WATER_SCALE / height as f32
            });
            let mut normals = Vec::new();
            precompute_normals(&levels, &mut normals);
            let expected = norm3([-0.1, 0.2, 1.0]);
            let actual = normals[(width / 2) * height + height / 2];
            for i in 0..3 {
                assert!((actual[i] - expected[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn distributed_water_light_uses_volume_escape_instead_of_a_back_wall_filter() {
        let length = 0.4;
        let water = WaterTransport::new(length, 1.0, WATER_SCATTER);
        let pigment = vec![[0.4, 0.3, 0.2]; 4];
        let material = MaterialGrid {
            dye: Some(&pigment),
            ..MaterialGrid::clear()
        };
        let dye = material.dye_transport(
            SampleFootprint::new(2, 2, 0.5, 0.5),
            1.0,
            water.tau,
            water.transmission,
        );
        let extinction = [0.94_f64, 0.93, 0.615];
        for c in 0..3 {
            let total_tau =
                (ABSORB[c] as f64 + WATER_SCATTER as f64) * length as f64 + extinction[c] * 0.85;
            let quadrature = (0..4096)
                .map(|i| (-total_tau * (i as f64 + 0.5) / 4096.0).exp())
                .sum::<f64>()
                / 4096.0;
            let expected = WATER_SCATTER as f64 * length as f64 * quadrature;
            let actual = water.scatter[c] * dye.volume[c];
            assert!(
                (actual as f64 - expected).abs() < 2e-9,
                "channel{c}: {actual} vs{expected}"
            );
            assert!(actual > water.scatter[c] * dye.transmission[c]);
        }
    }

    #[test]
    fn scattering_has_physical_thin_thick_and_absorbing_limits() {
        assert_eq!(WaterTransport::new(100.0, 1.0, 0.0).scatter, [0.0; 3]);
        assert_eq!(
            WaterTransport::new(0.0, 1.0, WATER_SCATTER).transmission,
            [1.0; 3]
        );
        let thin = WaterTransport::new(1e-5, 0.2, WATER_SCATTER);
        for c in 0..3 {
            assert!((thin.scatter[c] / (WATER_SCATTER * 1e-5) - 1.0).abs() < 1e-5);
        }
        let mut prior = [1.0; 3];
        for absorption in [0.0, 0.2, 1.0, 10.0] {
            let transport = WaterTransport::new(0.7, absorption, WATER_SCATTER);
            let thick = WaterTransport::new(f32::INFINITY, absorption, WATER_SCATTER);
            for c in 0..3 {
                assert!(transport.scatter[c] <= prior[c]);
                assert!(transport.scatter[c] + transport.transmission[c] <= 1.000001);
                assert!(
                    (thick.scatter[c] - WATER_SCATTER / (ABSORB[c] * absorption + WATER_SCATTER))
                        .abs()
                        < 1e-6
                );
            }
            prior = transport.scatter;
        }
    }

    #[test]
    fn rose_glass_absorption_matches_beer_lambert_and_fades_monotonically() {
        let coefficients = [1.2_f32, 1.6, 1.2];
        let mut previous = [1.0; 3];
        for depth in [0.0, 0.01, 0.1, 0.7, 1.5, 4.0, 100.0, f32::INFINITY] {
            let transmitted = beer_lambert(depth);
            for channel in 0..3 {
                assert!(transmitted[channel].is_finite());
                assert!((0.0..=previous[channel]).contains(&transmitted[channel]));
                let expected = (-coefficients[channel] * depth).exp();
                assert!((transmitted[channel] - expected).abs() < 1e-6);
            }
            assert_eq!(transmitted[0], transmitted[2]);
            previous = transmitted;
        }
    }

    #[test]
    fn manual_scene_switches_and_repeated_switches_begin_at_the_visible_frame() {
        let water = Water::new(24, 12);
        let mut renderer = Renderer::new(24, 12);
        renderer
            .draw_at(&mut Vec::new(), &water, 12.0, 0.1)
            .unwrap();
        let before = renderer.snapshot_rgb();
        renderer.set_scene(SceneKind::Prism);
        renderer
            .draw_at(&mut Vec::new(), &water, 12.0, 0.0)
            .unwrap();
        assert!(
            renderer.snapshot_rgb() == before,
            "a paused switch must not hard-cut"
        );
        for _ in 0..7 {
            renderer
                .draw_at(&mut Vec::new(), &water, 12.0, 0.1)
                .unwrap();
        }
        let mixed = renderer.snapshot_rgb();
        assert_ne!(
            mixed, before,
            "the transition must progress while time advances"
        );
        renderer.set_scene(SceneKind::Lava);
        renderer
            .draw_at(&mut Vec::new(), &water, 12.0, 0.0)
            .unwrap();
        assert!(
            renderer.snapshot_rgb() == mixed,
            "a repeated switch must start at the current blend"
        );
        for _ in 0..19 {
            renderer
                .draw_at(&mut Vec::new(), &water, 12.0, 0.1)
                .unwrap();
        }
        assert!(
            renderer.scene_transition.is_some(),
            "the full two-second blend must be retained"
        );
        renderer
            .draw_at(&mut Vec::new(), &water, 12.0, 0.11)
            .unwrap();
        assert!(
            renderer.scene_transition.is_none(),
            "completed transitions must release the captured frame"
        );
    }

    #[test]
    fn extra_intensity_reaches_the_rendered_image() {
        let water = Water::new(24, 16);
        let image = |dose: f32| {
            let mut renderer = Renderer::new(24, 12);
            renderer.set_intensity(dose);
            for frame in 0..24 {
                renderer
                    .draw_at(&mut Vec::new(), &water, 12.0 + frame as f32 * 0.1, 0.1)
                    .unwrap();
            }
            renderer.snapshot_rgb()
        };
        assert!(
            image(2.0) != image(4.0),
            "extra dose must not be silently clamped"
        );
    }

    #[test]
    fn intensity_changes_keep_temporal_history_and_exposure() {
        let water = Water::new(16, 8);
        let mut renderer = Renderer::new(16, 8);
        renderer
            .draw_at(&mut Vec::new(), &water, 12.0, 0.1)
            .unwrap();
        let exposure = renderer.adapted_exposure;
        let history = renderer.prev_hdr_buf.clone();
        renderer.set_intensity(1.4);
        assert!(renderer.prev_cam.is_some());
        assert_eq!(renderer.prev_hdr_buf, history);
        assert_eq!(renderer.adapted_exposure, exposure);
    }

    #[test]
    fn spatial_resolve_softens_subpixel_edges_without_blurring_flat_areas() {
        let mut renderer = Renderer::new(7, 4);
        renderer.prepare_buffers();
        renderer.hdr_buf.fill([0.2; 3]);
        renderer.resolve_spatial_edges();
        assert!(renderer.hdr_buf.iter().all(|pixel| *pixel == [0.2; 3]));
        let center = 3 * renderer.render_width + 3;
        renderer.hdr_buf[center] = [2.0; 3];
        renderer.resolve_spatial_edges();
        assert!(renderer.hdr_buf[center][0] < 2.0);
        assert!(renderer.hdr_buf[center - 1][0] > 0.2);
        assert_eq!(renderer.hdr_buf[0], [0.2; 3]);
    }

    #[test]
    fn a_frozen_scene_settles_without_pixel_shimmer() {
        let water = Water::new(40, 20);
        for quality in [
            RenderQuality::High,
            RenderQuality::Balanced,
            RenderQuality::Low,
        ] {
            let mut renderer = Renderer::new(32, 12);
            renderer.set_quality(quality);
            // This contract isolates physical reconstruction. A subjective
            // layer deliberately keeps drifting while active wall time passes.
            renderer.set_subjective_intensity(0.0);
            // Retinal light approaches its limit exponentially. A tiny monotonic
            // residual can cross a byte threshold long after it looks settled,
            // so wait for a bounded float-state fixed point before testing shimmer.
            let mut steady_frames = 0;
            for _ in 0..600 {
                let previous_hdr = renderer.hdr_buf.clone();
                let previous_exposure = renderer.adapted_exposure;
                renderer
                    .draw_at(&mut Vec::new(), &water, 12.5, 0.1)
                    .unwrap();
                if renderer.hdr_buf == previous_hdr
                    && renderer.adapted_exposure == previous_exposure
                {
                    steady_frames += 1;
                } else {
                    steady_frames = 0;
                }
                if steady_frames == 32 {
                    break;
                }
            }
            assert_eq!(
                steady_frames, 32,
                "retinal light never settles in {quality:?}"
            );
            let settled = renderer.snapshot_rgb();
            for _ in 0..64 {
                renderer
                    .draw_at(&mut Vec::new(), &water, 12.5, 0.1)
                    .unwrap();
                let changed = renderer
                    .snapshot_rgb()
                    .iter()
                    .zip(&settled)
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(changed, 0, "frozen scene still shimmers in {quality:?}");
            }
        }
    }

    #[test]
    fn quality_changes_preserve_uniform_temporal_history() {
        let mut renderer = Renderer::new(13, 7);
        renderer.prepare_buffers();
        renderer.prev_hdr_buf.fill([0.3, 0.6, 0.9]);
        renderer.prev_cam = Some(Camera::new(
            1.0,
            13,
            7,
            &SceneFrame::new(SceneKind::Acid, 1.0, 1.0),
        ));
        for quality in [
            RenderQuality::Balanced,
            RenderQuality::Low,
            RenderQuality::High,
        ] {
            renderer.set_quality(quality);
            renderer.prepare_buffers();
            assert!(
                renderer.prev_cam.is_some(),
                "quality changes must preserve valid history"
            );
            for color in &renderer.prev_hdr_buf {
                for (actual, expected) in color.iter().zip([0.3, 0.6, 0.9]) {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn unresolved_normals_and_larger_pixels_broaden_specular_highlights() {
        let smooth = filtered_roughness_squared([0.0, 0.0, 1.0, 0.0], 1.0);
        let unresolved = filtered_roughness_squared([0.0, 0.0, 0.7, 0.0], 1.0);
        assert!(unresolved > smooth * 2.0);
        let detailed = [0.0, 0.0, 1.0, 0.02];
        assert!(
            filtered_roughness_squared(detailed, 4.0) > filtered_roughness_squared(detailed, 1.0)
        );
        assert!(filtered_roughness_squared(detailed, 100.0) <= 1.0);
    }

    #[test]
    fn scoped_shading_matches_serial_rgb_and_terminal_bytes_exactly() {
        for rapids in [false, true] {
            let mut water = Water::new(40, 30);
            water.set_rapids(rapids);
            water.update(std::time::Duration::from_millis(500));
            for quality in [RenderQuality::High, RenderQuality::Balanced] {
                let mut serial = Renderer::new(120, 50);
                let mut parallel = Renderer::new(120, 50);
                serial.worker_count = 1;
                parallel.worker_count = 4;
                serial.set_quality(quality);
                parallel.set_quality(quality);
                for frame in 0..2 {
                    let time = 11.0 + frame as f32 / 60.0;
                    let mut serial_bytes = Vec::new();
                    let mut parallel_bytes = Vec::new();
                    serial
                        .draw_at(&mut serial_bytes, &water, time, 1.0 / 60.0)
                        .unwrap();
                    parallel
                        .draw_at(&mut parallel_bytes, &water, time, 1.0 / 60.0)
                        .unwrap();
                    assert_eq!(serial.snapshot_rgb(), parallel.snapshot_rgb());
                    assert_eq!(serial_bytes, parallel_bytes);
                }
            }
        }
    }

    #[test]
    fn primary_hits_refresh_after_camera_quality_and_size_changes() {
        let mut renderer = Renderer::new(17, 9);
        let water = Water::new(31, 17);
        for (quality, width, height, time) in [
            (RenderQuality::Balanced, 17, 9, 1.0),
            (RenderQuality::Low, 13, 7, 8.0),
            (RenderQuality::High, 13, 7, 23.0),
            (RenderQuality::Balanced, 120, 50, 80.0),
            (RenderQuality::Low, 1, 1, 112.0),
        ] {
            renderer.primary_hits.fill(Some(([f32::NAN; 2], f32::NAN)));
            renderer.set_quality(quality);
            renderer.resize(width, height);
            renderer.render_at(&water, time, 1.0 / 60.0).unwrap();
            let w = renderer.render_width;
            let h = renderer.render_height;
            assert_eq!(renderer.primary_hits.len(), w * h);
            // Check the cache against the actual smoothed view; requested scene
            // time deliberately jumps here while presentation remains continuous.
            let camera = renderer.prev_cam.expect("rendered camera");
            let bounds = water
                .levels
                .iter()
                .fold([f32::INFINITY, f32::NEG_INFINITY], |b, z| {
                    [b[0].min(*z), b[1].max(*z)]
                });
            for y in 0..h {
                for x in 0..w {
                    let expected = if quality == RenderQuality::High {
                        None
                    } else {
                        let ray =
                            camera.ray((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                        camera.hit_surface(ray, &water.levels, bounds)
                    };
                    assert_eq!(renderer.primary_hits[y * w + x], expected);
                }
            }
        }
    }

    #[test]
    fn empty_water_does_not_present_a_stale_rendered_frame() {
        let mut renderer = Renderer::new(13, 7);
        let water = Water::new(17, 9);
        renderer
            .draw_at(&mut Vec::new(), &water, 1.0, 1.0 / 60.0)
            .unwrap();
        renderer.invalidate();
        let mut output = Vec::new();
        renderer
            .draw_at(&mut output, &Water::new(0, 0), 2.0, 1.0 / 60.0)
            .unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn low_quality_reconstructs_between_columns_including_odd_widths() {
        let mut renderer = Renderer::new(13, 7);
        renderer.set_quality(RenderQuality::Low);
        let water = Water::new(17, 9);
        for frame in 0..3 {
            renderer
                .draw_at(
                    &mut Vec::new(),
                    &water,
                    4.0 + frame as f32 / 60.0,
                    1.0 / 60.0,
                )
                .unwrap();
            let varied_pairs = renderer
                .cells
                .chunks(13)
                .flat_map(|row| row.as_chunks::<2>().0)
                .filter(|pair| pair[0] != pair[1])
                .count();
            assert!(
                varied_pairs > 21,
                "bilinear reconstruction must not repeat whole column pairs"
            );
        }
    }

    #[test]
    fn downward_camera_receives_light_shafts_from_the_water_reflection() {
        let scene = SceneFrame::new(SceneKind::Acid, 12.5, 1.0);
        let camera = Camera::new(12.5, 200, 60, &scene);
        assert!(camera.project_dir(scene.light_dir).is_none());
        let projected = projected_light_source(&camera, scene.light_dir);
        assert!(
            projected.is_some(),
            "the visible reflected emitter must drive the radial pass"
        );
        let (u, v) = projected.unwrap();
        assert!(camera.hit_water(camera.ray(u, v)).is_some());
    }

    #[test]
    fn a_smooth_tilted_plane_has_no_geometric_roughness() {
        let levels = Array2::from_shape_fn((16, 16), |(x, y)| x as f32 * 0.002 + y as f32 * 0.003);
        let mut normals = Vec::new();
        precompute_normals(&levels, &mut normals);
        assert!(normals[8 * 16 + 8][3] < 1e-8);
    }

    #[test]
    fn color_hysteresis_is_bounded_and_accumulated_motion_still_updates() {
        let previous = (100, 120, 140);
        assert_eq!(stable_color((102, 118, 141), previous), previous);
        assert_eq!(stable_color((103, 120, 140), previous), (103, 120, 140));
        let mut shown = previous;
        for step in 1..=20 {
            let desired = (100 + step, 120 + step, 140 + step);
            shown = stable_color(desired, shown);
            assert!(shown.0.abs_diff(desired.0) <= 2);
            assert!(shown.1.abs_diff(desired.1) <= 2);
            assert!(shown.2.abs_diff(desired.2) <= 2);
        }
        assert!(shown.0 >= 118);
    }

    #[test]
    fn hdr_texel_centers_preserve_exact_colors() {
        let pixels = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [
            1.0, 1.0, 1.0,
        ]];
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(
                    sample_hdr(
                        &pixels,
                        2,
                        2,
                        (x as f32 + 0.5) / 2.0,
                        (y as f32 + 0.5) / 2.0
                    ),
                    pixels[y * 2 + x]
                );
            }
        }
    }

    #[test]
    fn singleton_buffers_are_safe_to_sample_at_edges() {
        let pixel = [0.2, 0.4, 0.8];
        for uv in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            assert_eq!(sample_hdr(&[pixel], 1, 1, uv, uv), pixel);
            assert_eq!(sample_grid1(&[0.7], 1, 1, uv, uv), 0.7);
            assert_eq!(sample_grid4(&[[0.1, 0.2, 0.3, 0.4]], 1, 1, uv, uv), [
                0.1, 0.2, 0.3, 0.4
            ]);
        }
    }

    #[test]
    fn all_quality_modes_render_tiny_odd_and_resized_screens() {
        for rapids in [false, true] {
            let mut water = Water::new(1, 1);
            water.set_rapids(rapids);
            water.update(std::time::Duration::from_millis(500));
            let mut renderer = Renderer::new(0, 0);
            let mut output = Vec::new();
            for quality in [
                RenderQuality::High,
                RenderQuality::Balanced,
                RenderQuality::Low,
            ] {
                renderer.set_quality(quality);
                for (w, h) in [(0, 0), (0, 1), (1, 0), (1, 1), (1, 7), (9, 1), (13, 7)] {
                    renderer.resize(w, h);
                    for frame in 0..2 {
                        renderer
                            .draw_at(&mut output, &water, 2.0 + frame as f32 / 60.0, 1.0 / 60.0)
                            .unwrap();
                        assert_eq!(renderer.snapshot_rgb().len(), w as usize * h as usize * 6);
                        assert!(
                            renderer
                                .hdr_buf
                                .iter()
                                .flatten()
                                .all(|value| value.is_finite())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn identical_inputs_produce_identical_rgb_and_ansi_frames() {
        let water = Water::new(17, 9);
        let mut a = Renderer::new(23, 11);
        let mut b = Renderer::new(23, 11);
        for frame in 0..3 {
            let time = 18.0 + frame as f32 / 60.0;
            let mut output_a = Vec::new();
            let mut output_b = Vec::new();
            a.draw_at(&mut output_a, &water, time, 1.0 / 60.0).unwrap();
            b.draw_at(&mut output_b, &water, time, 1.0 / 60.0).unwrap();
            assert_eq!(a.snapshot_rgb(), b.snapshot_rgb());
            assert_eq!(output_a, output_b);
        }
    }

    #[test]
    fn bloom_preserves_a_uniform_hdr_field_at_every_edge() {
        let scene = SceneFrame::new(SceneKind::Acid, 0.0, 0.0);
        for (width, height) in [(1, 1), (9, 7)] {
            let mut renderer = Renderer::new(width, height);
            renderer.prepare_buffers();
            let expected = [2.0, 0.7, 1.1];
            renderer.hdr_buf.fill(expected);
            let cam = Camera::new(0.0, width, height, &scene);
            renderer.postprocess(&cam, &scene, 0.0);
            for pixel in &renderer.hdr_buf {
                for channel in 0..3 {
                    assert!((pixel[channel] - expected[channel]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn film_reflection_remains_bounded_and_vanishes_at_zero_strength() {
        let base = [0.02, 0.03, 0.04];
        assert_eq!(thin_film_fresnel(base, 0.7, 0.1, 0.0, 3.0), base);
        let a = thin_film_fresnel(base, 0.7, 0.1, 1.0, 3.0);
        let b = thin_film_fresnel(base, 0.3, 0.1, 1.0, 3.0);
        assert_ne!(a, b);
        assert!(a.iter().chain(&b).all(|v| (0.0..=1.0).contains(v)));
        assert_eq!(thin_film_fresnel([1.0; 3], 0.0, 0.0, 1.0, 1.0), [1.0; 3]);
    }

    #[test]
    fn pupil_opening_cannot_brighten_an_extinguished_retinal_afterimage() {
        let mut renderer = Renderer::new(3, 2);
        renderer.prepare_buffers();
        renderer.adapted_exposure = AUTO_EXP_MIN;
        let scene = SceneFrame::new(SceneKind::Acid, 0.0, 2.0);
        renderer.hdr_buf.fill([1.0; 3]);
        renderer.adapt_perception(&scene, 0.0, 1.0 / 60.0, 1.0 / 60.0);
        let mut previous = f32::INFINITY;
        for frame in 1..=240 {
            renderer.hdr_buf.fill([0.0; 3]);
            let grade =
                renderer.adapt_perception(&scene, frame as f32 / 60.0, 1.0 / 60.0, 1.0 / 60.0);
            // A uniform field removes spatial warps from the test. No photons
            // arrive after extinguishing the source: a remembered signal may
            // fade slowly but a later pupil opening must not amplify it.
            let color = grade.apply(renderer.hdr_buf[0]);
            let y = dot3(color, [0.2126, 0.7152, 0.0722]);
            assert!(
                y <= previous + 1e-6,
                "frame {frame}: afterimage brightened {previous} -> {y}"
            );
            previous = y;
        }
        assert!(
            previous < 0.4,
            "the extinguished source must eventually fade"
        );
    }

    #[test]
    fn pupil_exposure_is_applied_once_with_and_without_retinal_persistence() {
        let scene = SceneFrame::new(SceneKind::Acid, 0.0, 2.0);
        let input = [0.32, 0.17, 0.24];
        for exposure in [AUTO_EXP_MIN, 1.0, AUTO_EXP_MAX] {
            for enabled in [false, true] {
                let mut renderer = Renderer::new(3, 2);
                renderer.prepare_buffers();
                renderer.retina.set_enabled(enabled);
                renderer.adapted_exposure = exposure;
                renderer.hdr_buf.fill(input);
                let grade = renderer.adapt_perception(&scene, 0.0, 0.0, 0.0);
                let actual = grade.apply(renderer.hdr_buf[0]);
                let expected = DisplayTransform::new(&scene, exposure).apply(input);
                for c in 0..3 {
                    assert!(
                        (actual[c] - expected[c]).abs() < 2e-6,
                        "{exposure}, {enabled}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn pupils_constrict_faster_than_they_adapt_to_darkness() {
        let toward_bright = 1.0 - adapt_exposure(1.0, 0.5, 0.1);
        let toward_dark = adapt_exposure(1.0, 1.5, 0.1) - 1.0;
        assert!(toward_bright > toward_dark * 2.0);
        let mut thirty_hz = 1.0;
        let mut sixty_hz = 1.0;
        for _ in 0..30 {
            thirty_hz = adapt_exposure(thirty_hz, 2.0, 1.0 / 30.0);
        }
        for _ in 0..60 {
            sixty_hz = adapt_exposure(sixty_hz, 2.0, 1.0 / 60.0);
        }
        assert!((thirty_hz - sixty_hz).abs() < 1e-5);
    }
}

#[inline(always)]
#[cfg(test)]
fn beer_lambert(path: f32) -> [f32; 3] {
    // The coefficients share a 0.4 base: powers 3/4/3 need only one exponential.
    let transmission = (-ABSORB[1] * 0.25 * path.max(0.0)).exp();
    let squared = transmission * transmission;
    let rose = squared * transmission;
    [rose, squared * squared, rose]
}

#[derive(Clone, Copy)]
struct WaterTransport {
    tau: [f32; 3],
    transmission: [f32; 3],
    scatter: [f32; 3],
}

impl WaterTransport {
    /// Homogeneous extinction and the single-scatter integral for a constant
    /// incident source: W = sigma_s/sigma_t * (1-exp(-sigma_t*length)).
    /// Absorbed photons never become a separate gray source.
    fn new(
        length: f32,
        absorption_scale: f32,
        scattering: f32,
    ) -> Self {
        let length = length.max(0.0);
        let extinction = ABSORB.map(|a| a * absorption_scale.max(0.0) + scattering.max(0.0));
        let tau = extinction.map(|sigma| if sigma == 0.0 { 0.0 } else { sigma * length });
        let red = (-tau[0]).exp();
        let green = (-tau[1]).exp();
        let transmission = [red, green, red]; // authored red/blue absorption are equal
        let scatter = std::array::from_fn(|c| {
            if scattering <= 0.0 || extinction[c] == 0.0 {
                return 0.0;
            }
            let removed = if tau[c] < 1e-4 {
                tau[c] * (1.0 + tau[c] * (-0.5 + tau[c] / 6.0))
            } else {
                1.0 - transmission[c]
            };
            scattering / extinction[c] * removed
        });
        Self {
            tau,
            transmission,
            scatter,
        }
    }
}

#[inline(always)]
fn filtered_roughness_squared(
    normal: [f32; 4],
    footprint: f32,
) -> f32 {
    // Bilinear averaging shortens disagreeing normals. Preserve that lost length
    // as unresolved variance rather than renormalizing it into a sparkling mirror.
    let length_squared = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    let unresolved = (1.0 - length_squared).max(0.0);
    let variance = normal[3] + unresolved;
    (GGX_ROUGHNESS * GGX_ROUGHNESS + variance * footprint.max(1.0).powi(2) * NDF_FILTER_K).min(1.0)
}

#[inline(always)]
fn projected_light_source(
    camera: &Camera,
    light: [f32; 3],
) -> Option<(f32, f32)> {
    camera.project_dir(light).or_else(|| {
        // Looking down at water, the sun is behind the camera but its planar
        // reflection is visible. Use that emitter for shafts through the bright mask.
        let reflected = [light[0], light[1], -light[2]];
        camera.hit_water(reflected)?;
        camera.project_dir(reflected)
    })
}

fn stable_color(
    current: (u8, u8, u8),
    previous: (u8, u8, u8),
) -> (u8, u8, u8) {
    let retain = |now: u8, before: u8| {
        if now.abs_diff(before) <= 2 {
            before
        } else {
            now
        }
    };
    (
        retain(current.0, previous.0),
        retain(current.1, previous.1),
        retain(current.2, previous.2),
    )
}
// Auto-exposure (eye adaptation)
const AUTO_EXP_TARGET: f32 = 0.25; // target geometric-mean luminance
const AUTO_EXP_MIN: f32 = 0.4; // min exposure multiplier
const AUTO_EXP_MAX: f32 = 6.4; // bounded dark-scene exposure headroom
const PUPIL_EXPOSURE_REFERENCE: f32 = 2.5; // preserve existing optical effect sizes

fn adapt_exposure(
    current: f32,
    target: f32,
    dt: f32,
) -> f32 {
    let rate = if target < current { 4.5 } else { 1.1 };
    current + (target - current) * (1.0 - (-rate * dt).exp())
}

// ── vector helpers ──────────────────────────────────────────────────

#[inline(always)]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2 < 1e-20 {
        return [0.0, 0.0, 1.0];
    }
    let il = 1.0 / l2.sqrt(); // rsqrt pattern: 1 div + 3 muls instead of 3 divs
    [v[0] * il, v[1] * il, v[2] * il]
}

#[inline(always)]
fn dot3(
    a: [f32; 3],
    b: [f32; 3],
) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Exact unpolarized dielectric Fresnel; eta is n_transmitted/n_incident.
#[inline(always)]
fn fresnel_scalar(
    cos_theta: f32,
    eta: f32,
) -> f32 {
    let cosine = cos_theta.abs().clamp(0.0, 1.0);
    let sin2 = (1.0 - cosine * cosine) / (eta * eta);
    if sin2 >= 1.0 {
        return 1.0;
    }
    let transmitted = (1.0 - sin2).sqrt();
    let parallel = (eta * cosine - transmitted) / (eta * cosine + transmitted);
    let perpendicular = (cosine - eta * transmitted) / (cosine + eta * transmitted);
    0.5 * (parallel * parallel + perpendicular * perpendicular)
}

#[inline(always)]
fn fresnel_rgb(cos_theta: f32) -> [f32; 3] { IOR_RGB.map(|eta| fresnel_scalar(cos_theta, eta)) }

#[inline(always)]
fn smith_ggx(
    n_dot_v: f32,
    n_dot_l: f32,
    alpha2: f32,
) -> f32 {
    let v = n_dot_v.clamp(0.0, 1.0);
    let l = n_dot_l.clamp(0.0, 1.0);
    let a = (alpha2 + (1.0 - alpha2) * v * v).sqrt();
    let b = (alpha2 + (1.0 - alpha2) * l * l).sqrt();
    2.0 * v * l / (l * a + v * b).max(1e-8)
}

#[inline(always)]
fn thin_film_fresnel(
    base: [f32; 3],
    cos_view: f32,
    height: f32,
    strength: f32,
    phase: f32,
) -> [f32; 3] {
    crate::optics::film_reflection(base, cos_view, height, strength, phase)
}

// ── interleaved gradient noise (Jimenez 2014) ──────────────────────

#[inline(always)]
fn ign(
    x: f32,
    y: f32,
    frame: f32,
) -> f32 {
    let t = 0.06711056 * x + 0.00583715 * y + 0.00519673 * frame;
    ((52.982_918 * t.fract()).fract()) - 0.5
}

// ── sky envmap (octahedral, precomputed per frame) ──────────────────

struct SkyEnvironment {
    bed: crate::bed::BedWorld,
    size: usize,
    pixels: Vec<(f32, f32, f32)>,
    directions: Vec<[f32; 3]>,
    // Horizontal diffuse illumination after entry Fresnel (irradiance / pi).
    bed_irradiance: [f32; 3],
    // Isotropic phase-source quadrature of transmitted incident light (1/4pi).
    volume_incident: [f32; 3],
    bed_illumination_path: f32,
    // Bounded spectral transmission of an artistic projected-light filter.
    // It colors incident illumination; bed albedo and pigment mass stay physical.
    bed_light_filter: Vec<[f32; 3]>,
    bed_light_offset: [f32; 2],
    bed_emission_strength: f32,
    // Isotropic single-scattering airlight closure: a smooth hemisphere mean.
    airlight: [f32; 3],
}

/// Broad diffuse illumination for aerated water. Each lobe is a normalized
/// cosine-weighted hemisphere integral of the cached sky, i.e. irradiance / pi.
/// Six lobes keep the per-hit evaluation smooth and inexpensive.
struct FoamLighting {
    lobes: [[f32; 3]; 6],
}

impl FoamLighting {
    fn from_sky(sky: &SkyEnvironment) -> Self {
        let mut lobes = [[0.0; 3]; 6];
        for (index, lobe) in lobes.iter_mut().enumerate() {
            let axis = index / 2;
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            for sample in 0..16 {
                let radius2 = (sample as f32 + 0.5) / 16.0;
                let radius = radius2.sqrt();
                let (sine, cosine) = (sample as f32 * 2.399_963_1).sin_cos();
                let mut direction = [0.0; 3];
                direction[axis] = sign * (1.0 - radius2).sqrt();
                direction[(axis + 1) % 3] = radius * cosine;
                direction[(axis + 2) % 3] = radius * sine;
                let incoming = sample_sky(sky, direction);
                for (channel, value) in [incoming.0, incoming.1, incoming.2].iter().enumerate() {
                    lobe[channel] += value * (1.0 / 16.0);
                }
            }
        }
        Self { lobes }
    }

    #[inline]
    fn radiance(
        &self,
        normal: [f32; 3],
        n_dot_l: f32,
    ) -> [f32; 3] {
        let weight_sum = normal
            .iter()
            .map(|value| value.abs())
            .sum::<f32>()
            .max(1e-6);
        // Same neutral daylight emitter as the water's direct specular lobe.
        // Lambert's 1/pi and an albedo below one keep this non-emissive.
        let mut result = [0.18 / PI * n_dot_l.max(0.0); 3];
        for (axis, component) in normal.iter().enumerate() {
            let lobe = self.lobes[axis * 2 + usize::from(*component < 0.0)];
            let weight = component.abs() / weight_sum;
            for channel in 0..3 {
                result[channel] += lobe[channel] * weight;
            }
        }
        result.map(|value| value * FOAM_ALBEDO)
    }
}

#[inline]
fn mix_foam(
    water: [f32; 3],
    foam: [f32; 3],
    coverage: f32,
) -> [f32; 3] {
    if coverage <= 0.0 {
        return water;
    }
    if coverage >= 1.0 {
        return foam;
    }
    std::array::from_fn(|channel| water[channel] * (1.0 - coverage) + foam[channel] * coverage)
}

/// Three passive cyan/magenta/amber pigments. The field stores column-integrated
/// mass, so depth cancels between volumetric concentration and optical path.
/// Positive, broad extinction bands retain the soft palette without emission.
#[inline]
fn dye_transmittance(
    areal_mass: [f32; 3],
    path_over_depth: f32,
) -> [f32; 3] {
    crate::pigment_optics::transmittance(areal_mass, path_over_depth)
}

/// Optional material fields share the contiguous x-major water grid. The clear
/// branch retains the original constant-bottom arithmetic exactly.
#[derive(Clone, Copy)]
struct MaterialGrid<'a> {
    depths: Option<&'a [f32]>,
    foam: Option<&'a [f32]>,
    dye: Option<&'a [[f32; 3]]>,
    foam_light: Option<&'a FoamLighting>,
    bed_caustics: Option<&'a [f32]>,
}

impl MaterialGrid<'_> {
    fn clear() -> Self {
        Self {
            depths: None,
            foam: None,
            dye: None,
            foam_light: None,
            bed_caustics: None,
        }
    }

    #[inline(always)]
    fn depth(
        self,
        footprint: SampleFootprint,
        height: f32,
    ) -> f32 {
        self.depths.map_or_else(
            || (WATER_DEPTH + height).max(0.001),
            |depths| footprint.scalar(depths).max(0.001),
        )
    }

    #[inline]
    fn dye_mass(
        self,
        footprint: SampleFootprint,
    ) -> Option<[f32; 3]> {
        let dye = self.dye?;
        let [a, b, c, d] = footprint.indices.map(|index| dye[index]);
        Some(std::array::from_fn(|channel| {
            let upper = a[channel] + (b[channel] - a[channel]) * footprint.tx;
            let lower = c[channel] + (d[channel] - c[channel]) * footprint.tx;
            upper + (lower - upper) * footprint.ty
        }))
    }

    #[inline]
    fn dye_transmittance(
        self,
        footprint: SampleFootprint,
        path_over_depth: f32,
    ) -> [f32; 3] {
        self.dye_mass(footprint)
            .map_or([1.0; 3], |mass| dye_transmittance(mass, path_over_depth))
    }

    #[inline]
    fn dye_transport(
        self,
        footprint: SampleFootprint,
        path_over_depth: f32,
        water_tau: [f32; 3],
        beer: [f32; 3],
    ) -> crate::pigment_optics::PigmentTransport {
        let Some(mass) = self.dye_mass(footprint) else {
            return crate::pigment_optics::PigmentTransport {
                transmission: [1.0; 3],
                volume: [1.0; 3],
            };
        };
        crate::pigment_optics::transport(mass, path_over_depth, water_tau, beer)
    }
}

impl SkyEnvironment {
    fn new() -> Self {
        let mut bed = crate::bed::BedWorld::new(7);
        bed.prepare([-1.0; 2], 7);
        Self {
            bed,
            size: 0,
            pixels: Vec::new(),
            directions: Vec::new(),
            bed_irradiance: [0.0; 3],
            volume_incident: [0.0; 3],
            bed_illumination_path: 1.0,
            bed_light_filter: Vec::new(),
            bed_light_offset: [0.0; 2],
            bed_emission_strength: 0.0,
            airlight: [0.0; 3],
        }
    }

    fn prepare_world_bed(
        &mut self,
        origin: [f64; 2],
        seed: u64,
    ) {
        self.bed.prepare(origin, seed);
    }

    #[cfg(test)]
    fn constant(rgb: (f32, f32, f32)) -> Self {
        let mut sky = Self::new();
        sky.size = 4;
        sky.pixels = vec![rgb; 16];
        sky.update_diffuse_lighting();
        sky
    }

    fn update_diffuse_lighting(&mut self) {
        self.bed_irradiance = [0.0; 3];
        self.volume_incident = [0.0; 3];
        self.airlight = [0.0; 3];
        let (mut path_sum, mut path_weight) = (0.0, 0.0);
        // Equal solid-angle upper-hemisphere quadrature. Bed irradiance uses
        // cos(theta)/pi; a uniform source stays bounded by its input radiance.
        // The cached path is a radiance-weighted mean secant after refraction,
        // an inexpensive diffuse-illumination approximation, not a spectral RT solve.
        const SAMPLES: usize = 32;
        for i in 0..SAMPLES {
            let z = (i as f32 + 0.5) / SAMPLES as f32;
            let radius = (1.0 - z * z).sqrt();
            let (sin, cos) = (i as f32 * 2.399_963_1).sin_cos();
            let incoming = sample_sky(self, [radius * cos, radius * sin, z]);
            let rgb = [incoming.0, incoming.1, incoming.2];
            let entry = fresnel_rgb(z);
            let cosine_weight = 2.0 * z / SAMPLES as f32;
            let transmitted =
                std::array::from_fn::<_, 3, _>(|c| rgb[c] * (1.0 - entry[c]) * cosine_weight);
            let weight = dot3(transmitted, [0.2126, 0.7152, 0.0722]);
            let cos_t = (1.0 - (1.0 - z * z) / (IOR_WATER * IOR_WATER)).sqrt();
            path_sum += weight / cos_t;
            path_weight += weight;
            for c in 0..3 {
                self.bed_irradiance[c] += transmitted[c];
                // Snell angular Jacobian combined with radiance transport:
                // L_t dOmega_t = (1-F) L_i cos_i/cos_t dOmega_i.
                self.volume_incident[c] +=
                    rgb[c] * (1.0 - entry[c]) * z / (cos_t * 2.0 * SAMPLES as f32);
                self.airlight[c] += rgb[c] / SAMPLES as f32;
            }
        }
        self.bed_illumination_path = if path_weight > 1e-12 {
            path_sum / path_weight
        } else {
            1.0
        };
    }

    fn update_bed_light_filter(
        &mut self,
        scene: &SceneFrame,
    ) {
        let strength = (scene.light_complexity * 2.0).clamp(0.0, 1.0);
        self.bed_emission_strength = strength;
        if strength == 0.0 {
            self.bed_light_filter.clear();
            return;
        }
        // A world lattice keeps the sampling grid stable across patch shifts,
        // including the 1.5 m refracted halo. Only the source phase moves.
        const DENSITY: f64 = 12.0;
        let origin = self.bed.origin();
        let minimum = origin
            .map(|v| ((v - f64::from(crate::bed::REFRACTION_HALO)) * DENSITY).floor() as i64 - 1);
        self.bed_light_offset =
            std::array::from_fn(|c| (origin[c] * DENSITY - minimum[c] as f64) as f32);
        self.bed_light_filter
            .resize(BED_LIGHT_SIZE * BED_LIGHT_SIZE, [1.0; 3]);
        for y in 0..BED_LIGHT_SIZE {
            for x in 0..BED_LIGHT_SIZE {
                let point = [
                    (minimum[0] + x as i64) as f64 / DENSITY,
                    (minimum[1] + y as i64) as f64 / DENSITY,
                ];
                let color = crate::bed::projected_light(point, scene.sky_phase[2], self.bed.seed());
                self.bed_light_filter[y * BED_LIGHT_SIZE + x] = color.map(|c| {
                    // Convex passive spectral filter; no unbounded white fill.
                    1.0 + strength * (0.12 + 0.88 * c - 1.0)
                });
            }
        }
    }

    #[inline]
    fn bed_light_transmission(
        &self,
        point: [f32; 2],
        channel: usize,
    ) -> f32 {
        if self.bed_light_filter.is_empty() {
            return 1.0;
        }
        let x = (point[0] * PATCH_LENGTH * 12.0 + self.bed_light_offset[0])
            .clamp(0.0, (BED_LIGHT_SIZE - 1) as f32);
        let y = (point[1] * PATCH_LENGTH * 12.0 + self.bed_light_offset[1])
            .clamp(0.0, (BED_LIGHT_SIZE - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = (
            (x0 + 1).min(BED_LIGHT_SIZE - 1),
            (y0 + 1).min(BED_LIGHT_SIZE - 1),
        );
        let (u, v) = (x - x0 as f32, y - y0 as f32);
        let a = self.bed_light_filter[y0 * BED_LIGHT_SIZE + x0][channel];
        let b = self.bed_light_filter[y0 * BED_LIGHT_SIZE + x1][channel];
        let c = self.bed_light_filter[y1 * BED_LIGHT_SIZE + x0][channel];
        let d = self.bed_light_filter[y1 * BED_LIGHT_SIZE + x1][channel];
        let top = a + (b - a) * u;
        top + ((c + (d - c) * u) - top) * v
    }

    fn update(
        &mut self,
        size: usize,
        elapsed: f32,
        scene: &SceneFrame,
    ) {
        if self.size != size {
            self.size = size;
            self.pixels.resize(size * size, (0.0, 0.0, 0.0));
            self.directions.clear();
            for y in 0..size {
                for x in 0..size {
                    self.directions.push(oct_to_dir(
                        (x as f32 + 0.5) / size as f32,
                        (y as f32 + 0.5) / size as f32,
                    ));
                }
            }
        }
        for (pixel, &dir) in self.pixels.iter_mut().zip(&self.directions) {
            *pixel = sky_color(elapsed, dir, scene);
        }
        self.update_diffuse_lighting();
        self.update_bed_light_filter(scene);
    }
}

/// Octahedral mapping: direction → UV [0,1]² (no transcendentals).
#[inline(always)]
fn dir_to_oct(d: [f32; 3]) -> (f32, f32) {
    let inv = 1.0 / (d[0].abs() + d[1].abs() + d[2].abs());
    let mut u = d[0] * inv;
    let mut v = d[1] * inv;
    if d[2] < 0.0 {
        let su = u;
        let sv = v;
        u = (1.0 - sv.abs()) * if su >= 0.0 { 1.0 } else { -1.0 };
        v = (1.0 - su.abs()) * if sv >= 0.0 { 1.0 } else { -1.0 };
    }
    (u * 0.5 + 0.5, v * 0.5 + 0.5)
}

/// Inverse: UV [0,1]² → direction.
#[inline(always)]
fn oct_to_dir(
    uu: f32,
    vv: f32,
) -> [f32; 3] {
    let u = uu * 2.0 - 1.0;
    let v = vv * 2.0 - 1.0;
    let z = 1.0 - u.abs() - v.abs();
    let (u, v) = if z < 0.0 {
        (
            (1.0 - v.abs()) * if u >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - u.abs()) * if v >= 0.0 { 1.0 } else { -1.0 },
        )
    } else {
        (u, v)
    };
    norm3([u, v, z])
}

#[inline(always)]
fn sample_sky(
    environment: &SkyEnvironment,
    dir: [f32; 3],
) -> (f32, f32, f32) {
    let env = &environment.pixels;
    let n = environment.size as f32;
    let (u, v) = dir_to_oct(dir);
    let fx = (u * n - 0.5).clamp(0.0, n - 1.0);
    let fy = (v * n - 0.5).clamp(0.0, n - 1.0);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let ns = environment.size;
    let p00 = env[y0 * ns + x0];
    let x1 = (x0 + 1).min(ns - 1);
    let y1 = (y0 + 1).min(ns - 1);
    let p10 = env[y0 * ns + x1];
    let p01 = env[y1 * ns + x0];
    let p11 = env[y1 * ns + x1];
    (
        (p00.0 + (p10.0 - p00.0) * tx) * (1.0 - ty) + (p01.0 + (p11.0 - p01.0) * tx) * ty,
        (p00.1 + (p10.1 - p00.1) * tx) * (1.0 - ty) + (p01.1 + (p11.1 - p01.1) * tx) * ty,
        (p00.2 + (p10.2 - p00.2) * tx) * (1.0 - ty) + (p01.2 + (p11.2 - p01.2) * tx) * ty,
    )
}

// ── camera ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Camera {
    pos: [f32; 3],
    ground: [f32; 2], // look-at point, independent of the sampled patch
    patch_center: [f32; 2],
    world_origin: Option<[f64; 2]>, // lower-left world metres; geometry stays local
    fwd: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    half_w: f32,
    half_h: f32,
}

impl Camera {
    const NOMINAL_HEIGHTS: [f32; 2] = [-0.16, 0.12];
    const TARGET_RESERVE: f32 = 0.04;

    fn place_in_world(
        &mut self,
        center: [f64; 2],
        origin: [f64; 2],
    ) {
        for axis in 0..2 {
            let target = (center[axis] - origin[axis] - f64::from(PATCH_LENGTH) * 0.5) as f32;
            self.pos[axis] += target - self.ground[axis];
            self.ground[axis] = target;
        }
        self.patch_center = [0.0; 2];
        self.world_origin = Some(origin);
    }

    /// Reconstruct a current sample in the previous local coordinate frame.
    /// Large absolute positions never enter per-pixel f32 geometry.
    fn history_direction(
        &self,
        previous: &Self,
        point: [f32; 2],
        height: f32,
    ) -> Option<[f32; 3]> {
        let delta = match (self.world_origin, previous.world_origin) {
            (Some(current), Some(old)) => {
                std::array::from_fn(|axis| (current[axis] - old[axis]) as f32)
            },
            (None, None) => [0.0; 2],
            _ => return None,
        };
        let local = std::array::from_fn::<_, 2, _>(|axis| {
            (point[axis] - 0.5) * PATCH_LENGTH + delta[axis] + previous.patch_center[axis]
        });
        if (0..2)
            .any(|axis| (local[axis] - previous.patch_center[axis]).abs() >= PATCH_LENGTH * 0.5)
        {
            return None;
        }
        Some([
            local[0] - previous.pos[0],
            local[1] - previous.pos[1],
            height - previous.pos[2],
        ])
    }

    /// Fit a complete ray cone, independent of roll, orbit heading and normal
    /// rain extrema. The fixed height envelope and 4 cm target allowance absorb
    /// normal patch rebases and presentation lag without pumping the lens.
    fn fit_surface(
        &mut self,
        bounds: [f32; 2],
    ) {
        let heights = [
            Self::NOMINAL_HEIGHTS[0].min(bounds[0]),
            Self::NOMINAL_HEIGHTS[1].max(bounds[1]),
        ];
        let offset =
            (self.ground[0] - self.patch_center[0]).hypot(self.ground[1] - self.patch_center[1]);
        let radius = (PATCH_LENGTH * 0.47 - Self::TARGET_RESERVE.max(offset)).max(0.005);
        let mut s = -self.fwd[2];
        let mut c = self.fwd[0].hypot(self.fwd[1]);
        let mut distance = self.pos[2] / s;
        // Extraordinary heights can put even the central ray outside the patch.
        // Lift the orbit only then; ordinary waves never alter its pose.
        let required_elevation = (heights[0].abs().max(heights[1].abs()) / radius).atan() + 0.01;
        if s.atan2(c) < required_elevation || self.pos[2] < heights[1] + 0.08 {
            let elevation = s.atan2(c).max(required_elevation);
            let heading = self.fwd[1].atan2(self.fwd[0]);
            let unrolled_right = norm3([self.fwd[1], -self.fwd[0], 0.0]);
            let roll = (-dot3(self.up, unrolled_right)).atan2(dot3(self.right, unrolled_right));
            s = elevation.sin();
            c = elevation.cos();
            distance = distance.max((heights[1] + 0.08) / s);
            self.orient(heading, elevation, roll, distance);
        }
        // A cone cuts each horizontal height plane in an ellipse. Its two
        // major-axis endpoints bound every ray for every rotation; solving them
        // against the reserved circle gives these limits on tan(half angle).
        let mut limit = s / c.max(0.00001) * 0.99;
        for height in heights {
            for sign in [-1.0, 1.0] {
                let numerator = radius * s + sign * height * c;
                let denominator = distance - height * s + sign * radius * c;
                if denominator > 0.0 {
                    limit = limit.min((numerator / denominator).max(0.00001));
                }
            }
        }
        let requested = self.half_w.hypot(self.half_h);
        // C1 smooth minimum avoids a velocity kink when a slowly changing scene
        // FOV crosses the safety cone. It never exceeds either aperture bound.
        let rounding = 0.02;
        let blend = (1.0 - (requested - limit).abs() / rounding).max(0.0);
        let cone = requested.min(limit) - 0.25 * rounding * blend * blend;
        let scale = cone.max(0.00001) / requested.max(0.00001);
        self.half_w *= scale;
        self.half_h *= scale;
    }

    fn orient(
        &mut self,
        heading: f32,
        elevation: f32,
        roll: f32,
        distance: f32,
    ) {
        let (se, ce) = elevation.sin_cos();
        let (sh, ch) = heading.sin_cos();
        self.pos = [
            self.ground[0] - distance * ce * ch,
            self.ground[1] - distance * ce * sh,
            distance * se,
        ];
        self.fwd = norm3([ce * ch, ce * sh, -se]);
        let right = norm3([self.fwd[1], -self.fwd[0], 0.0]);
        let up = [
            right[1] * self.fwd[2] - right[2] * self.fwd[1],
            right[2] * self.fwd[0] - right[0] * self.fwd[2],
            right[0] * self.fwd[1] - right[1] * self.fwd[0],
        ];
        let (sr, cr) = roll.sin_cos();
        self.right = std::array::from_fn(|i| right[i] * cr + up[i] * sr);
        self.up = std::array::from_fn(|i| up[i] * cr - right[i] * sr);
    }

    /// Bracket against the height slab and the finite patch, then solve z=eta(x,y).
    /// Bounded work, including a grazing-ray scan when the patch clips the slab.
    fn hit_surface(
        &self,
        dir: [f32; 3],
        levels: &Array2<f32>,
        bounds: [f32; 2],
    ) -> Option<([f32; 2], f32)> {
        let (mut near, mut far) = if dir[2].abs() > 1e-7 {
            let a = (bounds[0] - self.pos[2]) / dir[2];
            let b = (bounds[1] - self.pos[2]) / dir[2];
            (a.min(b).max(0.0), a.max(b))
        } else {
            if self.pos[2] < bounds[0] || self.pos[2] > bounds[1] {
                return None;
            }
            (0.0, PATCH_LENGTH * 4.0)
        };
        for (axis, ray_component) in dir.iter().enumerate().take(2) {
            let origin = (self.pos[axis] - self.patch_center[axis]) / PATCH_LENGTH + 0.5;
            let direction = ray_component / PATCH_LENGTH;
            if direction.abs() < 1e-8 {
                if !(0.0..=1.0).contains(&origin) {
                    return None;
                }
            } else {
                let a = -origin / direction;
                let b = (1.0 - origin) / direction;
                near = near.max(a.min(b));
                far = far.min(a.max(b));
            }
        }
        if far < near {
            return None;
        }
        let point = |t: f32| {
            [
                (self.pos[0] + t * dir[0] - self.patch_center[0]) / PATCH_LENGTH + 0.5,
                (self.pos[1] + t * dir[1] - self.patch_center[1]) / PATCH_LENGTH + 0.5,
            ]
        };
        let residual = |t: f32| {
            let wp = point(t);
            self.pos[2] + t * dir[2] - sample(levels, wp[0], wp[1])
        };
        let mut a = residual(near);
        let mut b = residual(far);
        if a.abs() < 1e-6 {
            return Some((point(near), near));
        }
        if b.abs() < 1e-6 {
            return Some((point(far), far));
        }
        if a < 0.0 || b > 0.0 || dir[2] >= -0.30 {
            let start = near;
            let span = far - near;
            let mut found = false;
            for step in 1..=16 {
                let t = start + span * step as f32 / 16.0;
                let value = residual(t);
                if a >= 0.0 && value <= 0.0 {
                    far = t;
                    b = value;
                    found = true;
                    break;
                }
                near = t;
                a = value;
            }
            if !found {
                return None;
            }
        }
        let mut t = near;
        for _ in 0..12 {
            let fraction = (a / (a - b)).clamp(0.02, 0.98);
            t = near + (far - near) * fraction;
            let value = residual(t);
            if value.abs() < 1e-5 {
                return Some((point(t), t));
            }
            if value > 0.0 {
                near = t;
                a = value;
            } else {
                far = t;
                b = value;
            }
        }
        (residual(t).abs() < 1e-4).then(|| (point(t), t))
    }

    fn new(
        elapsed: f32,
        sw: u16,
        sh: u16,
        scene: &SceneFrame,
    ) -> Self {
        // Heading/roll use the synchronous scene clock. World translation is
        // installed separately, before optional presentation smoothing.
        let heading = elapsed / DRIFT_TURN_PERIOD * 2.0 * PI;
        let drift_r = DRIFT_SPEED * DRIFT_TURN_PERIOD / (2.0 * PI);
        let ground = [drift_r * heading.sin(), -drift_r * heading.cos()];
        let aspect = sw as f32 / (sh as f32 * CHAR_ASPECT);
        let half_w = (scene.fov * 0.5).tan();
        let mut camera = Self {
            pos: [0.0; 3],
            ground,
            patch_center: ground,
            world_origin: None,
            fwd: [0.0; 3],
            right: [0.0; 3],
            up: [0.0; 3],
            half_w,
            half_h: half_w / aspect,
        };
        camera.orient(
            heading,
            scene.camera_elevation,
            scene.camera_roll,
            CAMERA_DIST,
        );
        camera
    }

    /// Perspective ray direction for screen-space (u,v) in [0,1]².
    #[inline(always)]
    fn ray(
        &self,
        u: f32,
        v: f32,
    ) -> [f32; 3] {
        let su = (u - 0.5) * 2.0 * self.half_w;
        let sv = (0.5 - v) * 2.0 * self.half_h; // screen top = up
        norm3([
            self.fwd[0] + su * self.right[0] + sv * self.up[0],
            self.fwd[1] + su * self.right[1] + sv * self.up[1],
            self.fwd[2] + su * self.right[2] + sv * self.up[2],
        ])
    }

    /// Inverse of `ray`: project a direction back to screen-space (u,v).
    #[inline(always)]
    fn project_dir(
        &self,
        d: [f32; 3],
    ) -> Option<(f32, f32)> {
        let sf = dot3(d, self.fwd);
        if sf <= 0.0 {
            return None;
        }
        let u = 0.5 + dot3(d, self.right) / (sf * 2.0 * self.half_w);
        let v = 0.5 - dot3(d, self.up) / (sf * 2.0 * self.half_h);
        if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 {
            return None;
        }
        Some((u, v))
    }

    /// Intersect ray with the z=0 water plane; returns normalised (wx,wy) or None.
    #[inline(always)]
    fn hit_water(
        &self,
        dir: [f32; 3],
    ) -> Option<[f32; 2]> {
        if dir[2] >= -1e-6 {
            return None;
        }
        let t = -self.pos[2] / dir[2];
        let wx = (self.pos[0] + t * dir[0] - self.patch_center[0]) / WATER_SCALE + 0.5;
        let wy = (self.pos[1] + t * dir[1] - self.patch_center[1]) / WATER_SCALE + 0.5;
        if wx < 0.0 || wx >= 1.0 || wy < 0.0 || wy >= 1.0 {
            return None;
        }
        Some([wx, wy])
    }
}

/// Critically damp the view in absolute world coordinates, so a quantized
/// simulation-window rebase cannot become a camera movement. Terminal rendering
/// and simulation share one scene clock; no snapshot prediction is necessary.
struct CameraPresentation {
    pose: Option<[f64; 5]>, // world x/y, elevation, FOV, roll minus its steady spin
    velocity: [f64; 5],
    last_elapsed: f32,
    world_coordinates: bool,
    envelope: [f32; 2],
}

impl Default for CameraPresentation {
    fn default() -> Self {
        Self {
            pose: None,
            velocity: [0.0; 5],
            last_elapsed: 0.0,
            world_coordinates: false,
            envelope: Camera::NOMINAL_HEIGHTS,
        }
    }
}

impl CameraPresentation {
    fn reset(&mut self) { *self = Self::default(); }

    fn prepare(
        &mut self,
        camera: &mut Camera,
        elapsed: f32,
        delta: f32,
        scene: &SceneFrame,
        bounds: [f32; 2],
    ) {
        let origin = camera.world_origin.map_or([0.0; 2], |origin| {
            origin.map(|x| x + f64::from(PATCH_LENGTH) * 0.5)
        });
        let roll_rate = 4.0 * std::f64::consts::PI / 180.0;
        let desired = [
            origin[0] + f64::from(camera.ground[0]),
            origin[1] + f64::from(camera.ground[1]),
            f64::from(scene.camera_elevation),
            f64::from(scene.fov),
            f64::from(scene.camera_roll) - f64::from(elapsed) * roll_rate,
        ];
        let dt = if delta.is_finite() {
            f64::from(delta.clamp(0.0, 0.1))
        } else {
            0.0
        };
        let world_coordinates = camera.world_origin.is_some();
        if self.pose.is_none()
            || world_coordinates != self.world_coordinates
            || elapsed < self.last_elapsed
        {
            self.pose = Some(desired);
            self.velocity = [0.0; 5];
            self.last_elapsed = elapsed;
            self.world_coordinates = world_coordinates;
            self.envelope = Camera::NOMINAL_HEIGHTS;
        } else if dt > 0.0
            && let Some(pose) = &mut self.pose
        {
            let omega = 1.0 / 0.06;
            let decay = (-omega * dt).exp();
            for axis in 0..5 {
                let displacement = pose[axis] - desired[axis];
                let transient = self.velocity[axis] + omega * displacement;
                pose[axis] = desired[axis] + (displacement + transient * dt) * decay;
                self.velocity[axis] = (self.velocity[axis] - omega * transient * dt) * decay;
            }
            self.last_elapsed = elapsed;
        }
        let pose = self.pose.unwrap();
        camera.ground = std::array::from_fn(|axis| (pose[axis] - origin[axis]) as f32);
        let aspect = camera.half_w / camera.half_h;
        camera.half_w = (pose[3] as f32 * 0.5).tan();
        camera.half_h = camera.half_w / aspect;
        camera.orient(
            self.last_elapsed / DRIFT_TURN_PERIOD * 2.0 * PI,
            pose[2] as f32,
            (pose[4] + f64::from(self.last_elapsed) * roll_rate) as f32,
            CAMERA_DIST,
        );
        // Expand immediately if an exceptional wave requires it; return to the
        // usual envelope over two seconds. A held frame cannot release it.
        let desired_envelope = [
            Camera::NOMINAL_HEIGHTS[0].min(bounds[0] - 0.002),
            Camera::NOMINAL_HEIGHTS[1].max(bounds[1] + 0.002),
        ];
        let release = (-dt / 2.0).exp() as f32;
        for (axis, desired) in desired_envelope.into_iter().enumerate() {
            let released = desired + (self.envelope[axis] - desired) * release;
            self.envelope[axis] = if axis == 0 {
                desired.min(released)
            } else {
                desired.max(released)
            };
        }
        camera.fit_surface(self.envelope);
    }
}

#[cfg(test)]
mod terminal_camera_tests {
    use super::*;

    fn scene() -> SceneFrame {
        let mut scene = SceneFrame::new(SceneKind::Prism, 0.0, 4.0);
        scene.camera_elevation = 70.0_f32.to_radians();
        scene.fov = 61.0_f32.to_radians();
        scene.camera_roll = 0.0;
        scene
    }

    fn world_camera(
        scene: &SceneFrame,
        time: f32,
        target: [f64; 2],
        origin: [f64; 2],
    ) -> Camera {
        let mut camera = Camera::new(time, 200, 60, scene);
        camera.place_in_world(target, origin);
        camera
    }

    fn world_eye(camera: &Camera) -> [f64; 3] {
        let origin = camera.world_origin.unwrap();
        [
            origin[0] + f64::from(PATCH_LENGTH) * 0.5 + f64::from(camera.pos[0]),
            origin[1] + f64::from(PATCH_LENGTH) * 0.5 + f64::from(camera.pos[1]),
            f64::from(camera.pos[2]),
        ]
    }

    #[test]
    fn normal_rain_rotation_and_rebase_do_not_zoom_the_lens() {
        let mut maximum_wave_jump = 0.0_f32;
        let mut maximum_rebase_jump = 0.0_f32;
        for seconds in 0..=180 {
            let mut scene = SceneFrame::new(SceneKind::Prism, 0.0, 4.0);
            scene.camera_roll = seconds as f32 * 4.0_f32.to_radians();
            let mut calm = Camera::new(seconds as f32, 200, 60, &scene);
            calm.place_in_world([0.0; 2], [-1.0; 2]);
            let mut splash = calm;
            calm.fit_surface([-0.03, 0.03]);
            splash.fit_surface([-0.08, 0.08]);
            maximum_wave_jump = maximum_wave_jump.max((splash.half_w / calm.half_w - 1.0).abs());
            let mut before = Camera::new(seconds as f32, 200, 60, &scene);
            let mut after = before;
            let spacing = 2.0 / 96.0;
            before.place_in_world([spacing * 0.5 - 1e-6, 0.0], [-1.0; 2]);
            after.place_in_world([spacing * 0.5 + 1e-6, 0.0], [-1.0 + spacing, -1.0]);
            before.fit_surface([-0.03, 0.03]);
            after.fit_surface([-0.03, 0.03]);
            maximum_rebase_jump =
                maximum_rebase_jump.max((after.half_w / before.half_w - 1.0).abs());
        }
        println!(
            "maximum_wave_aperture_jump_percent={}; maximum_rebase_aperture_jump_percent={}",
            maximum_wave_jump * 100.0,
            maximum_rebase_jump * 100.0
        );
        assert!(
            maximum_wave_jump < 0.000001,
            "normal rain changed the camera aperture by {}%",
            maximum_wave_jump * 100.0
        );
        assert!(
            maximum_rebase_jump < 0.000001,
            "water-window rebase changed the camera aperture by {}%",
            maximum_rebase_jump * 100.0
        );
    }

    #[test]
    fn full_orbit_and_roll_have_a_constant_aperture() {
        let mut scene = scene();
        let mut minimum = f32::INFINITY;
        let mut maximum = 0.0_f32;
        for step in 0..=720 {
            let time = step as f32 * 0.25;
            scene.camera_roll = time * 0.13;
            let mut camera = world_camera(&scene, time, [0.0; 2], [-1.0; 2]);
            camera.fit_surface([-0.03, 0.03]);
            minimum = minimum.min(camera.half_w);
            maximum = maximum.max(camera.half_w);
        }
        println!("full_orbit_aperture_range={}", maximum - minimum);
        assert!(maximum - minimum < 0.000001);
    }

    #[test]
    fn camera_damping_is_cadence_invariant_and_holds_across_a_rebase() {
        let mut scene = scene();
        let origin = [-1.0; 2];
        let mut whole = CameraPresentation::default();
        let mut split = CameraPresentation::default();
        let mut a = world_camera(&scene, 0.0, [0.0; 2], origin);
        let mut b = a;
        whole.prepare(&mut a, 0.0, 0.0, &scene, [-0.03, 0.03]);
        split.prepare(&mut b, 0.0, 0.0, &scene, [-0.03, 0.03]);
        scene.camera_elevation += 0.05;
        scene.fov -= 0.08;
        scene.camera_roll += 0.03;
        a = world_camera(&scene, 0.0, [0.01, 0.005], origin);
        whole.prepare(&mut a, 0.0, 0.1, &scene, [-0.03, 0.03]);
        for _ in 0..6 {
            b = world_camera(&scene, 0.0, [0.01, 0.005], origin);
            split.prepare(&mut b, 0.0, 1.0 / 60.0, &scene, [-0.03, 0.03]);
        }
        for (x, y) in whole.pose.unwrap().into_iter().zip(split.pose.unwrap()) {
            assert!((x - y).abs() < 0.00000001);
        }
        assert!((a.half_w - b.half_w).abs() < 0.000001);
        let held_eye = world_eye(&a);
        let rebased_origin = [origin[0] + 2.0 / 96.0, origin[1] - 2.0 / 64.0];
        let mut held = world_camera(&scene, 100.0, [0.5, -0.5], rebased_origin);
        whole.prepare(&mut held, 100.0, 0.0, &scene, [-0.03, 0.03]);
        for (before, after) in held_eye.into_iter().zip(world_eye(&held)) {
            assert!((before - after).abs() < 0.000001, "held world eye moved");
        }
        assert!((a.half_w - held.half_w).abs() < 0.000001);
        assert_eq!(a.right, held.right);
    }

    #[test]
    fn world_pose_smoothing_removes_sample_velocity_jumps_without_lens_pumping() {
        let mut presentation = CameraPresentation::default();
        let mut scene = scene();
        let mut previous_raw = 0.0;
        let mut previous_smooth = 0.0;
        let mut previous_raw_velocity = 0.0;
        let mut previous_smooth_velocity = 0.0;
        let mut max_raw_jump = 0.0_f64;
        let mut max_smooth_jump = 0.0_f64;
        let mut max_offset = 0.0_f32;
        let mut aperture: Option<f32> = None;
        for frame in 0..=360 {
            let time = f64::from(frame) / 60.0;
            let sample_time = (time * 30.0 + 0.000001).floor() / 30.0;
            let target = [0.13 * sample_time, 0.02 * sample_time];
            let origin = [
                -1.0 + (target[0] / (2.0 / 96.0)).round() * (2.0 / 96.0),
                -1.0 + (target[1] / (2.0 / 64.0)).round() * (2.0 / 64.0),
            ];
            let elapsed = (time * 2.4) as f32;
            scene.camera_roll = elapsed * 4.0_f32.to_radians();
            let mut camera = world_camera(&scene, elapsed, target, origin);
            presentation.prepare(&mut camera, elapsed, 1.0 / 60.0, &scene, [-0.03, 0.03]);
            let smooth = origin[0] + 1.0 + f64::from(camera.ground[0]);
            let raw_velocity = (target[0] - previous_raw) * 60.0;
            let smooth_velocity = (smooth - previous_smooth) * 60.0;
            if frame > 2 {
                max_raw_jump = max_raw_jump.max((raw_velocity - previous_raw_velocity).abs());
                max_smooth_jump =
                    max_smooth_jump.max((smooth_velocity - previous_smooth_velocity).abs());
            }
            previous_raw = target[0];
            previous_smooth = smooth;
            previous_raw_velocity = raw_velocity;
            previous_smooth_velocity = smooth_velocity;
            max_offset = max_offset.max(camera.ground[0].hypot(camera.ground[1]));
            if let Some(aperture) = aperture {
                assert!((camera.half_w - aperture).abs() < 0.000001);
            } else {
                aperture = Some(camera.half_w);
            }
        }
        println!(
            "raw_velocity_jump_m_s={max_raw_jump}; smooth_velocity_jump_m_s={max_smooth_jump}; \
             target_residual_m={max_offset}"
        );
        assert!(max_offset < Camera::TARGET_RESERVE);
        assert!(max_smooth_jump < max_raw_jump * 0.15);
    }

    #[test]
    fn hidden_terminal_restores_at_the_current_world_window() {
        let water = Water::new(31, 17);
        let mut renderer = Renderer::new(17, 9);
        renderer.set_quality(RenderQuality::Balanced);
        renderer.set_world_view([0.0; 2], [-1.0; 2], 7);
        renderer.render_at(&water, 0.0, 1.0 / 60.0).unwrap();
        renderer.resize(0, 0);
        // Physics and travel continue while a hidden terminal skips rendering.
        renderer.render_at(&water, 72.0, 1.0 / 60.0).unwrap();
        renderer.resize(17, 9);
        renderer.set_world_view([3.0, -1.2], [2.0, -2.2], 7);
        renderer.render_at(&water, 72.0, 1.0 / 60.0).unwrap();
        let restored = renderer.prev_cam.unwrap();
        assert!(
            restored.ground[0].hypot(restored.ground[1]) < Camera::TARGET_RESERVE,
            "restored camera retained the hidden patch's pose: {:?}",
            restored.ground
        );
        for u in [0.0, 0.5, 1.0] {
            for v in [0.0, 0.5, 1.0] {
                let hit = restored.hit_water(restored.ray(u, v)).unwrap();
                assert!(
                    hit.into_iter()
                        .all(|x| x.is_finite() && (0.03..=0.97).contains(&x))
                );
            }
        }
        // Ordinary visible resizes still preserve smoothing state.
        renderer.resize(23, 11);
        renderer.set_world_view([3.01, -1.2], [2.0, -2.2], 7);
        renderer.render_at(&water, 72.04, 1.0 / 60.0).unwrap();
        let next = renderer.prev_cam.unwrap();
        assert!(next.ground[0] > 0.0 && next.ground[0] < 0.009);
    }

    #[test]
    fn emergency_fit_releases_gradually_and_keeps_displaced_water_inside_the_view() {
        let scene = scene();
        let mut presentation = CameraPresentation::default();
        let make = || world_camera(&scene, 0.0, [0.0; 2], [-1.0; 2]);
        let mut normal = make();
        presentation.prepare(&mut normal, 0.0, 0.0, &scene, [-0.03, 0.03]);
        let mut extreme = make();
        presentation.prepare(&mut extreme, 0.0, 1.0 / 60.0, &scene, [-0.4, 0.03]);
        let mut release = make();
        presentation.prepare(&mut release, 0.0, 1.0 / 60.0, &scene, [-0.03, 0.03]);
        assert!(extreme.half_w < release.half_w && release.half_w < normal.half_w);
        assert!(release.half_w - extreme.half_w < (normal.half_w - extreme.half_w) * 0.02);

        let mut intersections = 0;
        for time in (0..=180).step_by(6) {
            for size in [(200, 60), (80, 80), (1, 80), (400, 1)] {
                for elevation in [0.45, 0.8, 1.1] {
                    for target in [[0.0, 0.0], [0.03, -0.02], [0.08, 0.02]] {
                        for bounds in [[-0.16, 0.12], [-0.35, 0.25], [-0.55, 0.60]] {
                            let mut scene = scene;
                            scene.camera_elevation = elevation;
                            scene.camera_roll = time as f32 * 0.13;
                            let mut camera = Camera::new(time as f32, size.0, size.1, &scene);
                            camera.place_in_world(target, [-1.0; 2]);
                            camera.fit_surface(bounds);
                            for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                                for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                                    let ray = camera.ray(u, v);
                                    assert!(ray[2] < 0.0);
                                    for height in bounds {
                                        let distance = (height - camera.pos[2]) / ray[2];
                                        assert!(distance > 0.0);
                                        for (axis, &direction) in ray.iter().enumerate().take(2) {
                                            let hit = camera.pos[axis] + distance * direction;
                                            assert!(
                                                hit.abs() <= PATCH_LENGTH * 0.47001,
                                                "unsafe cone: time={time}, size={size:?}, \
                                                 elevation={elevation}, target={target:?}, \
                                                 bounds={bounds:?}, hit={hit}"
                                            );
                                        }
                                        intersections += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("safe_frustum_intersections={intersections}");
    }
}

// ── bilinear sampling ───────────────────────────────────────────────

#[inline(always)]
fn sample(
    levels: &Array2<f32>,
    wx: f32,
    wy: f32,
) -> f32 {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    let fx = (wx * gw as f32).clamp(0.0, gw as f32 - 1.0);
    let fy = (wy * gh as f32).clamp(0.0, gh as f32 - 1.0);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(gw - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let a = levels[(x0, y0)] + (levels[(x1, y0)] - levels[(x0, y0)]) * tx;
    let b = levels[(x0, y1)] + (levels[(x1, y1)] - levels[(x0, y1)]) * tx;
    a + (b - a) * ty
}

// ── precomputed grids (metric normals + refractive caustics) ─────────
// Computed once per frame from the level grid using direct neighbor access
// (no bilinear sampling). Shading then bilinearly samples these grids,
// replacing ~40 bilinear level lookups per shade_water call with ~5.

/// One set of coordinates serves height, normal, and curvature at a water hit.
#[derive(Clone, Copy)]
struct SampleFootprint {
    indices: [usize; 4],
    tx: f32,
    ty: f32,
}

fn backtrack_surface(
    point: [f32; 2],
    footprint: SampleFootprint,
    velocity: &[[f32; 2]],
    dt: f32,
    coverage: f32,
) -> [f32; 2] {
    let [a, b, c, d] = footprint.indices.map(|i| velocity[i]);
    std::array::from_fn(|axis| {
        let upper = a[axis] + (b[axis] - a[axis]) * footprint.tx;
        let lower = c[axis] + (d[axis] - c[axis]) * footprint.tx;
        point[axis] - (upper + (lower - upper) * footprint.ty) * dt * coverage / WATER_SCALE
    })
}

impl SampleFootprint {
    #[inline(always)]
    fn new(
        gw: usize,
        gh: usize,
        wx: f32,
        wy: f32,
    ) -> Self {
        let fx = (wx * gw as f32).clamp(0.0, gw as f32 - 1.0);
        let fy = (wy * gh as f32).clamp(0.0, gh as f32 - 1.0);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let x1 = (x0 + 1).min(gw - 1);
        let y1 = (y0 + 1).min(gh - 1);
        Self {
            indices: [x0 * gh + y0, x1 * gh + y0, x0 * gh + y1, x1 * gh + y1],
            tx: fx - x0 as f32,
            ty: fy - y0 as f32,
        }
    }

    #[inline(always)]
    fn scalar(
        self,
        grid: &[f32],
    ) -> f32 {
        // Water owns contiguous gw*gh grids; the constructor clamps both neighbors.
        unsafe {
            let [a, b, c, d] = self.indices.map(|index| *grid.get_unchecked(index));
            let upper = a + (b - a) * self.tx;
            let lower = c + (d - c) * self.tx;
            upper + (lower - upper) * self.ty
        }
    }

    #[inline(always)]
    fn normal(
        self,
        grid: &[[f32; 4]],
    ) -> [f32; 4] {
        unsafe {
            let [a, b, c, d] = self.indices.map(|index| grid.get_unchecked(index));
            std::array::from_fn(|i| {
                let upper = a[i] + (b[i] - a[i]) * self.tx;
                let lower = c[i] + (d[i] - c[i]) * self.tx;
                upper + (lower - upper) * self.ty
            })
        }
    }
}

#[inline(always)]
fn sample_grid4(
    grid: &[[f32; 4]],
    gw: usize,
    gh: usize,
    wx: f32,
    wy: f32,
) -> [f32; 4] {
    SampleFootprint::new(gw, gh, wx, wy).normal(grid)
}

#[inline(always)]
fn sample_grid1(
    grid: &[f32],
    gw: usize,
    gh: usize,
    wx: f32,
    wy: f32,
) -> f32 {
    SampleFootprint::new(gw, gh, wx, wy).scalar(grid)
}

/// Precompute Scharr normals + slope variance for the entire level grid.
/// Output layout: flat Vec, index = x * gh + y. Each element = [nx, ny, nz, slope_var].
fn precompute_normals(
    levels: &Array2<f32>,
    out: &mut Vec<[f32; 4]>,
) {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    out.resize(gw * gh, [0.0; 4]);
    for x in 0..gw {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(gw - 1);
        let x_strength = gw as f32 / (PATCH_LENGTH * (xp - xm).max(1) as f32);
        for y in 0..gh {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(gh - 1);
            let y_strength = gh as f32 / (PATCH_LENGTH * (yp - ym).max(1) as f32);
            // Scharr 3×3: direct neighbor access (no bilinear needed)
            unsafe {
                let tl = *levels.uget((xm, ym));
                let ml = *levels.uget((xm, y));
                let bl = *levels.uget((xm, yp));
                let tm = *levels.uget((x, ym));
                let bm = *levels.uget((x, yp));
                let tr = *levels.uget((xp, ym));
                let mr = *levels.uget((xp, y));
                let br = *levels.uget((xp, yp));
                let center = *levels.uget((x, y));

                let ddx = (tr * 3.0 + mr * 10.0 + br * 3.0 - tl * 3.0 - ml * 10.0 - bl * 3.0)
                    * (1.0 / 16.0)
                    * x_strength;
                let ddy = (bl * 3.0 + bm * 10.0 + br * 3.0 - tl * 3.0 - tm * 10.0 - tr * 3.0)
                    * (1.0 / 16.0)
                    * y_strength;
                // Slope is orientation, not roughness. Only variation in neighboring
                // normals broadens a highlight; a perfectly tilted plane stays glossy.
                let curvature_x = (mr + ml - 2.0 * center) * x_strength;
                let curvature_y = (bm + tm - 2.0 * center) * y_strength;
                let variance = (curvature_x * curvature_x + curvature_y * curvature_y) * 0.25;
                let il = 1.0 / (ddx * ddx + ddy * ddy + 1.0).sqrt();
                *out.get_unchecked_mut(x * gh + y) = [-ddx * il, -ddy * il, il, variance];
            }
        }
    }
}

/// Paraxial bottom irradiance from the full refractive Jacobian.
/// Finite emitter/resolution regularization bounds focal singularities.
fn precompute_caustics(
    levels: &Array2<f32>,
    depths: Option<&[f32]>,
    out: &mut Vec<f32>,
) {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    out.resize(gw * gh, 0.0);
    for x in 0..gw {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(gw - 1);
        for y in 0..gh {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(gh - 1);
            unsafe {
                let c = *levels.uget((x, y));
                let dx = gw as f32 / PATCH_LENGTH;
                let dy = gh as f32 / PATCH_LENGTH;
                let xx = if x > 0 && x + 1 < gw {
                    (*levels.uget((xp, y)) + *levels.uget((xm, y)) - 2.0 * c) * dx * dx
                } else {
                    0.0
                };
                let yy = if y > 0 && y + 1 < gh {
                    (*levels.uget((x, yp)) + *levels.uget((x, ym)) - 2.0 * c) * dy * dy
                } else {
                    0.0
                };
                let xy = (*levels.uget((xp, yp)) - *levels.uget((xp, ym)) - *levels.uget((xm, yp))
                    + *levels.uget((xm, ym)))
                    * dx
                    * dy
                    / ((xp - xm).max(1) * (yp - ym).max(1)) as f32;
                let depth = depths.map_or_else(
                    || (DEPTH + c).max(0.001),
                    |depths| depths[x * gh + y].max(0.001),
                );
                let distance = depth * (1.0 - 1.0 / IOR_WATER);
                let determinant =
                    (1.0 + distance * xx) * (1.0 + distance * yy) - distance * distance * xy * xy;
                *out.get_unchecked_mut(x * gh + y) =
                    (1.0 / (determinant * determinant + 0.02).sqrt()).min(6.0);
            }
        }
    }
    // Finite-patch flux normalization: caustics redistribute incident energy.
    let mean = out.iter().sum::<f32>() / out.len().max(1) as f32;
    for value in out {
        *value /= mean.max(1e-6);
    }
}

#[cfg(test)]
fn bed_material(point: [f32; 2]) -> [f32; 4] {
    static BED: std::sync::OnceLock<crate::bed::BedWorld> = std::sync::OnceLock::new();
    BED.get_or_init(|| {
        let mut bed = crate::bed::BedWorld::new(7);
        bed.prepare([-1.0; 2], 7);
        bed
    })
    .sample_patch(point)
}

#[cfg(test)]
fn bed_albedo(point: [f32; 2]) -> f32 { bed_material(point)[0] }

/// Snell ray from air into water and its intersection with a local horizontal bed.
/// Depth is the actual vertical column. The generated bed continues beyond the
/// numerical water patch. Extreme grazing paths keep their optical length but
/// cap lateral lookup displacement at the fixed 1.5 m streaming halo; this local
/// plane approximation is not an unbounded bathymetric ray marcher.
#[inline]
fn refracted_bed_point(
    point: [f32; 2],
    normal: [f32; 3],
    incoming: [f32; 3],
    depth: f32,
    ior: f32,
) -> Option<([f32; 2], f32)> {
    let cosine = (-dot3(incoming, normal)).clamp(0.0, 1.0);
    let eta = 1.0 / ior;
    let transmitted = (1.0 - eta * eta * (1.0 - cosine * cosine)).max(0.0).sqrt();
    let ray: [f32; 3] =
        std::array::from_fn(|c| eta * incoming[c] + (eta * cosine - transmitted) * normal[c]);
    if ray[2] >= -0.001 {
        return None;
    }
    let path_over_depth = 1.0 / -ray[2];
    let displacement = [
        ray[0] * depth * path_over_depth,
        ray[1] * depth * path_over_depth,
    ];
    let maximum = displacement[0].abs().max(displacement[1].abs());
    let scale = (crate::bed::REFRACTION_HALO / maximum.max(1e-6)).min(1.0);
    let point = std::array::from_fn(|c| point[c] + displacement[c] * scale / PATCH_LENGTH);
    Some((point, path_over_depth))
}

/// Beer attenuation on the illumination and view legs. Each leg samples actual
/// depth-averaged pigment at its midpoint; combining the integrated optical mass
/// gives one exponential per channel rather than two unrelated color filters.
#[inline]
fn bed_path_transmittance(
    material: MaterialGrid<'_>,
    view_footprint: SampleFootprint,
    light_footprint: SampleFootprint,
    depth: f32,
    view_path: f32,
    light_path: f32,
    absorption_scale: f32,
) -> [f32; 3] {
    let water = WaterTransport::new(
        depth * (view_path + light_path),
        absorption_scale,
        WATER_SCATTER,
    )
    .transmission;
    let Some(view_mass) = material.dye_mass(view_footprint) else {
        return water;
    };
    let light_mass = material.dye_mass(light_footprint).unwrap_or([0.0; 3]);
    let integrated = std::array::from_fn(|c| view_mass[c] * view_path + light_mass[c] * light_path);
    let pigment =
        crate::pigment_optics::transport(integrated, 1.0, [0.0; 3], [1.0; 3]).transmission;
    std::array::from_fn(|c| water[c] * pigment[c])
}

/// Neutral diffuse mineral bed with authored colored illumination and sparse
/// emissive veins. Camera-independent irradiance replaces the unphysical
/// flipped-sky boundary. Exit Fresnel is applied by the caller; entry Fresnel and
/// both Beer legs belong to diffuse reflection; emitted light takes only the view
/// leg. This is a 2-D column/local-bed approximation,
/// not ray tracing through arbitrary bathymetry or volumetric multiple scattering.
#[inline]
#[allow(clippy::too_many_arguments)]
fn submerged_bed_radiance(
    material: MaterialGrid<'_>,
    sky: &SkyEnvironment,
    gw: usize,
    gh: usize,
    point: [f32; 2],
    depth: f32,
    normal: [f32; 3],
    incoming: [f32; 3],
    light: [f32; 3],
    scene: &SceneFrame,
) -> [f32; 3] {
    let Some((bed, view_path)) = refracted_bed_point(point, normal, incoming, depth, IOR_WATER)
    else {
        return [0.0; 3];
    };
    let midpoint = std::array::from_fn::<_, 2, _>(|c| (point[c] + bed[c]) * 0.5);
    let view_footprint = SampleFootprint::new(gw, gh, midpoint[0], midpoint[1]);
    let bed_footprint = SampleFootprint::new(gw, gh, bed[0], bed[1]);
    let ambient = bed_path_transmittance(
        material,
        view_footprint,
        bed_footprint,
        depth,
        view_path,
        sky.bed_illumination_path,
        scene.absorption_scale,
    );
    let spectral_points = IOR_RGB.map(|ior| {
        refracted_bed_point(point, normal, incoming, depth, ior).map_or(bed, |hit| hit.0)
    });
    let bed_materials = spectral_points.map(|p| sky.bed.sample_patch(p));
    let albedo = bed_materials.map(|p| p[0]);
    let mut result = std::array::from_fn(|c| {
        albedo[c]
            * sky.bed_irradiance[c]
            * sky.bed_light_transmission(spectral_points[c], c)
            * ambient[c]
    });
    if sky.bed_emission_strength > 0.0 {
        // Authored fluorescent veins are a BED source, not light added at the
        // camera. They traverse only the view leg and the caller's exit Fresnel.
        let water = WaterTransport::new(depth * view_path, scene.absorption_scale, WATER_SCATTER);
        let beer = water.transmission;
        let pigment = material
            .dye_transport(view_footprint, view_path, water.tau, beer)
            .transmission;
        for c in 0..3 {
            result[c] += bed_materials[c][c + 1] * sky.bed_emission_strength * beer[c] * pigment[c];
        }
    }
    let incidence = dot3(normal, light).max(0.0);
    if incidence > 0.0 && light[2] > 0.0 {
        let downward = [-light[0], -light[1], -light[2]];
        let eta = 1.0 / IOR_WATER;
        let cos_t = (1.0 - eta * eta * (1.0 - incidence * incidence))
            .max(0.0)
            .sqrt();
        let ray: [f32; 3] =
            std::array::from_fn(|c| eta * downward[c] + (eta * incidence - cos_t) * normal[c]);
        if ray[2] < -0.001 {
            let light_path = 1.0 / -ray[2];
            // Reverse the full downward ray from the bed before clamping the
            // midpoint. Clipping the displacement first biases boundary paths.
            let middle = std::array::from_fn::<_, 2, _>(|c| {
                (bed[c] - 0.5 * ray[c] * depth * light_path / PATCH_LENGTH).clamp(0.001, 0.999)
            });
            let light_footprint = SampleFootprint::new(gw, gh, middle[0], middle[1]);
            let transmission = bed_path_transmittance(
                material,
                view_footprint,
                light_footprint,
                depth,
                view_path,
                light_path,
                scene.absorption_scale,
            );
            let entry = fresnel_rgb(incidence);
            for c in 0..3 {
                // The existing flux-normalized paraxial caustic grid is a bed
                // irradiance approximation. Chromatic Snell rays sample it at
                // distinct physical bed positions, without artificial RGB offsets.
                let caustic = material.bed_caustics.map_or(1.0, |grid| {
                    sample_grid1(grid, gw, gh, spectral_points[c][0], spectral_points[c][1])
                });
                result[c] += albedo[c]
                    * (DIRECT_RADIANCE / PI)
                    * light[2]
                    * (1.0 - entry[c])
                    * caustic
                    * transmission[c];
            }
        }
    }
    result
}

// ── water shading with secondary bounce ─────────────────────────────

fn surface_slope_bound(levels: &Array2<f32>) -> f32 {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    let mut dx = 0.0_f32;
    let mut dy = 0.0_f32;
    for x in 0..gw {
        for y in 0..gh {
            let value = levels[(x, y)];
            if x + 1 < gw {
                dx = dx.max((levels[(x + 1, y)] - value).abs() * gw as f32 / PATCH_LENGTH);
            }
            if y + 1 < gh {
                dy = dy.max((levels[(x, y + 1)] - value).abs() * gh as f32 / PATCH_LENGTH);
            }
        }
    }
    (dx * dx + dy * dy).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn trace_reflections(
    cam: &Camera,
    levels: &Array2<f32>,
    material: MaterialGrid<'_>,
    normals: &[[f32; 4]],
    sky: &SkyEnvironment,
    point: [f32; 2],
    height: f32,
    normal: [f32; 3],
    mut direction: [f32; 3],
    scene: &SceneFrame,
    bounds: [f32; 2],
    slope_bound: f32,
    limit: usize,
) -> (f32, f32, f32) {
    let gw = levels.shape()[0];
    let gh = levels.shape()[1];
    let mut origin = [
        (point[0] - 0.5) * PATCH_LENGTH + cam.patch_center[0],
        (point[1] - 0.5) * PATCH_LENGTH + cam.patch_center[1],
        height,
    ];
    for c in 0..3 {
        origin[c] += normal[c] * 0.0002;
    }
    let mut throughput = [1.0; 3];
    let mut radiance = [0.0; 3];
    for _ in 0..limit {
        // A global Lipschitz slope bound proves these rays cannot hit the surface.
        let horizontal = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if direction[2] >= slope_bound * horizontal {
            break;
        }
        let ray = Camera {
            pos: origin,
            ..*cam
        };
        let Some((wp, _)) = ray.hit_surface(direction, levels, bounds) else {
            break;
        };
        let h = sample(levels, wp[0], wp[1]);
        let ng = sample_grid4(normals, gw, gh, wp[0], wp[1]);
        let n = norm3([ng[0], ng[1], ng[2]]);
        let nv = (-dot3(direction, n)).max(0.0);
        if nv <= 1e-5 {
            break;
        }
        let footprint = SampleFootprint::new(gw, gh, wp[0], wp[1]);
        if let (Some(foam), Some(lighting)) = (material.foam, material.foam_light) {
            let coverage = footprint.scalar(foam).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let diffuse = lighting.radiance(n, dot3(n, scene.light_dir));
                // Dissolved pigment also occupies the thin liquid films around
                // bubbles. A short effective diffuse path tints aerated water;
                // leaving this white masked almost every drop in the roller.
                let pigment = material.dye_transmittance(footprint, 1.5);
                for c in 0..3 {
                    radiance[c] += throughput[c] * coverage * (diffuse[c] * pigment[c]);
                    throughput[c] *= 1.0 - coverage;
                }
                // Opaque foam terminates this path; partial coverage leaves the
                // same water transmission/reflection split on its clear area.
                if coverage >= 1.0 {
                    break;
                }
            }
        }
        let fresnel = thin_film_fresnel(
            fresnel_rgb(nv),
            nv,
            h,
            scene.interference_strength,
            scene.interference_phase,
        );
        let cos_t = (1.0 - (1.0 - nv * nv) / (IOR_WATER * IOR_WATER))
            .max(0.0)
            .sqrt();
        let depth = material.depth(footprint, h);
        let water = WaterTransport::new(
            depth / cos_t.max(0.001),
            scene.absorption_scale,
            WATER_SCATTER,
        );
        let dye = material.dye_transport(
            footprint,
            1.0 / cos_t.max(0.001),
            water.tau,
            water.transmission,
        );
        let incident = volume_incident_light(sky, wp, n, scene.light_dir);
        let bed = submerged_bed_radiance(
            material,
            sky,
            gw,
            gh,
            wp,
            depth,
            n,
            direction,
            scene.light_dir,
            scene,
        );
        for c in 0..3 {
            let local = bed[c] + water.scatter[c] * incident[c] * dye.volume[c];
            radiance[c] += throughput[c] * (1.0 - fresnel[c]) * local;
            throughput[c] *= fresnel[c];
        }
        direction = norm3(std::array::from_fn(|c| direction[c] + 2.0 * nv * n[c]));
        origin = [
            (wp[0] - 0.5) * PATCH_LENGTH + cam.patch_center[0],
            (wp[1] - 0.5) * PATCH_LENGTH + cam.patch_center[1],
            h,
        ];
        for c in 0..3 {
            origin[c] += n[c] * 0.0002;
        }
        if throughput.iter().copied().fold(0.0_f32, f32::max) < 0.001 {
            break;
        }
    }
    // Retain the unresolved tail rather than losing energy when the budget ends.
    let tail = sample_sky(sky, direction);
    let tail = [tail.0, tail.1, tail.2];
    for c in 0..3 {
        radiance[c] += throughput[c] * tail[c];
    }
    (radiance[0], radiance[1], radiance[2])
}

/// Constant-source isotropic single scattering. The cached environment phase
/// integral and the directional emitter are actual light sources; neither
/// absorption nor a fixed material tint can illuminate an otherwise dark scene.
fn volume_incident_light(
    sky: &SkyEnvironment,
    point: [f32; 2],
    normal: [f32; 3],
    light: [f32; 3],
) -> [f32; 3] {
    let incidence = dot3(normal, light).max(0.0);
    let direct = if incidence > 0.0 && light[2] > 0.0 {
        fresnel_rgb(incidence).map(|f| DIRECT_RADIANCE * (1.0 - f) / (4.0 * PI))
    } else {
        [0.0; 3]
    };
    std::array::from_fn(|c| {
        sky.volume_incident[c] * sky.bed_light_transmission(point, c) + direct[c]
    })
}

/// Shade a water hit point — energy-conserving microfacet BRDF.
///
/// Rendering equation decomposition:
///   L_o = env × F(θ_v)                                — environment reflection
///       + D·G·F(θ_h) / (4·n·v)                        — Cook-Torrance direct specular
///       + (1-F_v) × L_bed(two attenuated legs)       — illuminated diffuse bed
///       + (1-F_v) × sigma_s/sigma_t × (1-T) × J     — single scattering
/// Authored bed emission is included in L_bed and traverses only its view leg.
// Explicit borrowed grids let LLVM inline the hot sampling path without owning a scene graph.
#[allow(clippy::too_many_arguments)]
fn shade_water(
    levels: &Array2<f32>,
    material: MaterialGrid<'_>,
    normal_grid: &[[f32; 4]],
    caustic_grid: &[f32],
    sky_env: &SkyEnvironment,
    gw: usize,
    gh: usize,
    light: [f32; 3],
    cam: &Camera,
    dir: [f32; 3],
    wp: [f32; 2],
    scene: &SceneFrame,
    pixel_footprint: f32,
    height_bounds: [f32; 2],
    max_slope: f32,
    reflection_limit: usize,
) -> [f32; 3] {
    let material = MaterialGrid {
        bed_caustics: Some(caustic_grid),
        ..material
    };
    let footprint = SampleFootprint::new(gw, gh, wp[0], wp[1]);
    let level = footprint.scalar(levels.as_slice().expect("water grid is contiguous"));
    let ng = footprint.normal(normal_grid);
    let n = norm3([ng[0], ng[1], ng[2]]);

    let view = [-dir[0], -dir[1], -dir[2]];
    let n_dot_v = dot3(n, view).max(0.001);

    // Fresnel at view angle (env reflection / transmission split)
    let fr_v = thin_film_fresnel(
        fresnel_rgb(n_dot_v),
        n_dot_v,
        level,
        scene.interference_strength,
        scene.interference_phase,
    );

    let refl_dir = [
        dir[0] + 2.0 * n_dot_v * n[0],
        dir[1] + 2.0 * n_dot_v * n[1],
        dir[2] + 2.0 * n_dot_v * n[2],
    ];

    // Bounded reflected transport; almost all upward rays escape analytically.
    let env = trace_reflections(
        cam,
        levels,
        material,
        normal_grid,
        sky_env,
        wp,
        level,
        n,
        refl_dir,
        scene,
        height_bounds,
        max_slope,
        reflection_limit,
    );

    // ── Direct lighting ──
    let n_dot_l = dot3(n, light).max(0.0);
    let h = norm3([light[0] + view[0], light[1] + view[1], light[2] + view[2]]);
    let n_dot_h = dot3(n, h).max(0.0);
    let v_dot_h = dot3(view, h).max(0.0);

    // GGX NDF with slope-variance filtering
    let a2 = filtered_roughness_squared(ng, pixel_footprint);
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (PI * denom * denom);

    // Exact height-correlated Smith masking/shadowing for GGX.
    let geom = smith_ggx(n_dot_v, n_dot_l, a2);

    // Fresnel at half-vector angle (correct for microfacet specular)
    let fr_h = fresnel_rgb(v_dot_h);

    // Cook-Torrance: n·l cancels with rendering equation denominator
    // The emitter's radiance sets highlight energy. A pre-tonemap hard clamp made
    // a wide, perfectly flat white plateau instead of a continuous GGX lobe.
    let spec_base = d_ggx * geom * 0.18 / (4.0 * n_dot_v);
    let direct_spec = [
        fr_h[0] * spec_base,
        fr_h[1] * spec_base,
        fr_h[2] * spec_base,
    ];

    // Shared extinction on every water path; only sigma_s produces volume
    // radiance. Actual absorbing pigment uses the homogeneous escape integral.
    let eta = 1.0 / IOR_WATER;
    let cos_t = (1.0 - eta * eta * (1.0 - n_dot_v * n_dot_v))
        .max(0.0)
        .sqrt()
        .max(0.001);
    let depth = material.depth(footprint, level);
    let water = WaterTransport::new(depth / cos_t, scene.absorption_scale, WATER_SCATTER);
    let dye = material.dye_transport(footprint, 1.0 / cos_t, water.tau, water.transmission);
    let bed = submerged_bed_radiance(material, sky_env, gw, gh, wp, depth, n, dir, light, scene);
    let incident = volume_incident_light(sky_env, wp, n, light);
    let reflected = [env.0, env.1, env.2];
    let water_radiance = std::array::from_fn(|c| {
        reflected[c] * fr_v[c]
            + direct_spec[c]
            + (1.0 - fr_v[c]) * (bed[c] + water.scatter[c] * incident[c] * dye.volume[c])
    });
    let coverage = material
        .foam
        .map_or(0.0, |foam| footprint.scalar(foam).clamp(0.0, 1.0));
    if coverage > 0.0
        && let Some(lighting) = material.foam_light
    {
        let diffuse = lighting.radiance(n, n_dot_l);
        let pigment = material.dye_transmittance(footprint, 1.5);
        return mix_foam(
            water_radiance,
            std::array::from_fn(|c| diffuse[c] * pigment[c]),
            coverage,
        );
    }
    water_radiance
}

type SurfaceHit = Option<([f32; 2], f32)>;

/// Cast a single ray at (u, v), shade water/sky, apply aerial fog.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn shade_ray(
    cam: &Camera,
    levels: &Array2<f32>,
    material: MaterialGrid<'_>,
    normal_grid: &[[f32; 4]],
    caustic_grid: &[f32],
    sky_env: &SkyEnvironment,
    gw: usize,
    gh: usize,
    light: [f32; 3],
    u: f32,
    v: f32,
    scene: &SceneFrame,
    pixel_footprint: f32,
    height_bounds: [f32; 2],
    max_slope: f32,
    reflection_limit: usize,
) -> ([f32; 3], SurfaceHit) {
    let dir = cam.ray(u, v);
    let hit = cam.hit_surface(dir, levels, height_bounds);
    let color = if let Some((wp, distance)) = hit {
        let water = shade_water(
            levels,
            material,
            normal_grid,
            caustic_grid,
            sky_env,
            gw,
            gh,
            light,
            cam,
            dir,
            wp,
            scene,
            pixel_footprint / (-dir[2]).max(0.05),
            height_bounds,
            max_slope,
            reflection_limit,
        );
        if dir[2] < -1e-6 {
            let fog = 1.0 - (-distance * scene.fog_density).exp();
            // Smooth cached in-scattering. Flattening view.z to the horizon
            // made a near-vertical view unfold sky azimuth into radial spokes.
            let hz = sky_env.airlight;
            [
                water[0] + (hz[0] - water[0]) * fog,
                water[1] + (hz[1] - water[1]) * fog,
                water[2] + (hz[2] - water[2]) * fog,
            ]
        } else {
            water
        }
    } else {
        let s = sample_sky(sky_env, dir);
        [s.0, s.1, s.2]
    };
    (color, hit)
}

// ── bilinear HDR buffer sampling (for TAA reprojection) ─────────────

#[inline(always)]
fn sample_hdr(
    buf: &[[f32; 3]],
    w: usize,
    vh: usize,
    u: f32,
    v: f32,
) -> [f32; 3] {
    let fx = (u * w as f32 - 0.5).clamp(0.0, w as f32 - 1.0);
    let fy = (v * vh as f32 - 0.5).clamp(0.0, vh as f32 - 1.0);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(vh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut r = [0.0_f32; 3];
    for c in 0..3 {
        let a = buf[y0 * w + x0][c] * (1.0 - tx) + buf[y0 * w + x1][c] * tx;
        let b = buf[y1 * w + x0][c] * (1.0 - tx) + buf[y1 * w + x1][c] * tx;
        r[c] = a * (1.0 - ty) + b * ty;
    }
    r
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderQuality {
    High,
    Balanced,
    Low,
}

impl RenderQuality {
    pub fn name(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Balanced => "balanced",
            Self::Low => "low",
        }
    }
}

pub struct Renderer {
    world_view: Option<([f64; 2], [f64; 2], u64)>,
    retina: crate::retina::Retina,
    effect_time: f64,
    source_dirty: bool,
    camera_presentation: CameraPresentation,
    output: TerminalOutput,
    cells: Vec<Cell>,
    scene_transition: Option<SceneTransition>,
    has_frame: bool,
    prev_hdr_buf: Vec<[f32; 3]>,
    prev_cam: Option<Camera>,
    prev_water_time: Option<std::time::Duration>,
    hdr_buf: Vec<[f32; 3]>,
    display_buf: Vec<[f32; 3]>,
    display_scratch: Vec<[f32; 3]>,
    primary_hits: Vec<SurfaceHit>,
    scratch: Vec<[f32; 3]>,
    bloom_a: Vec<[f32; 3]>,
    bloom_b: Vec<[f32; 3]>,
    normal_grid: Vec<[f32; 4]>,
    caustic_grid: Vec<f32>,
    sky_envmap: SkyEnvironment,
    adapted_exposure: f32,
    source_pupil: f32,
    scene: SceneKind,
    intensity: f32,
    quality: RenderQuality,
    width: u16,
    height: u16,
    render_width: usize,
    render_height: usize,
    worker_count: usize,
    buf: Vec<u8>,
}

struct SceneTransition {
    from: Vec<Cell>,
    elapsed: f32,
}

const SCENE_TRANSITION_SECONDS: f32 = 2.0;

#[inline(always)]
fn mix_display_color(
    from: (u8, u8, u8),
    to: (u8, u8, u8),
    blend: f32,
) -> (u8, u8, u8) {
    if blend <= 0.0 {
        return from;
    }
    if blend >= 1.0 {
        return to;
    }
    let mix = |a: u8, b: u8| {
        let linear = decode_byte(a) + (decode_byte(b) - decode_byte(a)) * blend;
        (encode_srgb(linear) * 255.0).round() as u8
    };
    (mix(from.0, to.0), mix(from.1, to.1), mix(from.2, to.2))
}

impl Renderer {
    pub fn new(
        width: u16,
        height: u16,
    ) -> Self {
        Self {
            world_view: None,
            retina: crate::retina::Retina::new(),
            effect_time: 0.0,
            source_dirty: true,
            camera_presentation: CameraPresentation::default(),
            output: TerminalOutput::new(width, height),
            cells: vec![((0, 0, 0), (0, 0, 0)); width as usize * height as usize],
            scene_transition: None,
            has_frame: false,
            prev_hdr_buf: Vec::new(),
            prev_cam: None,
            prev_water_time: None,
            hdr_buf: Vec::new(),
            display_buf: Vec::new(),
            display_scratch: Vec::new(),
            primary_hits: Vec::new(),
            scratch: Vec::new(),
            bloom_a: Vec::new(),
            bloom_b: Vec::new(),
            normal_grid: Vec::new(),
            caustic_grid: Vec::new(),
            sky_envmap: SkyEnvironment::new(),
            adapted_exposure: 1.0,
            source_pupil: (1.0 / PUPIL_EXPOSURE_REFERENCE).sqrt(),
            scene: SceneKind::Journey,
            intensity: crate::scene::DEFAULT_INTENSITY,
            quality: RenderQuality::High,
            width,
            height,
            render_width: 0,
            render_height: 0,
            worker_count: std::thread::available_parallelism()
                .map_or(1, |count| count.get().min(4)),
            buf: Vec::with_capacity(65536),
        }
    }

    pub fn set_world_view(
        &mut self,
        center: [f64; 2],
        origin: [f64; 2],
        seed: u64,
    ) {
        if center.into_iter().chain(origin).all(f64::is_finite) {
            let view = Some((center, origin, seed));
            self.source_dirty |= self.world_view != view;
            self.world_view = view;
        }
    }

    pub fn resize(
        &mut self,
        width: u16,
        height: u16,
    ) {
        if self.width == width && self.height == height {
            return;
        }
        // Travel keeps advancing while a hidden window skips rendering. Seed
        // its first visible camera from the current patch, not the old world pose.
        if (self.width == 0 || self.height == 0) && width > 0 && height > 0 {
            self.camera_presentation.reset();
        }
        self.width = width;
        self.height = height;
        self.cells
            .resize(width as usize * height as usize, ((0, 0, 0), (0, 0, 0)));
        self.output.resize(width, height);
        self.prev_cam = None;
        self.scene_transition = None;
        self.has_frame = false;
        self.invalidate_geometry();
        self.retina.reset_history();
    }

    pub fn invalidate(&mut self) { self.output.invalidate(); }

    /// A held control may change the scene source without advancing its clocks.
    pub fn invalidate_geometry(&mut self) { self.source_dirty = true; }

    pub fn set_trails(
        &mut self,
        enabled: bool,
    ) {
        self.retina.set_enabled(enabled);
        self.invalidate();
    }

    pub fn set_rgb_shift(
        &mut self,
        radius: f32,
    ) {
        self.retina.set_rgb_shift(radius);
        self.invalidate();
    }

    pub fn set_subjective_intensity(
        &mut self,
        intensity: f32,
    ) {
        self.retina.set_subjective_intensity(intensity);
        self.invalidate();
    }

    pub fn set_status(
        &mut self,
        status: Option<&str>,
    ) {
        self.output.set_status(status);
    }

    fn resolve_spatial_edges(&mut self) {
        let w = self.render_width;
        let h = self.render_height;
        if w < 3 || h < 3 {
            return;
        }
        self.scratch.copy_from_slice(&self.hdr_buf);
        let luma = |p: [f32; 3]| dot3(p, [0.2126, 0.7152, 0.0722]);
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                let center = self.hdr_buf[i];
                let neighbors = [
                    self.hdr_buf[i - 1],
                    self.hdr_buf[i + 1],
                    self.hdr_buf[i - w],
                    self.hdr_buf[i + w],
                ];
                let mut low = luma(center);
                let mut high = low;
                for neighbor in neighbors {
                    let value = luma(neighbor);
                    low = low.min(value);
                    high = high.max(value);
                }
                // Flat fields and gentle gradients retain their exact samples.
                // Only a strong local edge receives a small, energy-balanced resolve.
                if high - low > 0.035_f32.max(high * 0.14) {
                    self.scratch[i] = std::array::from_fn(|channel| {
                        let average = neighbors.iter().map(|p| p[channel]).sum::<f32>() * 0.25;
                        center[channel] + (average - center[channel]) * 0.24
                    });
                }
            }
        }
        std::mem::swap(&mut self.hdr_buf, &mut self.scratch);
    }

    pub fn set_quality(
        &mut self,
        quality: RenderQuality,
    ) {
        self.source_dirty |= self.quality != quality;
        self.quality = quality;
    }

    pub fn set_scene(
        &mut self,
        scene: SceneKind,
    ) {
        if scene != self.scene {
            // Capture the visible mixture, rather than the previous scene's name:
            // rapid repeated switches then begin at exactly the displayed frame.
            self.scene_transition = self.has_frame.then(|| SceneTransition {
                from: self.cells.clone(),
                elapsed: 0.0,
            });
            self.scene = scene;
            self.source_dirty = true;
            self.retina.reset_history();
            self.invalidate();
        }
    }

    pub fn set_intensity(
        &mut self,
        intensity: f32,
    ) {
        if intensity.is_finite()
            && self.intensity != intensity.clamp(0.0, crate::scene::MAX_INTENSITY)
        {
            self.intensity = intensity.clamp(0.0, crate::scene::MAX_INTENSITY);
            self.source_dirty = true;
            self.invalidate();
        }
    }

    /// RGB scanlines at terminal width × twice terminal height, useful for PPM export.
    pub fn snapshot_rgb(&self) -> Vec<u8> {
        let width = self.width as usize;
        let mut rgb = Vec::with_capacity(self.cells.len() * 6);
        if width == 0 {
            return rgb;
        }
        for row in self.cells.chunks(width) {
            for cell in row {
                rgb.extend_from_slice(&[cell.0.0, cell.0.1, cell.0.2]);
            }
            for cell in row {
                rgb.extend_from_slice(&[cell.1.0, cell.1.1, cell.1.2]);
            }
        }
        rgb
    }

    fn prepare_display(
        &mut self,
        grade: &DisplayTransform,
        vignette_strength: f32,
        elapsed: f32,
    ) {
        let out_w = self.width as usize;
        let out_h = self.height as usize * 2;
        self.display_buf.resize(out_w * out_h, [0.0; 3]);
        for y in 0..out_h {
            for x in 0..out_w {
                let u = (x as f32 + 0.5) / out_w as f32;
                let v = (y as f32 + 0.5) / out_h as f32;
                let hdr = if self.render_width == out_w && self.render_height == out_h {
                    self.hdr_buf[y * out_w + x]
                } else {
                    sample_hdr(&self.hdr_buf, self.render_width, self.render_height, u, v)
                };
                let dx = u * 2.0 - 1.0;
                let dy = v * 2.0 - 1.0;
                let vignette = 1.0 - vignette_strength * (dx * dx + dy * dy);
                self.display_buf[y * out_w + x] = hdr.map(|c| c * vignette);
            }
        }
        // Recombine displaced linear-HDR channels before one common display
        // transform. These output-only effects never feed TAA or reservoirs.
        self.retina.apply_channels(
            &mut self.display_buf,
            &mut self.display_scratch,
            (out_w, out_h),
            (out_w, out_h),
            elapsed,
            self.intensity,
        );
        self.retina.apply_chroma(
            &mut self.display_buf,
            (out_w, out_h),
            elapsed,
            self.intensity,
        );
        for pixel in &mut self.display_buf {
            *pixel = grade.apply_linear(*pixel);
        }
    }

    fn prepare_buffers(&mut self) {
        let divisor = if self.quality == RenderQuality::Low {
            2
        } else {
            1
        };
        let w = (self.width as usize).div_ceil(divisor);
        let h = (self.height as usize * 2).div_ceil(divisor);
        if self.render_width == w && self.render_height == h {
            return;
        }
        if self.prev_cam.is_some()
            && self.render_width > 0
            && self.render_height > 0
            && w > 0
            && h > 0
        {
            let mut resampled = vec![[0.0; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    resampled[y * w + x] = sample_hdr(
                        &self.prev_hdr_buf,
                        self.render_width,
                        self.render_height,
                        (x as f32 + 0.5) / w as f32,
                        (y as f32 + 0.5) / h as f32,
                    );
                }
            }
            self.prev_hdr_buf = resampled;
        } else {
            self.prev_hdr_buf.resize(w * h, [0.0; 3]);
            self.prev_cam = None;
        }
        self.render_width = w;
        self.render_height = h;
        self.hdr_buf.resize(w * h, [0.0; 3]);
        self.primary_hits.resize(w * h, None);
        self.scratch.resize(w * h, [0.0; 3]);
        self.bloom_a.resize(w.div_ceil(2) * h.div_ceil(2), [0.0; 3]);
        self.bloom_b.resize(self.bloom_a.len(), [0.0; 3]);
    }

    fn resolve_history(
        &mut self,
        cam: &Camera,
        dt: f32,
        water: &Water,
        height_bounds: [f32; 2],
    ) {
        let levels = &water.levels;
        let physical_dt = self.prev_water_time.map_or(0.0, |previous| {
            water.elapsed().saturating_sub(previous).as_secs_f32()
        });
        self.prev_water_time = Some(water.elapsed());
        let Some(pc) = self.prev_cam else {
            self.prev_hdr_buf.copy_from_slice(&self.hdr_buf);
            self.prev_cam = Some(*cam);
            return;
        };
        if dt == 0.0 {
            if self.source_dirty {
                self.prev_hdr_buf.copy_from_slice(&self.hdr_buf);
                self.prev_cam = Some(*cam);
            } else {
                self.hdr_buf.copy_from_slice(&self.prev_hdr_buf);
            }
            return;
        }
        let w = self.render_width;
        let h = self.render_height;
        // Keep the same adaptation time at both 30 and 60 Hz.
        let history_weight = TAA_KEEP.powf((dt * 30.0).clamp(0.25, 4.0));
        for y in 0..h {
            for x in 0..w {
                let index = y * w + x;
                let cur = self.hdr_buf[index];
                let dir = cam.ray((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                // Center-sampled qualities already solved this exact ray during
                // primary shading. High's four AA rays do not include its center.
                let water_hit = if self.quality == RenderQuality::High {
                    cam.hit_surface(dir, levels, height_bounds)
                } else {
                    self.primary_hits[index]
                };
                // Reproject an actual world point, including window-origin motion.
                let previous_dir = if let Some((wp, _)) = water_hit {
                    let previous_point = if water.has_numerical_flow() && physical_dt > 0.0 {
                        let footprint = SampleFootprint::new(
                            levels.shape()[0],
                            levels.shape()[1],
                            wp[0],
                            wp[1],
                        );
                        // Foam follows material motion; clear-water highlights and
                        // a stationary jump stay Eulerian. One coverage-weighted
                        // reprojection approximates this mixed pixel's history.
                        backtrack_surface(
                            wp,
                            footprint,
                            &water.surface_velocity,
                            physical_dt,
                            footprint.scalar(&water.foam).clamp(0.0, 1.0),
                        )
                    } else {
                        wp
                    };
                    cam.history_direction(&pc, previous_point, sample(levels, wp[0], wp[1]))
                } else {
                    Some(dir)
                };
                let mut filtered = cur;
                if let Some((u, v)) =
                    previous_dir.and_then(|previous_dir| pc.project_dir(previous_dir))
                {
                    let prev = sample_hdr(&self.prev_hdr_buf, w, h, u, v);
                    let mut lo = cur;
                    let mut hi = cur;
                    for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                        for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                            let neighbor = self.hdr_buf[ny * w + nx];
                            for c in 0..3 {
                                lo[c] = lo[c].min(neighbor[c]);
                                hi[c] = hi[c].max(neighbor[c]);
                            }
                        }
                    }
                    for c in 0..3 {
                        filtered[c] += (prev[c].clamp(lo[c], hi[c]) - cur[c]) * history_weight;
                    }
                }
                self.scratch[index] = filtered;
            }
        }
        std::mem::swap(&mut self.hdr_buf, &mut self.scratch);
        self.prev_hdr_buf.copy_from_slice(&self.hdr_buf);
        self.prev_cam = Some(*cam);
    }

    /// Adapt the pupil and resolve perceptual persistence in linear light.
    fn adapt_perception(
        &mut self,
        scene: &SceneFrame,
        elapsed: f32,
        dt: f32,
        transition_dt: f32,
    ) -> DisplayTransform {
        let (w, h) = (self.render_width, self.render_height);
        let mut log_sum = 0.0;
        let mut count = 0;
        for pixel in self.hdr_buf.iter().step_by(4) {
            let luma = dot3(*pixel, [0.2126, 0.7152, 0.0722]);
            log_sum += luma.max(0.001).ln();
            count += 1;
        }
        let average = (log_sum / count as f32).exp();
        let target = (AUTO_EXP_TARGET / average).clamp(AUTO_EXP_MIN, AUTO_EXP_MAX);
        self.adapted_exposure = adapt_exposure(self.adapted_exposure, target, dt);
        // The pupil attenuates incoming light before perceptual persistence.
        // Histories store exposed retinal light: opening the pupil later must
        // not brighten photons remembered from an earlier, brighter view.
        for pixel in &mut self.hdr_buf {
            for channel in pixel {
                *channel *= self.adapted_exposure;
            }
        }
        self.retina.apply(
            &mut self.hdr_buf,
            &mut self.scratch,
            (w, h),
            (self.width as usize, self.height as usize * 2),
            (elapsed, transition_dt),
            self.intensity,
        );
        self.retina.apply_diffraction(
            &mut self.hdr_buf,
            &mut self.scratch,
            (w, h),
            (self.width as usize, self.height as usize * 2),
            self.intensity,
        );
        DisplayTransform::new(scene, 1.0)
    }

    fn postprocess(
        &mut self,
        cam: &Camera,
        scene: &SceneFrame,
        dt: f32,
    ) {
        // Exposure adapts after this optical pass. A held redraw of the same
        // source must retain the pupil that originally formed its glare.
        if dt > 0.0 || self.source_dirty {
            self.source_pupil = (self.adapted_exposure / PUPIL_EXPOSURE_REFERENCE)
                .sqrt()
                .clamp(0.0, 1.0);
        }
        let w = self.render_width;
        let h = self.render_height;
        let bw = w.div_ceil(2);
        let bh = h.div_ceil(2);
        // Chromatic aberration now respects texel centers: zero strength is identity.
        if scene.chromatic_aberration > 0.00001 {
            for y in 0..h {
                let v = (y as f32 + 0.5) / h as f32;
                for x in 0..w {
                    let u = (x as f32 + 0.5) / w as f32;
                    let du = u - 0.5;
                    let dv = v - 0.5;
                    let shift = (du * du + dv * dv) * scene.chromatic_aberration;
                    let red = sample_hdr(&self.hdr_buf, w, h, u + du * shift, v + dv * shift)[0];
                    let blue = sample_hdr(&self.hdr_buf, w, h, u - du * shift, v - dv * shift)[2];
                    self.scratch[y * w + x] = [red, self.hdr_buf[y * w + x][1], blue];
                }
            }
            std::mem::swap(&mut self.hdr_buf, &mut self.scratch);
        }
        // Bright extraction at half resolution keeps bloom broad and makes its cost bounded.
        for y in 0..bh {
            for x in 0..bw {
                let mut color = [0.0; 3];
                let mut count = 0.0;
                for sy in y * 2..(y * 2 + 2).min(h) {
                    for sx in x * 2..(x * 2 + 2).min(w) {
                        let p = self.hdr_buf[sy * w + sx];
                        let y = crate::color_pipeline::luminance(p);
                        let gate = (1.0 - BLOOM_THRESHOLD / y.max(1e-6)).max(0.0);
                        for c in 0..3 {
                            color[c] += p[c] * gate;
                        }
                        count += 1.0;
                    }
                }
                self.bloom_a[y * bw + x] = color.map(|c| c / count);
            }
        }
        for y in 0..bh {
            for x in 0..bw {
                let mut sum = [0.0; 3];
                for (tap, weight) in BLOOM_KERNEL.iter().enumerate() {
                    let sx = (x as isize + tap as isize - 2).clamp(0, bw as isize - 1) as usize;
                    let p = self.bloom_a[y * bw + sx];
                    for c in 0..3 {
                        sum[c] += p[c] * weight;
                    }
                }
                self.bloom_b[y * bw + x] = sum;
            }
        }
        for y in 0..bh {
            for x in 0..bw {
                let mut sum = [0.0; 3];
                for (tap, weight) in BLOOM_KERNEL.iter().enumerate() {
                    let sy = (y as isize + tap as isize - 2).clamp(0, bh as isize - 1) as usize;
                    let p = self.bloom_b[sy * bw + x];
                    for c in 0..3 {
                        sum[c] += p[c] * weight;
                    }
                }
                self.bloom_a[y * bw + x] = sum;
            }
        }
        // Aperture diffraction redistributes a little bloom into fine horizontal
        // streaks and a weaker vertical cross. The normalized kernel preserves energy.
        let glare = 0.16 + (1.0 - self.source_pupil) * 0.12;
        for y in 0..bh {
            for x in 0..bw {
                let u = (x as f32 + 0.5) / bw as f32;
                let v = (y as f32 + 0.5) / bh as f32;
                let center = self.bloom_a[y * bw + x];
                let mut result = [0.0; 3];
                for c in 0..3 {
                    // Longer wavelengths make wider diffraction wings.
                    let scale = [640.0_f32 / 545.0, 1.0, 450.0 / 545.0][c];
                    let mut streak = 0.0;
                    for offset in [-13.0_f32, -7.0, -3.0, 3.0, 7.0, 13.0] {
                        streak +=
                            sample_hdr(&self.bloom_a, bw, bh, u + offset * scale / bw as f32, v)[c];
                    }
                    for offset in [-4.0_f32, 4.0] {
                        streak +=
                            sample_hdr(&self.bloom_a, bw, bh, u, v + offset * scale / bh as f32)[c]
                                * 0.35;
                    }
                    for (dx, dy) in [(-5.0_f32, -5.0), (5.0, -5.0), (-5.0, 5.0), (5.0, 5.0)] {
                        streak += sample_hdr(
                            &self.bloom_a,
                            bw,
                            bh,
                            u + dx * scale / bw as f32,
                            v + dy * scale / bh as f32,
                        )[c] * 0.20;
                    }
                    result[c] = center[c] + (streak / 7.5 - center[c]) * glare;
                }
                self.bloom_b[y * bw + x] = result;
            }
        }
        std::mem::swap(&mut self.bloom_a, &mut self.bloom_b);
        // Radial shafts follow the visible sun or its reflected image in the water.
        if let Some((sun_u, sun_v)) = projected_light_source(cam, scene.light_dir) {
            for y in 0..bh {
                for x in 0..bw {
                    let u = (x as f32 + 0.5) / bw as f32;
                    let v = (y as f32 + 0.5) / bh as f32;
                    let mut color = self.bloom_a[y * bw + x];
                    let mut total_weight = 1.0;
                    for tap in 1..=6 {
                        let t = tap as f32 * (0.5 / 6.0);
                        let p = sample_hdr(
                            &self.bloom_a,
                            bw,
                            bh,
                            u + (sun_u - u) * t,
                            v + (sun_v - v) * t,
                        );
                        let weight = 0.025 * (1.0 - t);
                        for c in 0..3 {
                            color[c] += p[c] * weight;
                        }
                        total_weight += weight;
                    }
                    self.bloom_b[y * bw + x] = color.map(|c| c / total_weight);
                }
            }
            std::mem::swap(&mut self.bloom_a, &mut self.bloom_b);
        }
        for y in 0..h {
            for x in 0..w {
                let bloom = sample_hdr(
                    &self.bloom_a,
                    bw,
                    bh,
                    (x as f32 + 0.5) / w as f32,
                    (y as f32 + 0.5) / h as f32,
                );
                let original = self.hdr_buf[y * w + x];
                let y_linear = crate::color_pipeline::luminance(original);
                let gate = (1.0 - BLOOM_THRESHOLD / y_linear.max(1e-6)).max(0.0);
                for (c, light) in bloom.iter().enumerate() {
                    self.hdr_buf[y * w + x][c] =
                        original[c] * (1.0 - gate * BLOOM_INTENSITY) + light * BLOOM_INTENSITY;
                }
            }
        }
    }

    /// Compute a complete scene frame independently of terminal presentation.
    pub fn render_at(
        &mut self,
        water: &Water,
        elapsed: f32,
        dt: f32,
    ) -> Result<()> {
        if self.width == 0 || self.height == 0 || water.levels.is_empty() {
            return Ok(());
        }
        self.prepare_buffers();
        let gap = dt.is_finite() && dt > 0.25;
        if gap {
            self.retina.reset_history();
            self.camera_presentation.reset();
            self.source_dirty = true;
        }
        let transition_dt = if dt.is_finite() && !gap {
            dt.max(0.0)
        } else {
            0.0
        };
        let dt = transition_dt;
        self.effect_time += f64::from(transition_dt);
        let scene = SceneFrame::new(self.scene, elapsed, self.intensity);
        let mut cam = Camera::new(elapsed, self.width, self.height, &scene);
        if let Some((center, origin, seed)) = self.world_view {
            cam.place_in_world(center, origin);
            self.sky_envmap.prepare_world_bed(origin, seed);
        }
        let levels = &water.levels;
        let gw = levels.shape()[0];
        let gh = levels.shape()[1];
        let material = if water.has_numerical_flow() {
            // SampleFootprint uses unchecked indexing after constructing bounded
            // coordinates, so enforce all material grids' shared shape here.
            assert_eq!(water.depths.len(), levels.len(), "water depth grid shape");
            assert_eq!(water.foam.len(), levels.len(), "water foam grid shape");
            assert_eq!(
                water.surface_velocity.len(),
                levels.len(),
                "water velocity grid shape"
            );
            MaterialGrid {
                depths: Some(&water.depths),
                foam: Some(&water.foam),
                foam_light: None,
                ..MaterialGrid::clear()
            }
        } else {
            MaterialGrid::clear()
        };
        let material = if water.has_dye() {
            assert_eq!(water.dye.len(), levels.len(), "water dye grid shape");
            MaterialGrid {
                dye: Some(&water.dye),
                ..material
            }
        } else {
            material
        };
        let height_bounds = levels
            .iter()
            .fold([f32::INFINITY, f32::NEG_INFINITY], |bounds, value| {
                [bounds[0].min(*value), bounds[1].max(*value)]
            });
        self.camera_presentation
            .prepare(&mut cam, elapsed, transition_dt, &scene, height_bounds);
        let max_slope = surface_slope_bound(levels);
        let reflection_limit = match self.quality {
            RenderQuality::High => 3,
            RenderQuality::Balanced => 2,
            RenderQuality::Low => 1,
        };
        precompute_normals(levels, &mut self.normal_grid);
        precompute_caustics(levels, material.depths, &mut self.caustic_grid);
        // Keep directional source detail consistent when quality changes.
        self.sky_envmap.update(64, elapsed, &scene);
        let foam_light = water
            .has_numerical_flow()
            .then(|| FoamLighting::from_sky(&self.sky_envmap));
        let material = MaterialGrid {
            foam_light: foam_light.as_ref(),
            ..material
        };
        let w = self.render_width;
        let h = self.render_height;
        let pixel_footprint = (2.0 * cam.half_w * gw as f32 / w as f32)
            .max(2.0 * cam.half_h * gh as f32 / h as f32)
            * cam.pos[2]
            / WATER_SCALE;
        const HIGH_AA: [(f32, f32); 4] = [
            (-0.375, -0.125),
            (0.125, -0.375),
            (-0.125, 0.375),
            (0.375, 0.125),
        ];
        const CENTER_AA: [(f32, f32); 1] = [(0.0, 0.0)];
        let offsets: &[(f32, f32)] = if self.quality == RenderQuality::High {
            &HIGH_AA
        } else {
            &CENTER_AA
        };
        let sample_weight = 1.0 / offsets.len() as f32;
        let cache_primary_hit = self.quality != RenderQuality::High;
        let normal_grid = &self.normal_grid;
        let caustic_grid = &self.caustic_grid;
        let sky = &self.sky_envmap;
        let shade_rows = |first_row: usize, pixels: &mut [[f32; 3]], hits: &mut [SurfaceHit]| {
            for (row_index, (row, row_hits)) in
                pixels.chunks_mut(w).zip(hits.chunks_mut(w)).enumerate()
            {
                let y = first_row + row_index;
                for (x, (pixel, primary_hit)) in row.iter_mut().zip(row_hits).enumerate() {
                    let u = (x as f32 + 0.5) / w as f32;
                    let v = (y as f32 + 0.5) / h as f32;
                    let mut color = [0.0; 3];
                    *primary_hit = None;
                    for &(offset_x, offset_y) in offsets {
                        let (p, hit) = shade_ray(
                            &cam,
                            levels,
                            material,
                            normal_grid,
                            caustic_grid,
                            sky,
                            gw,
                            gh,
                            scene.light_dir,
                            u + offset_x / w as f32,
                            v + offset_y / h as f32,
                            &scene,
                            pixel_footprint,
                            height_bounds,
                            max_slope,
                            reflection_limit,
                        );
                        if cache_primary_hit {
                            *primary_hit = hit;
                        }
                        for c in 0..3 {
                            color[c] += p[c] * sample_weight;
                        }
                    }
                    *pixel = color;
                }
            }
        };
        // Scoped workers only own disjoint rows. All optical inputs are immutable,
        // so serial and parallel paths perform exactly the same floating-point work.
        if self.worker_count > 1 && w * h >= 12_000 && self.quality != RenderQuality::Low {
            let rows_per_worker = h.div_ceil(self.worker_count);
            std::thread::scope(|scope| {
                for (tile, (pixels, hits)) in self
                    .hdr_buf
                    .chunks_mut(rows_per_worker * w)
                    .zip(self.primary_hits.chunks_mut(rows_per_worker * w))
                    .enumerate()
                {
                    scope.spawn(move || shade_rows(tile * rows_per_worker, pixels, hits));
                }
            });
        } else {
            shade_rows(0, &mut self.hdr_buf, &mut self.primary_hits);
        }
        if self.quality != RenderQuality::High {
            self.resolve_spatial_edges();
        }
        self.resolve_history(&cam, dt, water, height_bounds);
        self.postprocess(&cam, &scene, dt);
        let grade = self.adapt_perception(&scene, self.effect_time as f32, dt, transition_dt);
        let pupil = (self.adapted_exposure / PUPIL_EXPOSURE_REFERENCE)
            .sqrt()
            .clamp(0.0, 1.0);
        let vignette_strength = 0.025 + pupil * 0.04;
        let out_w = self.width as usize;
        self.prepare_display(&grade, vignette_strength, self.effect_time as f32);
        let transition_blend = self.scene_transition.as_mut().map(|transition| {
            transition.elapsed += transition_dt;
            let t = (transition.elapsed / SCENE_TRANSITION_SECONDS).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        });
        for y in 0..self.height as usize {
            for x in 0..out_w {
                let mut halves = [(0, 0, 0); 2];
                for (half, color) in halves.iter_mut().enumerate() {
                    let vy = y * 2 + half;
                    let rgb = self.display_buf[vy * out_w + x].map(encode_srgb);
                    // Spatial noise is temporally stable; static cells remain eligible for diff output.
                    let dither = ign(x as f32, vy as f32, 0.0);
                    *color = (
                        (rgb[0] * 255.0 + dither + 0.5).clamp(0.0, 255.0) as u8,
                        (rgb[1] * 255.0 + dither + 0.5).clamp(0.0, 255.0) as u8,
                        (rgb[2] * 255.0 + dither + 0.5).clamp(0.0, 255.0) as u8,
                    );
                }
                let previous = self.cells[y * out_w + x];
                if let (Some(transition), Some(blend)) = (&self.scene_transition, transition_blend)
                {
                    let from = transition.from[y * out_w + x];
                    halves[0] = mix_display_color(from.0, halves[0], blend);
                    halves[1] = mix_display_color(from.1, halves[1], blend);
                }
                self.cells[y * out_w + x] = (
                    stable_color(halves[0], previous.0),
                    stable_color(halves[1], previous.1),
                );
            }
        }
        if transition_blend == Some(1.0) {
            self.scene_transition = None;
        }
        self.has_frame = true;
        self.source_dirty = false;
        Ok(())
    }

    pub fn snapshot_cells(&self) -> Vec<Cell> { self.cells.clone() }

    pub fn status(&self) -> Option<String> { self.output.status().map(str::to_owned) }

    /// Headless export and benchmarks retain the synchronous complete-frame path.
    pub fn draw_at<W: Write>(
        &mut self,
        writer: &mut W,
        water: &Water,
        elapsed: f32,
        dt: f32,
    ) -> Result<()> {
        if self.width == 0 || self.height == 0 || water.levels.is_empty() {
            return Ok(());
        }
        self.render_at(water, elapsed, dt)?;
        self.output.encode(&self.cells, &mut self.buf);
        if !self.buf.is_empty() {
            writer.write_all(&self.buf)?;
            writer.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod display_channel_tests {
    use super::*;

    #[test]
    fn held_hdr_glint_reuses_the_source_pupil_after_exposure_adapts() {
        let mut renderer = Renderer::new(41, 20);
        renderer.prepare_buffers();
        let scene = SceneFrame::new(SceneKind::Prism, 7.0, 4.0);
        let camera = Camera::new(7.0, 41, 20, &scene);
        renderer.hdr_buf.fill([0.001; 3]);
        renderer.hdr_buf[20 * 41 + 20] = [32.0, 24.0, 16.0];
        let source = renderer.hdr_buf.clone();
        renderer.postprocess(&camera, &scene, 0.1);
        let first_glare = renderer.hdr_buf.clone();
        let previous_exposure = renderer.adapted_exposure;
        renderer.adapt_perception(&scene, 0.1, 0.1, 0.1);
        assert!(
            (renderer.adapted_exposure - previous_exposure).abs() > 0.1,
            "fixture must leave the pupil visibly adapting"
        );
        renderer.source_dirty = false;
        for _ in 0..3 {
            renderer.hdr_buf.copy_from_slice(&source);
            renderer.postprocess(&camera, &scene, 0.0);
            let maximum_change = renderer
                .hdr_buf
                .iter()
                .zip(&first_glare)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                .fold(0.0_f32, f32::max);
            assert_eq!(
                maximum_change, 0.0,
                "held glare changed after pupil adaptation"
            );
        }
        // A real source edit while held must still use the current pupil.
        renderer.source_dirty = true;
        renderer.hdr_buf.copy_from_slice(&source);
        renderer.postprocess(&camera, &scene, 0.0);
        assert_ne!(renderer.hdr_buf, first_glare);
    }

    #[test]
    fn paused_redraw_preserves_image_exposure_and_source_history() {
        let mut water = Water::new(24, 12);
        water.add_drop(0.4, 0.5, 0.2);
        let mut renderer = Renderer::new(24, 12);
        for frame in 0..4 {
            water.update(std::time::Duration::from_millis(16));
            renderer
                .render_at(&water, 7.0 + frame as f32 * 0.04, 1.0 / 60.0)
                .unwrap();
        }
        let exposure = renderer.adapted_exposure;
        let history = renderer.prev_hdr_buf.clone();
        let image = renderer.snapshot_rgb();
        for _ in 0..5 {
            renderer.set_status(Some("paused status redraw"));
            renderer.render_at(&water, 7.12, 0.0).unwrap();
            assert_eq!(
                renderer.adapted_exposure, exposure,
                "A held pupil must not keep adapting"
            );
            assert_eq!(
                renderer.prev_hdr_buf, history,
                "Held redraw must not feed TAA back into itself"
            );
            assert_eq!(
                renderer.snapshot_rgb(),
                image,
                "HUD-only redraw must preserve the held image"
            );
        }
    }

    #[test]
    fn held_subjective_controls_change_effects_without_advancing_the_scene() {
        let water = Water::new(32, 16);
        let mut renderer = Renderer::new(32, 16);
        renderer.render_at(&water, 7.0, 1.0 / 60.0).unwrap();
        let history = renderer.prev_hdr_buf.clone();
        let phase = renderer.effect_time;
        let exposure = renderer.adapted_exposure;
        renderer.set_subjective_intensity(0.0);
        renderer.render_at(&water, 7.0, 0.0).unwrap();
        let clean = renderer.snapshot_rgb();
        renderer.set_subjective_intensity(2.0);
        renderer.render_at(&water, 7.0, 0.0).unwrap();
        assert_ne!(
            renderer.snapshot_rgb(),
            clean,
            "LSD controls must reach the displayed image"
        );
        assert_eq!(renderer.prev_hdr_buf, history);
        assert_eq!(renderer.effect_time, phase);
        assert_eq!(renderer.adapted_exposure, exposure);
    }

    #[test]
    fn hidden_view_and_long_gaps_do_not_advance_subjective_phase() {
        let water = Water::new(16, 8);
        let mut renderer = Renderer::new(16, 8);
        renderer.render_at(&water, 7.0, 0.05).unwrap();
        let phase = renderer.effect_time;
        renderer.resize(0, 0);
        renderer.render_at(&water, 20.0, 30.0).unwrap();
        renderer.resize(16, 8);
        renderer.render_at(&water, 20.0, 30.0).unwrap();
        assert_eq!(renderer.effect_time, phase);
        renderer.render_at(&water, 20.01, 0.02).unwrap();
        assert!((renderer.effect_time - phase - 0.02).abs() < 1e-8);
        assert!(renderer.hdr_buf.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn first_frame_white_glint_gets_colored_wings_after_display_grade() {
        let mut renderer = Renderer::new(41, 20);
        renderer.prepare_buffers();
        renderer.hdr_buf.fill([0.02; 3]);
        renderer.hdr_buf[20 * 41 + 20] = [16.0; 3];
        let source = renderer.hdr_buf.clone();
        let grade = DisplayTransform::new(&SceneFrame::new(SceneKind::Acid, 0.0, 2.0), 1.0);
        renderer.prepare_display(&grade, 0.0, 7.0);
        let colorful_wings = renderer
            .display_buf
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                let encoded = p.map(encode_srgb);
                *i != 20 * 41 + 20
                    && encoded.iter().copied().fold(0.0_f32, f32::max)
                        - encoded.iter().copied().fold(1.0_f32, f32::min)
                        > 0.03
            })
            .count();
        assert!(
            colorful_wings >= 6,
            "only {colorful_wings} colored wing pixels"
        );
        assert_eq!(
            renderer.hdr_buf, source,
            "display warp must not feed HDR history"
        );
        renderer.set_trails(false);
        renderer.prepare_display(&grade, 0.0, 7.0);
        assert_eq!(
            renderer.display_buf,
            source
                .iter()
                .map(|p| grade.apply_linear(*p))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn display_buffers_follow_odd_low_quality_resizes_without_touching_render_scratch() {
        let mut renderer = Renderer::new(40, 20);
        renderer.set_quality(RenderQuality::Low);
        let grade = DisplayTransform::new(&SceneFrame::new(SceneKind::Acid, 0.0, 2.0), 1.0);
        for width in [40, 39, 40, 1, 41] {
            renderer.resize(width, 20);
            renderer.prepare_buffers();
            renderer.hdr_buf.fill([0.1; 3]);
            renderer.prepare_display(&grade, 0.0, 7.0);
            assert_eq!(renderer.display_buf.len(), width as usize * 40);
            assert_eq!(
                renderer.scratch.len(),
                renderer.render_width * renderer.render_height
            );
        }
    }
}
