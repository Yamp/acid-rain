//! A finite moving cache of a deterministic, infinite procedural bed.

const SIZE: usize = 336;
const RAW_SIZE: usize = SIZE + 4;
const TEXELS_PER_METRE: f64 = 64.0;
pub const REFRACTION_HALO: f32 = 1.5;

/// The filtered lattice has a 1.5 m refracted-ray halo on every side of the
/// 2 m water patch. Raw and filtered rings keep a fixed allocation at any travel
/// distance. Integer world addresses prevent pattern repetition at ring wraps.
pub struct BedWorld {
    seed: u64,
    origin: [f64; 2],
    minimum: Option<[i64; 2]>,
    offset: [f32; 2],
    ring_origin: [usize; 2],
    raw: Vec<[f32; 4]>,
    filtered: Vec<[f32; 4]>,
    #[cfg(test)]
    generated: usize,
}

impl BedWorld {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            origin: [-1.0; 2],
            minimum: None,
            offset: [0.0; 2],
            ring_origin: [0; 2],
            raw: vec![[0.0; 4]; RAW_SIZE * RAW_SIZE],
            filtered: vec![[0.0; 4]; SIZE * SIZE],
            #[cfg(test)]
            generated: 0,
        }
    }

    /// Reuse every overlapping texel. New work is linear in the entering strips,
    /// or one finite full cache after a teleport/seed change. Fractional camera
    /// motion changes only the bilinear sampling offset, never the material.
    pub fn prepare(
        &mut self,
        origin: [f64; 2],
        seed: u64,
    ) {
        assert!(origin.into_iter().all(f64::is_finite));
        let minimum = origin
            .map(|v| ((v - f64::from(REFRACTION_HALO)) * TEXELS_PER_METRE).floor() as i64 - 4);
        let previous = if self.seed == seed {
            self.minimum
        } else {
            None
        };
        self.seed = seed;
        self.origin = origin;
        self.offset =
            std::array::from_fn(|c| (origin[c] * TEXELS_PER_METRE - minimum[c] as f64) as f32);
        self.ring_origin = minimum.map(|c| c.rem_euclid(SIZE as i64) as usize);
        if previous == Some(minimum) {
            return;
        }
        let raw_minimum = minimum.map(|v| v - 2);
        entering_cells(
            previous.map(|p| p.map(|v| v - 2)),
            raw_minimum,
            RAW_SIZE,
            |x, y| {
                self.raw[address(x, y, RAW_SIZE)] = material(
                    x as f64 / TEXELS_PER_METRE,
                    y as f64 / TEXELS_PER_METRE,
                    seed,
                );
                #[cfg(test)]
                {
                    self.generated += 1;
                }
            },
        );
        // Positive normalized 5x5 binomial filtering, matching the old ~4 cm
        // emission bandwidth. Its halo is generated from the same world field,
        // so a newly entering edge is identical to a cache interior.
        const KERNEL: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
        entering_cells(previous, minimum, SIZE, |x, y| {
            let mut p = [self.raw[address(x, y, RAW_SIZE)][0], 0.0, 0.0, 0.0];
            for (i, wx) in KERNEL.into_iter().enumerate() {
                for (j, wy) in KERNEL.into_iter().enumerate() {
                    let source = self.raw[address(x + i as i64 - 2, y + j as i64 - 2, RAW_SIZE)];
                    for c in 1..4 {
                        p[c] += wx * wy * source[c];
                    }
                }
            }
            self.filtered[address(x, y, SIZE)] = p;
        });
        self.minimum = Some(minimum);
    }

    /// Local water-patch UV may extend into the refracted-ray halo. The only
    /// per-ray work is four cache reads and bilinear interpolation.
    #[inline]
    pub fn sample_patch(
        &self,
        point: [f32; 2],
    ) -> [f32; 4] {
        debug_assert!(
            self.minimum.is_some(),
            "prepare the world bed before shading"
        );
        let coordinate: [f32; 2] = std::array::from_fn(|c| {
            (point[c] * 128.0 + self.offset[c]).clamp(0.0, (SIZE - 1) as f32 - 0.0001)
        });
        let local = coordinate.map(|v| v as usize);
        let fraction: [f32; 2] = std::array::from_fn(|c| coordinate[c] - local[c] as f32);
        let ring: [usize; 2] = std::array::from_fn(|c| {
            let cell = self.ring_origin[c] + local[c];
            if cell >= SIZE { cell - SIZE } else { cell }
        });
        let next = ring.map(|c| if c + 1 == SIZE { 0 } else { c + 1 });
        let a = self.filtered[ring[0] * SIZE + ring[1]];
        let b = self.filtered[next[0] * SIZE + ring[1]];
        let c = self.filtered[ring[0] * SIZE + next[1]];
        let d = self.filtered[next[0] * SIZE + next[1]];
        std::array::from_fn(|channel| {
            let lo = a[channel] + (b[channel] - a[channel]) * fraction[0];
            let hi = c[channel] + (d[channel] - c[channel]) * fraction[0];
            lo + (hi - lo) * fraction[1]
        })
    }

    pub fn origin(&self) -> [f64; 2] { self.origin }

    pub fn seed(&self) -> u64 { self.seed }

    #[cfg(test)]
    fn resident_texels(&self) -> usize { self.raw.len() + self.filtered.len() }
}

fn address(
    x: i64,
    y: i64,
    size: usize,
) -> usize {
    x.rem_euclid(size as i64) as usize * size + y.rem_euclid(size as i64) as usize
}

/// Iterate the rectangle difference without scanning or allocating the overlap.
fn entering_cells(
    previous: Option<[i64; 2]>,
    next: [i64; 2],
    size: usize,
    mut visit: impl FnMut(i64, i64),
) {
    let end = next.map(|v| v + size as i64);
    let Some(old) = previous else {
        for x in next[0]..end[0] {
            for y in next[1]..end[1] {
                visit(x, y);
            }
        }
        return;
    };
    let lo = std::array::from_fn::<_, 2, _>(|c| next[c].max(old[c]));
    let hi = std::array::from_fn::<_, 2, _>(|c| end[c].min(old[c] + size as i64));
    if lo[0] >= hi[0] || lo[1] >= hi[1] {
        for x in next[0]..end[0] {
            for y in next[1]..end[1] {
                visit(x, y);
            }
        }
        return;
    }
    for x in (next[0]..lo[0]).chain(hi[0]..end[0]) {
        for y in next[1]..end[1] {
            visit(x, y);
        }
    }
    for x in lo[0]..hi[0] {
        for y in (next[1]..lo[1]).chain(hi[1]..end[1]) {
            visit(x, y);
        }
    }
}

fn hash(
    x: i64,
    y: i64,
    seed: u64,
) -> f32 {
    let mut n = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ seed;
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((n ^ (n >> 31)) >> 40) as f32 / 16_777_215.0
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn noise(
    x: f64,
    y: f64,
    seed: u64,
) -> f32 {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let smoother = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let u = smoother((x - ix as f64) as f32);
    let v = smoother((y - iy as f64) as f32);
    let a = hash(ix, iy, seed) * (1.0 - u) + hash(ix + 1, iy, seed) * u;
    let b = hash(ix, iy + 1, seed) * (1.0 - u) + hash(ix + 1, iy + 1, seed) * u;
    a * (1.0 - v) + b * v
}

fn material(
    x: f64,
    y: f64,
    seed: u64,
) -> [f32; 4] {
    let n = |x, y| noise(x, y, seed);
    let u = x + 0.19 * f64::from(n(x / 0.60 + 3.1, y / 0.60 + 1.9) - 0.5);
    let v = y + 0.19 * f64::from(n(x / 0.60 + 8.3, y / 0.60 + 7.1) - 0.5);
    let mineral = 0.52 * n(u / 0.50 + 2.73, v / 0.50 + 5.19)
        + 0.27 * n(u / 0.25 + 11.41, v / 0.25 + 7.83)
        + 0.14 * n(u / 0.12 + 1.37, v / 0.12 + 4.31)
        + 0.07 * n(u / 0.06 + 9.71, v / 0.06 + 3.13);
    // Seeded 2 m rooms contain broken spiral fossils. Their support dies before
    // the room boundary, where the uninterrupted mineral field takes over.
    let cell = [(u / 2.0).floor() as i64, (v / 2.0).floor() as i64];
    let dx = (u - (cell[0] as f64 + 0.5) * 2.0) as f32
        - 0.28 * (hash(cell[0], cell[1], seed ^ 0x195e) - 0.5);
    let dy = (v - (cell[1] as f64 + 0.5) * 2.0) as f32
        - 0.28 * (hash(cell[0], cell[1], seed ^ 0x792a) - 0.5);
    let radius = dx.hypot(dy);
    let support = smooth((0.78 - radius) / 0.18) * smooth(radius / 0.12);
    let hand = if hash(cell[0], cell[1], seed ^ 0x2a73) < 0.5 {
        -1.0
    } else {
        1.0
    };
    let spiral = (3.0 * dy.atan2(dx) + hand * radius * 20.0 + 2.0 * mineral)
        .sin()
        .abs();
    let fossil = smooth((0.42 - spiral) / 0.42) * support;
    // Two nearest jittered cells give a seam-free network of mineral plates.
    let (ix, iy) = (u.floor() as i64, v.floor() as i64);
    let (mut first, mut second) = (f32::INFINITY, f32::INFINITY);
    for cx in ix - 1..=ix + 1 {
        for cy in iy - 1..=iy + 1 {
            let dx = (u - cx as f64 - 0.1) as f32 - 0.8 * hash(cx, cy, seed ^ 0x8b61);
            let dy = (v - cy as f64 - 0.1) as f32 - 0.8 * hash(cx, cy, seed ^ 0xd583);
            let distance = dx * dx + dy * dy;
            if distance < first {
                second = first;
                first = distance;
            } else if distance < second {
                second = distance;
            }
        }
    }
    let plates = smooth((second.sqrt() - first.sqrt()) / 0.20);
    let region = smooth((n(u / 3.7 + 3.5, v / 3.7 + 2.1) - 0.28) / 0.44);
    let mineral = mineral + 0.16 * (plates - 0.5) * region + 0.12 * (fossil - 0.3) * (1.0 - region);
    let albedo = 0.10 + (0.90 - 0.10) * smooth((mineral - 0.25) / 0.50);
    let veins =
        0.68 * n(u / 0.56 + 1.91, v / 0.56 + 6.17) + 0.32 * n(u / 0.28 + 7.13, v / 0.28 + 2.71);
    let contour = ((veins * 3.5 + 0.17) * std::f32::consts::TAU).sin().abs();
    let band = smooth((0.42 - contour) / 0.42);
    let gate = smooth((n(u / 0.43 + 4.7, v / 0.43 + 8.2) - 0.50) / 0.23);
    let glow = 0.32 * ((1.0 - 0.6 * region) * band * gate + 0.6 * region * fossil).clamp(0.0, 1.0);
    let tint = n(u / 0.65 + 5.8, v / 0.65 + 2.6);
    // The accepted lilac/rose/cyan spectra, sparse emission, no neutral fill.
    [
        albedo,
        glow * (0.65 + 0.35 * tint),
        glow * (0.12 + 0.62 * (1.0 - tint)),
        glow,
    ]
}

/// Authored colored illumination in world coordinates. A continuous seeded
/// palette field replaces six lights that used to follow the camera. This is
/// an artistic incident-light filter, not extra material reflectance/emission.
pub fn projected_light(
    point: [f64; 2],
    phase: f32,
    seed: u64,
) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 6] = [
        [0.015, 0.78, 1.0], // cyan
        [0.04, 0.25, 1.0],  // blue
        [0.52, 0.08, 1.0],  // lilac
        [1.0, 0.06, 0.38],  // rose
        [1.0, 0.55, 0.04],  // amber
        [0.03, 1.0, 0.44],  // mint
    ];
    let (s, c) = phase.sin_cos();
    let x = point[0] + 0.12 * f64::from(c);
    let y = point[1] + 0.12 * f64::from(s);
    let warp = f64::from(noise(x / 0.70 + 2.1, y / 0.70 - 3.7, seed ^ 0xa349) - 0.5);
    let hue = (1.9 * noise(x / 2.3 + warp, y / 2.3 - warp, seed ^ 0x684b)
        + 0.24 * noise(x / 0.90 + 4.7, y / 0.90 - 1.3, seed ^ 0x95ef))
    .fract()
        * 6.0;
    let index = hue as usize;
    let weight = smooth(hue - index as f32);
    std::array::from_fn(|channel| {
        PALETTE[index][channel]
            + (PALETTE[(index + 1) % PALETTE.len()][channel] - PALETTE[index][channel]) * weight
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(bed: &BedWorld) -> Vec<[f32; 4]> {
        (0..35)
            .map(|i| bed.sample_patch([(i / 5) as f32 / 6.0, (i % 5) as f32 / 4.0]))
            .collect()
    }

    #[test]
    fn regions_have_bounded_material_and_visible_variation() {
        let mut bed = BedWorld::new(7);
        bed.prepare([0.0; 2], 7);
        let pixels = samples(&bed);
        for p in &pixels {
            assert!((0.10..=0.90).contains(&p[0]));
            assert!(
                p[1..]
                    .iter()
                    .all(|c| c.is_finite() && (0.0..=0.32).contains(c))
            );
        }
        let lo = pixels.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let hi = pixels
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(hi - lo > 0.25, "albedo variation {lo}..{hi}");
        assert!(pixels.iter().any(|p| p[1..].iter().any(|c| *c > 0.02)));
    }

    #[test]
    fn seed_and_world_position_survive_cache_eviction() {
        let mut a = BedWorld::new(7);
        a.prepare([-5.25, 7.125], 7);
        let before = samples(&a);
        a.prepare([20.0, -40.0], 7);
        a.prepare([-5.25, 7.125], 7);
        assert_eq!(before, samples(&a));
        let mut b = BedWorld::new(7);
        b.prepare([-5.25, 7.125], 7);
        assert_eq!(before, samples(&b));
        b.prepare([-5.25, 7.125], 8);
        assert_ne!(before, samples(&b));
    }

    #[test]
    fn overlapping_world_points_do_not_move_with_the_patch() {
        let mut bed = BedWorld::new(11);
        bed.prepare([-1.0; 2], 11);
        let before = bed.sample_patch([0.5, 0.5]);
        bed.prepare([-0.75, -1.375], 11);
        assert_eq!(before, bed.sample_patch([0.375, 0.6875]));
    }

    #[test]
    fn only_entering_rows_and_columns_are_generated() {
        let mut bed = BedWorld::new(7);
        bed.prepare([0.0; 2], 7);
        assert_eq!(bed.generated, RAW_SIZE * RAW_SIZE);
        let before = bed.generated;
        bed.prepare([0.002; 2], 7);
        assert_eq!(bed.generated, before);
        bed.prepare([0.25, -0.375], 7);
        assert_eq!(
            bed.generated - before,
            RAW_SIZE * RAW_SIZE - (RAW_SIZE - 16) * (RAW_SIZE - 24)
        );
    }

    #[test]
    fn streaming_is_continuous_at_cache_and_texel_boundaries() {
        let mut bed = BedWorld::new(91);
        bed.prepare([0.0; 2], 91);
        let a = bed.sample_patch([0.5 - 1e-5, 0.37]);
        let b = bed.sample_patch([0.5 + 1e-5, 0.37]);
        assert!(a.into_iter().zip(b).all(|(x, y)| (x - y).abs() < 0.002));
        let center = bed.sample_patch([0.5, 0.37]);
        bed.prepare([1.0 / TEXELS_PER_METRE, 0.0], 91);
        let shifted = bed.sample_patch([0.5 - 0.5 / TEXELS_PER_METRE as f32, 0.37]);
        assert_eq!(center, shifted);
    }

    #[test]
    fn memory_and_generation_stay_bounded_after_a_long_jump() {
        let mut bed = BedWorld::new(1);
        bed.prepare([0.0; 2], 1);
        let count = bed.resident_texels();
        let generated = bed.generated;
        bed.prepare([1e9 + 0.125, -1e9 - 0.5], 1);
        assert_eq!(bed.resident_texels(), count);
        assert_eq!(bed.generated - generated, RAW_SIZE * RAW_SIZE);
        assert_ne!(bed.sample_patch([0.2, 0.3]), bed.sample_patch([0.8, 0.9]));
    }
}
