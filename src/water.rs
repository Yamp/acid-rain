use crate::waves::calculate_wave_level;
use ndarray::Array2;
use std::time::{Duration, Instant};
use crate::settings::SETTINGS;

#[derive(Debug)]
pub struct WaveCenter {
    pub x: f32,
    pub y: f32,
    pub strength: f32,
    pub time: Instant,
}

impl WaveCenter {
    pub fn new(x: f32, y: f32, strength: f32, time: Instant) -> Self {
        Self { x, y, strength, time }
    }
}

#[derive(Debug)]
pub struct Water {
    pub levels: Array2<f32>,
    pub time: Instant,
    pub wave_centers: Vec<WaveCenter>,
}

impl Water {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            levels: Array2::zeros((width as usize, height as usize)),
            time: Instant::now(),
            wave_centers: Vec::new(),
        }
    }

    /// Resize the grid, keeping existing wave centers alive.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.levels = Array2::zeros((width as usize, height as usize));
    }

    #[inline(always)]
    pub fn height(&self) -> u16 {
        self.levels.shape()[1] as u16
    }

    #[inline(always)]
    pub fn width(&self) -> u16 {
        self.levels.shape()[0] as u16
    }

    fn remove_expired_wave_centers(&mut self, instant: Instant) {
        self.wave_centers
            .retain(|wc| instant - wc.time < SETTINGS.attenuation_time);
    }

    pub fn update(&mut self, delay: Duration) {
        self.time += delay;
        self.remove_expired_wave_centers(self.time);

        let w = self.width() as usize;
        let h = self.height() as usize;
        let inv_w = 1.0 / w as f32;
        let inv_h = 1.0 / h as f32;
        let att_secs = SETTINGS.attenuation_time.as_secs_f32();

        // Use raw slice: avoids ndarray indexing overhead in hot loop
        let levels = self.levels.as_slice_mut().unwrap();
        levels.fill(0.0);

        for wc in &self.wave_centers {
            let delta = (self.time - wc.time).as_secs_f32();
            let time_left = att_secs - delta;
            if time_left <= 0.0 {
                continue;
            }
            // Skip nearly-invisible wave centers
            let time_att = (time_left / att_secs) * (time_left / att_secs);
            if wc.strength * time_att < 0.001 {
                continue;
            }

            let wavefront = delta * SETTINGS.wave_speed;
            let wavefront2 = wavefront * wavefront;

            // Bounding box in pixel coords — only iterate pixels within wavefront
            let x_min = ((wc.x - wavefront) * w as f32).floor().max(0.0) as usize;
            let x_max = (((wc.x + wavefront) * w as f32).ceil() + 1.0).min(w as f32) as usize;
            let y_min = ((wc.y - wavefront) * h as f32).floor().max(0.0) as usize;
            let y_max = (((wc.y + wavefront) * h as f32).ceil() + 1.0).min(h as f32) as usize;

            for x in x_min..x_max {
                let nx = x as f32 * inv_w;
                let dx = nx - wc.x;
                let dx2 = dx * dx;
                let row_off = x * h;
                for y in y_min..y_max {
                    let ny = y as f32 * inv_h;
                    let dy = ny - wc.y;
                    let dist2 = dx2 + dy * dy;
                    if dist2 <= wavefront2 {
                        let dist = dist2.sqrt();
                        levels[row_off + y] += calculate_wave_level(
                            wc.strength, delta, dist, att_secs,
                        );
                    }
                }
            }
        }

        // Soft compression: linear for small amplitudes, saturates for large overlaps
        for level in levels.iter_mut() {
            *level = *level / (1.0 + level.abs() * 1.5);
        }
    }

    pub fn add_random_drop(&mut self) {
        let width = self.width();
        let height = self.height();

        let x = rand::random::<u16>() % width;
        let y = rand::random::<u16>() % height;

        self.wave_centers.push(WaveCenter::new(
            x as f32 / width as f32,
            y as f32 / height as f32,
            rand::random::<f32>() * 0.15,
            self.time,
        ));
    }
}
