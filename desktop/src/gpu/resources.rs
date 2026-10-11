use std::num::NonZeroU64;

use acid_rain_native_core::{
    ARDrop,
    ARUniforms,
    OwnedSnapshot,
};
use anyhow::{
    Result,
    ensure,
};

pub const UNIFORM_CAPACITY: u64 = 65_536;

#[derive(Clone)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}
impl Texture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        depth: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    pub fn field(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
    ) -> Self {
        Self::new(
            device,
            label,
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        values: &[[f32; 4]],
    ) {
        queue.write_texture(
            self.texture.as_image_copy(),
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.texture.width() * 16),
                rows_per_image: Some(self.texture.height()),
            },
            self.texture.size(),
        );
    }
}

/// Every dispatch owns a distinct aligned slice. Updating one buffer offset for
/// multiple encoded passes would silently give all passes the final constants.
pub struct UniformArena {
    pub buffer: wgpu::Buffer,
    bytes: Vec<u8>,
    alignment: usize,
}
impl UniformArena {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Per-frame immutable pass constants"),
                size: UNIFORM_CAPACITY,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bytes: Vec::with_capacity(UNIFORM_CAPACITY as usize),
            alignment: device.limits().min_uniform_buffer_offset_alignment as usize,
        }
    }

    pub fn reset(&mut self) { self.bytes.clear(); }

    pub fn push(
        &mut self,
        bytes: &[u8],
    ) -> Result<UniformSlice> {
        let offset = self.bytes.len().div_ceil(self.alignment) * self.alignment;
        ensure!(
            offset + bytes.len() <= UNIFORM_CAPACITY as usize,
            "Per-frame uniform arena exhausted"
        );
        self.bytes.resize(offset, 0);
        self.bytes.extend_from_slice(bytes);
        Ok(UniformSlice {
            offset: offset as u64,
            size: NonZeroU64::new(bytes.len() as u64).unwrap(),
        })
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(&self.buffer, 0, &self.bytes);
    }

    pub fn bind(
        &self,
        slice: UniformSlice,
    ) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: slice.offset,
            size: Some(slice.size),
        })
    }
}
#[derive(Clone, Copy)]
pub struct UniformSlice {
    offset: u64,
    size: NonZeroU64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawDrop {
    position: [f32; 4],
    pigment: [f32; 4],
    dynamics: [f32; 4],
    identity: [u32; 4],
}
impl From<&ARDrop> for RawDrop {
    fn from(value: &ARDrop) -> Self {
        Self {
            position: value.position.0,
            pigment: value.pigment.0,
            dynamics: value.dynamics.0,
            identity: value.identity,
        }
    }
}
pub fn uniform_values(u: &ARUniforms) -> [[f32; 4]; 13] {
    [
        u.viewport.0,
        u.time.0,
        u.camera.0,
        u.optics.0,
        u.gain.0,
        u.light.0,
        u.world.0,
        u.post.0,
        u.eye.0,
        u.forward.0,
        u.right.0,
        u.up.0,
        u.illumination.0,
    ]
}

pub struct FrameSlot {
    pub water: Texture,
    pub dye: Texture,
    pub flow: Texture,
    pub bed: Texture,
    pub sky: Texture,
    pub light: Texture,
    pub airborne: wgpu::Buffer,
    pub sources: wgpu::Buffer,
    pub wakes: wgpu::Buffer,
    pub normalization: wgpu::Buffer,
    pub uniforms: UniformArena,
    bed_revision: Option<u64>,
}
impl FrameSlot {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            water: Texture::field(device, "Water surface", 192, 128),
            dye: Texture::field(device, "Dye and depth", 192, 128),
            flow: Texture::field(device, "Driving velocity", 192, 128),
            bed: Texture::field(device, "Streamed bottom", 256, 256),
            sky: Texture::field(device, "Environment", 128, 64),
            light: Texture::field(device, "Incident bottom light", 64, 64),
            airborne: buffer("Airborne drops", 512 * 64),
            sources: buffer("New sources", 512 * 64),
            wakes: buffer("Impact wakes", 24 * 64),
            normalization: buffer("Source mass normalization", 512 * 16),
            uniforms: UniformArena::new(device),
            bed_revision: None,
        }
    }

    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        frame: &OwnedSnapshot,
    ) -> Result<()> {
        for (texture, values) in [
            (&self.water, &frame.water),
            (&self.dye, &frame.dye),
            (&self.flow, &frame.flow),
            (&self.sky, &frame.sky),
            (&self.light, &frame.bed_light),
        ] {
            ensure!(
                values.len() == (texture.texture.width() * texture.texture.height()) as usize,
                "Unexpected simulation texture size"
            );
            texture.upload(queue, values);
        }
        if self.bed_revision != Some(frame.bed_revision) {
            ensure!(
                frame.bed.len() == 256 * 256,
                "Unexpected streamed bottom size"
            );
            self.bed.upload(queue, &frame.bed);
            self.bed_revision = Some(frame.bed_revision);
        }
        self.uniforms.reset();
        Ok(())
    }
}
pub fn upload_drops(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    values: &[ARDrop],
) {
    if values.is_empty() {
        return;
    }
    let packed: Vec<RawDrop> = values.iter().map(RawDrop::from).collect();
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(&packed));
}

pub fn initialized_exposure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Texture {
    let texture = Texture::new(
        device,
        "Pupil exposure",
        1,
        1,
        1,
        wgpu::TextureDimension::D2,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
    );
    queue.write_texture(
        texture.texture.as_image_copy(),
        bytemuck::bytes_of(&1.0f32),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        texture.texture.size(),
    );
    texture
}

/// Offline/test-only: queue-ordered readback with row padding removed.
pub fn read_texture_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>> {
    let size = texture.size();
    let row = size.width * bytes_per_pixel;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let length = u64::from(padded) * u64::from(size.height) * u64::from(size.depth_or_array_layers);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Offline texture readback"),
        size: length,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Offline readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let mapped = buffer.slice(..).get_mapped_range();
    let mut output = Vec::with_capacity((row * size.height * size.depth_or_array_layers) as usize);
    for line in mapped.chunks(padded as usize) {
        output.extend_from_slice(&line[..row as usize]);
    }
    drop(mapped);
    buffer.unmap();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_layouts_match_shared_core_without_unsafe_casts() {
        assert_eq!(std::mem::size_of::<RawDrop>(), 64);
        assert_eq!(
            std::mem::size_of_val(&uniform_values(&ARUniforms::default())),
            208
        );
    }
    #[test]
    fn readback_padding_does_not_change_row_content() {
        for width in [1u32, 63, 64, 65, 1920] {
            let row = width * 4;
            let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            assert!(padded >= row);
            assert_eq!(padded % 256, 0);
        }
    }
}
