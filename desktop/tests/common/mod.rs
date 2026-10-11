//! Shared opt-in GPU fixtures. Pure Naga validation remains unconditional.
//! ACID_RAIN_GPU_TESTS=1 requires a working adapter; it never silently skips.
#![allow(dead_code)]

use std::{
    sync::mpsc,
    time::{
        Duration,
        Instant,
    },
};

use acid_rain_desktop::shaders::{
    self,
    ShaderKind,
};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub common_layout: wgpu::BindGroupLayout,
    pub common_samplers: wgpu::BindGroup,
    pub adapter_info: wgpu::AdapterInfo,
}

pub fn gpu() -> Option<Gpu> { gpu_with_optional_features(wgpu::Features::empty()) }

/// The no-feature baseline never skips an opted-in adapter failure. Optional
/// capability probes may skip only when that adapter lacks the requested feature.
pub fn gpu_with_optional_features(features: wgpu::Features) -> Option<Gpu> {
    if std::env::var("ACID_RAIN_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("GPU fixture skipped; set ACID_RAIN_GPU_TESTS=1 to execute it");
        return None;
    }
    let backends = match std::env::var("WGPU_BACKEND").as_deref() {
        Ok("metal") => wgpu::Backends::METAL,
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("dx12") => wgpu::Backends::DX12,
        Err(_) => wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        Ok(value) => panic!("unsupported WGPU_BACKEND={value}; use metal, vulkan or dx12"),
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: std::env::var("ACID_RAIN_SOFTWARE").as_deref() == Ok("1"),
        ..Default::default()
    }))
    .expect("GPU tests were requested but no matching adapter is available");
    let adapter_info = adapter.get_info();
    eprintln!("GPU fixture adapter: {adapter_info:?}");
    if !adapter.features().contains(features) {
        eprintln!(
            "Optional GPU path skipped: adapter lacks {:?}",
            features - adapter.features()
        );
        return None;
    }
    // The manual sampling path deliberately requests no FLOAT32_FILTERABLE or
    // storage-read-write texture extension; FP32 loads and write-only stores suffice.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("desktop correctness fixtures"),
        required_features: features,
        ..Default::default()
    }))
    .expect("could not create the requested GPU test device");
    let common_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shared test samplers"),
        entries: &std::array::from_fn::<_, 3, _>(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::all(),
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }),
    });
    let make_sampler = |label, u, v| {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: u,
            address_mode_v: v,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    };
    let clamp = make_sampler(
        "test clamp",
        wgpu::AddressMode::ClampToEdge,
        wgpu::AddressMode::ClampToEdge,
    );
    let sky = make_sampler(
        "test sky",
        wgpu::AddressMode::Repeat,
        wgpu::AddressMode::ClampToEdge,
    );
    let mirror = make_sampler(
        "test mirror",
        wgpu::AddressMode::MirrorRepeat,
        wgpu::AddressMode::MirrorRepeat,
    );
    let common_samplers = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shared test sampler group"),
        layout: &common_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&clamp),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sky),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&mirror),
            },
        ],
    });
    Some(Gpu {
        device,
        queue,
        common_layout,
        common_samplers,
        adapter_info,
    })
}

impl Gpu {
    pub fn shader(
        &self,
        kind: ShaderKind,
    ) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("production WGSL correctness fixture"),
                source: wgpu::ShaderSource::Wgsl(shaders::source(kind, false).into()),
            })
    }

    pub fn texture_2d(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        self.texture(width, height, 1, wgpu::TextureDimension::D2, format, label)
    }

    pub fn texture_3d(
        &self,
        width: u32,
        height: u32,
        depth: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        self.texture(
            width,
            height,
            depth,
            wgpu::TextureDimension::D3,
            format,
            label,
        )
    }

    fn texture(
        &self,
        width: u32,
        height: u32,
        depth: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub fn upload(
        &self,
        texture: &wgpu::Texture,
        bytes: &[u8],
        bytes_per_pixel: u32,
    ) {
        let size = texture.size();
        assert_eq!(
            bytes.len(),
            (size.width * size.height * size.depth_or_array_layers * bytes_per_pixel) as usize
        );
        self.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_pixel),
                rows_per_image: Some(size.height),
            },
            size,
        );
    }

    pub fn upload_rgba32f(
        &self,
        texture: &wgpu::Texture,
        values: &[[f32; 4]],
    ) {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba32Float);
        self.upload(texture, bytemuck::cast_slice(values), 16);
    }

    /// Read any uncompressed format, returning tightly packed rows and layers.
    pub fn read_texture(
        &self,
        texture: &wgpu::Texture,
        bytes_per_pixel: u32,
    ) -> Vec<u8> {
        let size = texture.size();
        let row = size.width * bytes_per_pixel;
        let padded =
            row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixture readback"),
            size: u64::from(padded)
                * u64::from(size.height)
                * u64::from(size.depth_or_array_layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
        self.queue.submit([encoder.finish()]);
        let mapped = self.mapped_bytes(&buffer);
        let mut bytes =
            Vec::with_capacity((row * size.height * size.depth_or_array_layers) as usize);
        for padded_row in mapped.chunks_exact(padded as usize) {
            bytes.extend_from_slice(&padded_row[..row as usize]);
        }
        bytes
    }

    /// Source buffer must have COPY_SRC usage; returns its complete contents.
    pub fn read_buffer(
        &self,
        source: &wgpu::Buffer,
    ) -> Vec<u8> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixture buffer readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
        self.queue.submit([encoder.finish()]);
        self.mapped_bytes(&buffer)
    }

    fn mapped_bytes(
        &self,
        buffer: &wgpu::Buffer,
    ) -> Vec<u8> {
        let (sender, receiver) = mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            self.device
                .poll(wgpu::PollType::Poll)
                .expect("GPU device polling failed");
            match receiver.recv_timeout(Duration::from_millis(2)) {
                Ok(result) => {
                    result.expect("GPU readback mapping failed");
                    break;
                },
                Err(mpsc::RecvTimeoutError::Timeout) if Instant::now() < deadline => {},
                failure => panic!("GPU readback timed out or disconnected: {failure:?}"),
            }
        }
        let mapped = buffer.slice(..).get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        buffer.unmap();
        bytes
    }

    pub fn read_rgba32f(
        &self,
        texture: &wgpu::Texture,
    ) -> Vec<[f32; 4]> {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba32Float);
        self.read_texture(texture, 16)
            .as_chunks::<16>()
            .0
            .iter()
            .map(|pixel| {
                std::array::from_fn(|channel| {
                    f32::from_ne_bytes(pixel[channel * 4..channel * 4 + 4].try_into().unwrap())
                })
            })
            .collect()
    }
}
