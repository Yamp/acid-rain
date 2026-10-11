use std::sync::{
    Arc,
    Mutex,
};

use super::{
    errors::Diagnostics,
    resources::UniformArena,
};

#[allow(clippy::too_many_arguments)]
pub fn checkpoint(
    enabled: bool,
    label: &str,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniforms: &UniformArena,
    errors: &Arc<Mutex<Diagnostics>>,
) -> anyhow::Result<()> {
    if !enabled {
        return Ok(());
    }
    use anyhow::Context;
    let start = std::time::Instant::now();
    eprintln!("GPU CHECKPOINT BEGIN {label}");
    uniforms.upload(queue);
    let next = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Serialized GPU diagnosis: next production pass"),
    });
    queue.submit([std::mem::replace(encoder, next).finish()]);
    let polled = device.poll(wgpu::PollType::wait_indefinitely());
    super::errors::check(device, errors)
        .with_context(|| format!("GPU checkpoint failed: {label}"))?;
    polled.with_context(|| format!("GPU checkpoint wait failed: {label}"))?;
    eprintln!(
        "GPU CHECKPOINT PASS {label} ({:.3} ms including CPU submission/wait)",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use super::*;

    #[test]
    fn checkpoint_submits_previous_commands_before_returning() {
        if std::env::var("ACID_RAIN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: std::env::var("ACID_RAIN_SOFTWARE").as_deref() == Ok("1"),
            ..Default::default()
        }))
        .expect("GPU diagnostics fixture requested but no adapter exists");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("checkpoint fixture input"),
            contents: bytemuck::bytes_of(&37u32),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("checkpoint fixture output"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("checkpoint fixture readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&input, 0, &output, 0, 4);
        checkpoint(
            true,
            "fixture transfer",
            &mut encoder,
            &device,
            &queue,
            &UniformArena::new(&device),
            &Arc::new(Mutex::new(Diagnostics::default())),
        )
        .unwrap();
        let mut read = device.create_command_encoder(&Default::default());
        read.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
        queue.submit([read.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range();
        assert_eq!(
            u32::from_ne_bytes(data[..4].try_into().unwrap()),
            37,
            "the diagnostic checkpoint must submit the same production commands"
        );
    }
}
