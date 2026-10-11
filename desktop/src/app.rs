use std::{
    sync::Arc,
    time::{
        Duration,
        Instant,
    },
};

use acid_rain_desktop::{
    gpu::{
        GpuConfig,
        GpuRenderer,
        RenderInput,
    },
    simulation::{
        Control,
        Worker,
    },
};
use egui_wgpu::{
    Renderer,
    RendererOptions,
    ScreenDescriptor,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{
        ElementState,
        MouseButton,
        WindowEvent,
    },
    event_loop::{
        ActiveEventLoop,
        ControlFlow,
        EventLoop,
        EventLoopProxy,
    },
    keyboard::{
        ModifiersState,
        PhysicalKey,
    },
    window::{
        Fullscreen,
        Window,
        WindowId,
    },
};

use crate::{
    cli::Options,
    ui::{
        self,
        Action,
        ViewState,
    },
};

#[derive(Debug)]
pub enum UserEvent {
    FrameReady,
}
pub fn run(options: Options) -> anyhow::Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let worker = Worker::from_simulation(options.simulation()?)?;
    worker.set_held(true);
    let mut app = App {
        view: ViewState::from_options(&options),
        options,
        worker,
        proxy,
        window: None,
        gpu: None,
        egui: None,
        epoch: Instant::now(),
        cadence: RenderCadence::default(),
        modifiers: ModifiersState::empty(),
        cursor: [0., 0.],
        occluded: false,
        minimized: false,
        suspended: true,
        held: true,
        fps: 0.,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.error {
        anyhow::bail!(error);
    }
    Ok(())
}
/// Display time advances only on submitted frames. Attempts retain a separate
/// pacing anchor so a saturated GPU cannot create either slow time or a busy loop.
#[derive(Default)]
struct RenderCadence {
    attempt: Option<Instant>,
    submitted: Option<Instant>,
    pending: bool,
}
impl RenderCadence {
    fn begin(
        &mut self,
        now: Instant,
    ) -> f64 {
        self.attempt = Some(now);
        self.pending = true;
        self.submitted
            .map_or(0., |last| now.saturating_duration_since(last).as_secs_f64())
    }

    fn submitted(
        &mut self,
        now: Instant,
    ) {
        self.submitted = Some(now);
        self.pending = false;
    }

    fn reset(&mut self) { *self = Self::default(); }

    fn deadline(&self) -> Instant {
        self.attempt.unwrap_or_else(Instant::now) + Duration::from_secs_f64(1. / 120.)
    }
}

struct Gui {
    ctx: egui::Context,
    input: egui_winit::State,
    renderer: Renderer,
}
struct App {
    options: Options,
    view: ViewState,
    worker: Worker,
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuRenderer>,
    egui: Option<Gui>,
    epoch: Instant,
    cadence: RenderCadence,
    modifiers: ModifiersState,
    cursor: [f64; 2],
    occluded: bool,
    minimized: bool,
    suspended: bool,
    held: bool,
    fps: f32,
    error: Option<String>,
}
impl App {
    fn visible(&self) -> bool { !self.occluded && !self.minimized && !self.suspended }

    fn sync_hold(&mut self) {
        if let Some(window) = &self.window {
            let size = window.inner_size();
            self.minimized =
                size.width == 0 || size.height == 0 || window.is_minimized().unwrap_or(false);
        }
        let held = self.view.help || !self.visible() || self.worker.latest().snapshot.paused;
        if held != self.held {
            self.held = held;
            self.cadence.reset();
        }
        // Core pause is independent of the external hold; it survives closing help.
        self.worker.set_held(self.view.help || !self.visible());
    }

    fn request_redraw(&self) {
        if self.visible()
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
    }

    fn control(
        &mut self,
        control: Control,
    ) {
        if let Err(error) = self.worker.command(control) {
            self.error = Some(error.to_string());
        }
    }

    fn action(
        &mut self,
        action: Action,
        event_loop: &ActiveEventLoop,
    ) {
        match action {
            Action::Quit if self.view.help => self.view.help = false,
            Action::Quit => {
                event_loop.exit();
                return;
            },
            Action::Help => self.view.help = !self.view.help,
            Action::Pause => self.control(Control::Set { code: 1, value: 0. }),
            Action::Fullscreen => {
                if let Some(w) = &self.window {
                    w.set_fullscreen(if w.fullscreen().is_some() {
                        None
                    } else {
                        Some(Fullscreen::Borderless(w.current_monitor()))
                    });
                }
            },
            Action::Set(code, value) => self.control(Control::Set { code, value }),
            Action::Adjust(code, delta) => self.control(Control::Adjust { code, delta }),
            Action::Quality(quality) => self.view.quality = quality,
            Action::QualityCycle => self.view.quality = self.view.quality.next(),
            Action::SceneCycle => self.control(Control::Adjust {
                code: 10,
                delta: 1.,
            }),
            Action::SceneReset => self.control(Control::Set {
                code: 10,
                value: 0.,
            }),
            Action::Drain => self.control(Control::ToggleMode(1)),
            Action::Rapids => self.control(Control::ToggleMode(2)),
            Action::ToggleInterpolation => self.view.interpolation = !self.view.interpolation,
            Action::ToggleStudy => self.view.cloud_study = !self.view.cloud_study,
            Action::ToggleDrops => self.view.show_drops = !self.view.show_drops,
            Action::Lsd(delta) => self.view.adjust_lsd(delta),
            Action::ResetLsd => self.view.lsd = 1.,
            Action::ResetRain => {
                self.control(Control::Set {
                    code: 3,
                    value: 32.,
                });
                self.control(Control::Set {
                    code: 4,
                    value: 1.5,
                });
            },
            Action::Drop => self.drop_at([0.5, 0.5]),
        }
        self.sync_hold();
        self.request_redraw();
    }

    fn drop_at(
        &mut self,
        uv: [f32; 2],
    ) {
        let snapshot = self.worker.latest().snapshot.clone();
        if let Some(world) = self.gpu.as_ref().and_then(|gpu| gpu.surface_point(uv)) {
            self.control(Control::Drop {
                world,
                size: snapshot.drop_size,
            });
        }
    }

    fn draw(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) {
        if !self.visible() {
            return;
        }
        if let Some(error) = self.worker.error() {
            self.error = Some(error);
            event_loop.exit();
            return;
        }
        self.sync_hold();
        let now = Instant::now();
        let delta = self.cadence.begin(now);
        let frame = self.worker.latest();
        let window = self.window.as_ref().unwrap();
        let gui = self.egui.as_mut().unwrap();
        let raw = gui.input.take_egui_input(window);
        let mut actions = Vec::new();
        let output = gui.ctx.run(raw, |ctx| {
            actions = ui::draw(
                ctx,
                &self.view,
                &frame.snapshot,
                self.fps,
                self.error.as_deref(),
            );
        });
        gui.input
            .handle_platform_output(window, output.platform_output);
        let jobs = gui.ctx.tessellate(output.shapes, output.pixels_per_point);
        let input = RenderInput {
            delta,
            now: now.duration_since(self.epoch).as_secs_f64(),
            sample_age: now
                .saturating_duration_since(frame.simulation_started)
                .as_secs_f64(),
            predict_camera: true,
            held: self.held,
            quality: self.view.quality,
            interpolation: self.view.interpolation,
            cloud_study: self.view.cloud_study,
            show_falling_drops: self.view.show_drops,
            subjective_intensity: self.view.lsd,
            backing_scale: window.scale_factor() as f32,
        };
        let gpu = self.gpu.as_mut().unwrap();
        for (id, delta) in &output.textures_delta.set {
            gui.renderer
                .update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        let result = gpu.render(
            &frame.snapshot,
            &input,
            |device, queue, encoder, target, _format, extent| {
                let descriptor = ScreenDescriptor {
                    size_in_pixels: extent,
                    pixels_per_point: output.pixels_per_point,
                };
                let additional =
                    gui.renderer
                        .update_buffers(device, queue, encoder, &jobs, &descriptor);
                // Our UI uses meshes/text only, so there are no external callback command buffers.
                debug_assert!(additional.is_empty());
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Interface"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                gui.renderer
                    .render(&mut pass.forget_lifetime(), &jobs, &descriptor);
            },
        );
        for id in &output.textures_delta.free {
            gui.renderer.free_texture(id);
        }
        match result {
            Ok(outcome) if outcome.submitted => {
                self.cadence.submitted(now);
                if delta > 0. && delta < 0.25 && !self.held {
                    let instant_fps = (1. / delta) as f32;
                    self.fps = if self.fps == 0. {
                        instant_fps
                    } else {
                        self.fps * 0.94 + instant_fps * 0.06
                    };
                }
            },
            Ok(_) => {},
            Err(error) => {
                self.error = Some(format!("Rendering failed: {error:#}"));
                event_loop.exit();
                return;
            },
        }
        for action in actions {
            self.action(action, event_loop);
        }
    }
}
impl ApplicationHandler<UserEvent> for App {
    fn resumed(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) {
        self.suspended = false;
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Acid Rain")
                .with_inner_size(LogicalSize::new(self.options.size[0], self.options.size[1]))
                .with_min_inner_size(LogicalSize::new(320., 240.));
            let result = (|| -> anyhow::Result<()> {
                let window = Arc::new(event_loop.create_window(attrs)?);
                if self.options.fullscreen {
                    window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
                }
                let size = window.inner_size();
                let gpu = pollster::block_on(GpuRenderer::new(Some(window.clone()), GpuConfig {
                    width: size.width.max(1),
                    height: size.height.max(1),
                    backends: self.options.backend.backends(),
                    force_fallback_adapter: self.options.software,
                    timestamp_queries: !self.options.no_gpu_timestamps,
                }))?;
                let ctx = egui::Context::default();
                ui::configure(&ctx);
                let input = egui_winit::State::new(
                    ctx.clone(),
                    egui::ViewportId::ROOT,
                    window.as_ref(),
                    Some(window.scale_factor() as f32),
                    window.theme(),
                    Some(gpu.device.limits().max_texture_dimension_2d as usize),
                );
                let renderer = Renderer::new(&gpu.device, gpu.format, RendererOptions::default());
                self.egui = Some(Gui {
                    ctx,
                    input,
                    renderer,
                });
                self.gpu = Some(gpu);
                self.window = Some(window);
                Ok(())
            })();
            if let Err(error) = result {
                self.error = Some(format!("Cannot open Acid Rain: {error:#}"));
                event_loop.exit();
                return;
            }
            let proxy = self.proxy.clone();
            self.worker.set_waker(Arc::new(move || {
                let _ = proxy.send_event(UserEvent::FrameReady);
            }));
        }
        self.sync_hold();
        self.request_redraw();
    }

    fn about_to_wait(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) {
        self.sync_hold();
        if self.visible() && (!self.held || self.cadence.pending) {
            // Bound retries when the GPU already has three frames in flight.
            // WaitUntil sleeps the event loop; there is no polling spin.
            let deadline = self.cadence.deadline();
            if Instant::now() >= deadline {
                self.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn suspended(
        &mut self,
        _event_loop: &ActiveEventLoop,
    ) {
        self.suspended = true;
        self.sync_hold();
    }

    fn user_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _event: UserEvent,
    ) {
        self.sync_hold();
        self.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        id: WindowId,
        event: WindowEvent,
    ) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        let response = self
            .egui
            .as_mut()
            .unwrap()
            .input
            .on_window_event(self.window.as_ref().unwrap(), &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.draw(event_loop),
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // Editing a focused slider must not also change the water.
                // Global escape/help/pause remain available in the guide.
                if let PhysicalKey::Code(key) = event.physical_key
                    && let Some(action) = ui::key_action(key, self.modifiers, event.repeat)
                    && (!response.consumed
                        || matches!(
                            action,
                            Action::Quit | Action::Help | Action::Pause | Action::Fullscreen
                        ))
                {
                    self.action(action, event_loop);
                }
            },
            WindowEvent::CursorMoved { position, .. } => self.cursor = [position.x, position.y],
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !response.consumed && !self.view.help => {
                let size = self.window.as_ref().unwrap().inner_size();
                if size.width > 0 && size.height > 0 {
                    self.drop_at([
                        (self.cursor[0] / f64::from(size.width)) as f32,
                        (self.cursor[1] / f64::from(size.height)) as f32,
                    ]);
                }
            },
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if !self.minimized {
                    self.gpu.as_mut().unwrap().resize(size.width, size.height);
                }
                self.sync_hold();
                self.request_redraw();
            },
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                self.sync_hold();
                self.request_redraw();
            },
            WindowEvent::Focused(_) => {
                self.sync_hold();
                self.request_redraw();
            },
            WindowEvent::ScaleFactorChanged { .. } => {
                self.cadence.reset();
                self.request_redraw();
            },
            _ => {},
        }
        if response.repaint {
            self.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn skipped_draws_preserve_elapsed_presentation_time_and_sleep_between_retries() {
        let origin = Instant::now();
        let mut cadence = RenderCadence::default();
        assert_eq!(cadence.begin(origin), 0.);
        assert!(cadence.pending);
        cadence.submitted(origin);
        assert!(!cadence.pending);
        for millis in [8, 16, 24] {
            let attempted = origin + Duration::from_millis(millis);
            assert!((cadence.begin(attempted) - millis as f64 / 1000.).abs() < 1e-9);
            assert!(cadence.deadline() > attempted);
        }
        let next = origin + Duration::from_millis(33);
        assert!((cadence.begin(next) - 0.033).abs() < 1e-9);
        assert!(cadence.pending);
        cadence.submitted(next);
        assert!(!cadence.pending);
        assert!((cadence.begin(next + Duration::from_millis(8)) - 0.008).abs() < 1e-9);
        cadence.reset();
        assert_eq!(cadence.begin(origin + Duration::from_secs(90)), 0.);
    }
}
