use acid_rain_desktop::quality::RenderQuality;
use acid_rain_native_core::OwnedSnapshot;
use egui::{
    Color32,
    RichText,
};
use winit::keyboard::{
    KeyCode,
    ModifiersState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Quit,
    Help,
    Pause,
    Fullscreen,
    QualityCycle,
    Quality(RenderQuality),
    SceneCycle,
    SceneReset,
    Drain,
    Rapids,
    ToggleInterpolation,
    ToggleStudy,
    ToggleDrops,
    Drop,
    Set(u32, f32),
    Adjust(u32, f32),
    Lsd(f32),
    ResetLsd,
    ResetRain,
}
#[derive(Clone, Debug)]
pub struct ViewState {
    pub help: bool,
    pub quality: RenderQuality,
    pub interpolation: bool,
    pub cloud_study: bool,
    pub show_drops: bool,
    pub lsd: f32,
}
impl ViewState {
    pub fn from_options(o: &crate::cli::Options) -> Self {
        Self {
            help: false,
            quality: o.quality.render(),
            interpolation: !o.no_interpolation,
            cloud_study: o.cloud_study || o.drop_demo,
            show_drops: o.show_drops,
            lsd: o.lsd,
        }
    }

    pub fn adjust_lsd(
        &mut self,
        delta: f32,
    ) {
        self.lsd = (self.lsd + delta).clamp(0., 2.);
    }
}
/// Physical keys work unchanged when the operating system uses a Cyrillic layout.
/// Continuous controls repeat; toggles and reset shortcuts are press-only.
pub fn key_action(
    key: KeyCode,
    mods: ModifiersState,
    repeat: bool,
) -> Option<Action> {
    use Action::*;
    use KeyCode::*;
    if key == F11 || (key == Enter && mods.alt_key()) || (key == KeyF && mods.super_key()) {
        return (!repeat).then_some(Fullscreen);
    }
    if mods.control_key() || mods.super_key() || mods.alt_key() {
        return None;
    }
    let shift = mods.shift_key();
    let (action, continuous) = match key {
        KeyQ | Escape => (Quit, false),
        KeyH => (Action::Help, false),
        Slash if shift => (Action::Help, false),
        Space => (Action::Pause, false),
        KeyN if shift => (ResetLsd, false),
        KeyN => (Lsd(-0.1), true),
        KeyM => (Lsd(0.1), true),
        KeyJ if shift => (Set(7, 2.0), false),
        KeyJ => (Adjust(7, -0.25), true),
        KeyL => (Adjust(7, 0.25), true),
        KeyS => (Set(2, 2.4), false),
        BracketLeft | ArrowLeft => (Adjust(2, -0.1), true),
        BracketRight | ArrowRight => (Adjust(2, 0.1), true),
        ArrowDown => (Adjust(3, -2.0), true),
        ArrowUp => (Adjust(3, 2.0), true),
        Comma => (Adjust(4, -0.25), true),
        Period => (Adjust(4, 0.25), true),
        KeyR if shift => (ResetRain, false),
        KeyR => (Set(14, 0.), false),
        KeyB if shift => (Set(5, 1.0), false),
        KeyV => (Adjust(5, -0.25), true),
        KeyB => (Adjust(5, 0.25), true),
        Minus | NumpadSubtract => (Adjust(6, -0.1), true),
        Equal | NumpadAdd => (Adjust(6, 0.1), true),
        KeyE => (Set(13, 0.), false),
        KeyC => (Set(11, 0.), false),
        KeyW => (Set(12, 0.), false),
        KeyX => (Set(16, 0.), false),
        KeyD => (Drain, false),
        KeyF => (Rapids, false),
        KeyG | Tab => (SceneCycle, false),
        KeyT => (SceneReset, false),
        KeyI => (ToggleInterpolation, false),
        KeyP => (ToggleStudy, false),
        KeyA => (ToggleDrops, false),
        KeyO => (Drop, false),
        KeyK => (QualityCycle, false),
        Digit0 => (Quality(RenderQuality::Auto), false),
        Digit1 => (Quality(RenderQuality::High), false),
        Digit2 => (Quality(RenderQuality::Balanced), false),
        Digit3 => (Quality(RenderQuality::Low), false),
        Digit4 => (Quality(RenderQuality::Native), false),
        _ => return None,
    };
    (!repeat || continuous).then_some(action)
}
pub fn scene_name(scene: u32) -> &'static str {
    [
        "Journey", "Acid", "Aurora", "Storm", "Abyss", "Lava", "Prism",
    ]
    .get(scene as usize)
    .copied()
    .unwrap_or("Journey")
}
pub fn mode_name(mode: u32) -> &'static str {
    ["Classic", "Drain", "Rapids"]
        .get(mode as usize)
        .copied()
        .unwrap_or("Classic")
}

pub fn configure(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.override_text_color = Some(Color32::from_rgb(228, 222, 240));
    style.visuals.window_fill = Color32::from_rgba_unmultiplied(13, 15, 28, 232);
    style.visuals.panel_fill = Color32::from_rgba_unmultiplied(13, 15, 28, 198);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(96, 65, 145);
    style.visuals.selection.bg_fill = Color32::from_rgb(105, 66, 153);
    style.spacing.item_spacing = egui::vec2(10., 8.);
    ctx.set_style(style);
}
pub fn draw(
    ctx: &egui::Context,
    state: &ViewState,
    frame: &OwnedSnapshot,
    fps: f32,
    error: Option<&str>,
) -> Vec<Action> {
    let mut actions = Vec::new();
    egui::Area::new("hud".into())
        .fixed_pos(egui::pos2(16., 16.))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(10, 12, 24, 184))
                .corner_radius(12)
                .inner_margin(12)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("ACID RAIN")
                                .strong()
                                .color(Color32::from_rgb(219, 199, 255)),
                        );
                        if ui
                            .small_button(if state.help {
                                "Close help"
                            } else {
                                "H · Controls"
                            })
                            .clicked()
                        {
                            actions.push(Action::Help);
                        }
                        if ui
                            .small_button(if frame.paused {
                                "▶ Resume"
                            } else {
                                "Ⅱ Pause"
                            })
                            .clicked()
                        {
                            actions.push(Action::Pause);
                        }
                    });
                    ui.label(format!(
                        "{} · {} · {:.2}× · {}",
                        scene_name(frame.scene),
                        mode_name(frame.mode),
                        frame.speed,
                        state.quality.name()
                    ));
                    ui.label(format!(
                        "Trip {:.1} · LSD {:.1} · RGB {:.2}px{}",
                        frame.trip,
                        state.lsd,
                        frame.rgb_shift,
                        if frame.effects { "" } else { " (off)" }
                    ));
                    ui.label(
                        RichText::new(format!(
                            "{fps:.0} fps · Rain {:.0}/s · Drops {:.2}×",
                            frame.rain, frame.drop_size
                        ))
                        .small()
                        .weak(),
                    );
                    if state.help || frame.paused {
                        ui.label(
                            RichText::new("Animation held").color(Color32::from_rgb(239, 210, 168)),
                        );
                    }
                    if state.cloud_study {
                        ui.label(RichText::new("Observation lighting").small());
                    }
                    if let Some(error) = error {
                        ui.colored_label(Color32::LIGHT_RED, error);
                    }
                });
        });
    if state.help {
        let mut open = true;
        egui::Window::new("Explore the water")
            .open(&mut open)
            .anchor(egui::Align2::CENTER_CENTER, [0., 0.])
            .default_width(680.)
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label(
                    "Animation waits while this panel is open. Click the water to add colour when \
                     you return.",
                );
                egui::ScrollArea::vertical()
                    .max_height((ctx.content_rect().height() - 170.).max(160.))
                    .show(ui, |ui| {
                        ui.add_space(6.);
                        egui::Grid::new("controls")
                            .num_columns(2)
                            .striped(true)
                            .spacing([26., 8.])
                            .show(ui, |ui| {
                                for (keys, description) in [
                                    ("Space", "Pause / resume"),
                                    ("H / ?", "Open / close this guide"),
                                    ("Click / O", "Add one coloured drop"),
                                    ("R / X", "Burst of drops / clear colour"),
                                    ("A", "Show / hide falling drops"),
                                    ("↑ / ↓", "Rain intensity"),
                                    (", / .", "Drop diameter"),
                                    ("Shift+R", "Restore rain and drop size"),
                                    ("[ / ] or ← / →", "Animation speed"),
                                    ("S", "Restore speed to 2.4×"),
                                    ("− / +", "Trip intensity"),
                                    ("N / M", "Subjective effects: less / more"),
                                    ("Shift+N", "Restore LSD to 1.0"),
                                    ("J / L", "RGB separation: less / more"),
                                    ("Shift+J", "Restore RGB separation to 2px"),
                                    ("E", "Toggle retinal and RGB effects"),
                                    ("V / B", "Dye fading: slower / faster"),
                                    ("Shift+B", "Restore dye fading"),
                                    ("D / F", "Drain / rapids (again for classic water)"),
                                    ("G / Tab", "Next scene"),
                                    ("T", "Return to Journey"),
                                    ("C / W", "Camera travel / water current"),
                                    ("P", "Observation lighting"),
                                    ("I", "Frame blending"),
                                    ("K", "Cycle quality"),
                                    ("0 / 1 / 2 / 3 / 4", "Auto / High / Balanced / Low / Native"),
                                    ("F11 / Alt+Enter / ⌘F", "Fullscreen"),
                                    ("Esc / Q", "Close guide, then quit"),
                                ] {
                                    ui.label(
                                        RichText::new(keys)
                                            .monospace()
                                            .color(Color32::from_rgb(207, 183, 251)),
                                    );
                                    ui.label(description);
                                    ui.end_row();
                                }
                            });
                        ui.separator();
                        ui.horizontal_wrapped(|ui| {
                            for quality in [
                                RenderQuality::Auto,
                                RenderQuality::High,
                                RenderQuality::Balanced,
                                RenderQuality::Low,
                                RenderQuality::Native,
                            ] {
                                if ui
                                    .selectable_label(state.quality == quality, quality.name())
                                    .clicked()
                                {
                                    actions.push(Action::Quality(quality));
                                }
                            }
                        });
                        let mut lsd = state.lsd;
                        if ui
                            .add(
                                egui::Slider::new(&mut lsd, 0.0..=2.0)
                                    .text("LSD")
                                    .step_by(0.1),
                            )
                            .changed()
                        {
                            actions.push(Action::Lsd(lsd - state.lsd));
                        }
                    });
            });
        if !open {
            actions.push(Action::Help);
        }
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operating_system_light_theme_cannot_override_the_dark_overlay() {
        let ctx = egui::Context::default();
        configure(&ctx);
        for theme in [egui::Theme::Light, egui::Theme::Dark, egui::Theme::Light] {
            let input = egui::RawInput {
                system_theme: Some(theme),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                assert_eq!(ctx.theme(), egui::Theme::Dark);
                assert!(ctx.style().visuals.dark_mode);
                assert_eq!(
                    ctx.style().visuals.window_fill,
                    Color32::from_rgba_unmultiplied(13, 15, 28, 232)
                );
                assert!(ctx.style().visuals.text_color().r() >= 150);
            });
        }
    }
    #[test]
    fn physical_keys_and_shift_resets() {
        assert_eq!(
            key_action(KeyCode::KeyN, ModifiersState::empty(), false),
            Some(Action::Lsd(-0.1))
        );
        assert_eq!(
            key_action(KeyCode::KeyN, ModifiersState::SHIFT, false),
            Some(Action::ResetLsd)
        );
        assert_eq!(
            key_action(KeyCode::KeyJ, ModifiersState::SHIFT, false),
            Some(Action::Set(7, 2.))
        );
        assert_eq!(
            key_action(KeyCode::KeyF, ModifiersState::SUPER, false),
            Some(Action::Fullscreen)
        );
    }
    #[test]
    fn only_continuous_controls_repeat() {
        for key in [
            KeyCode::Space,
            KeyCode::KeyE,
            KeyCode::KeyA,
            KeyCode::KeyO,
            KeyCode::KeyK,
        ] {
            assert_eq!(key_action(key, ModifiersState::empty(), true), None);
        }
        assert_eq!(key_action(KeyCode::KeyN, ModifiersState::SHIFT, true), None);
        assert_eq!(
            key_action(KeyCode::KeyM, ModifiersState::empty(), true),
            Some(Action::Lsd(0.1))
        );
    }
    #[test]
    fn fullscreen_chords_do_not_change_water_and_resets_do_not_repeat() {
        for (key, mods) in [
            (KeyCode::F11, ModifiersState::empty()),
            (KeyCode::Enter, ModifiersState::ALT),
            (KeyCode::KeyF, ModifiersState::SUPER),
        ] {
            assert_eq!(key_action(key, mods, false), Some(Action::Fullscreen));
            assert_eq!(key_action(key, mods, true), None);
        }
        assert_eq!(
            key_action(KeyCode::KeyF, ModifiersState::empty(), false),
            Some(Action::Rapids)
        );
        assert_eq!(
            key_action(KeyCode::KeyR, ModifiersState::SHIFT, false),
            Some(Action::ResetRain)
        );
        assert_eq!(key_action(KeyCode::KeyR, ModifiersState::SHIFT, true), None);
        assert_eq!(
            key_action(KeyCode::KeyM, ModifiersState::CONTROL, false),
            None
        );
    }
    #[test]
    fn intensity_clamps_and_reset_is_exact() {
        let o = <crate::cli::Options as clap::Parser>::parse_from(["water"]);
        let mut s = ViewState::from_options(&o);
        for _ in 0..40 {
            s.adjust_lsd(0.1);
        }
        assert_eq!(s.lsd, 2.);
        for _ in 0..40 {
            s.adjust_lsd(-0.1);
        }
        assert_eq!(s.lsd, 0.);
        s.lsd = 1.;
        assert_eq!(s.lsd, 1.);
    }
}
