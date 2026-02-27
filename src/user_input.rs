use std::io::Stdout;
use std::time::Duration;

use crossterm::event;

use anyhow::Result;
use crate::term::Renderer;
use crate::water::Water;

pub fn user_input(_stdout: &mut Stdout, water: &mut Water, renderer: &mut Renderer) -> Result<bool> {
    while event::poll(Duration::ZERO)? {
        match event::read()? {
            event::Event::Key(keyevent) => {
                if keyevent
                    == event::KeyEvent::new(event::KeyCode::Char('c'), event::KeyModifiers::CONTROL)
                    || keyevent
                        == event::KeyEvent::new(event::KeyCode::Esc, event::KeyModifiers::NONE)
                {
                    return Ok(false);
                }
                if keyevent.code == event::KeyCode::Char('g') {
                    renderer.cycle_preset();
                }
            }
            event::Event::Resize(w, h) => {
                water.resize(w, h);
            }
            _ => {}
        }
    }
    Ok(true)
}
