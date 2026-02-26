use std::io::Stdout;
use std::time::Duration;

use crossterm::event;

use anyhow::Result;
use crate::term::clear;
use crate::water::Water;

pub fn user_input(stdout: &mut Stdout, water: &mut Water) -> Result<bool> {
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
            }
            event::Event::Resize(w, h) => {
                clear(stdout)?;
                *water = Water::new(w, h);
            }
            _ => {}
        }
    }
    Ok(true)
}
