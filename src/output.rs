pub type Cell = ((u8, u8, u8), (u8, u8, u8));

const SYNC_BEGIN: &[u8] = b"\x1b[?2026h";
const SYNC_END: &[u8] = b"\x1b[?2026l";
const HALF_BLOCK: &[u8] = "▀".as_bytes();

/// Encodes RGB half-block cells without allocating strings or performing I/O.
pub struct TerminalOutput {
    width: u16,
    height: u16,
    previous: Vec<Cell>,
    valid: bool,
    clear_pending: bool,
    status: Vec<u8>,
    status_dirty: bool,
    painted_status_width: usize,
}

impl TerminalOutput {
    pub fn new(
        width: u16,
        height: u16,
    ) -> Self {
        Self {
            width,
            height,
            previous: vec![((0, 0, 0), (0, 0, 0)); usize::from(width) * usize::from(height)],
            valid: false,
            clear_pending: true,
            status: Vec::new(),
            status_dirty: false,
            painted_status_width: 0,
        }
    }

    pub fn resize(
        &mut self,
        width: u16,
        height: u16,
    ) {
        self.width = width;
        self.height = height;
        self.previous.resize(
            usize::from(width) * usize::from(height),
            ((0, 0, 0), (0, 0, 0)),
        );
        self.valid = false;
        self.clear_pending = true;
        self.painted_status_width = 0;
    }

    pub fn invalidate(&mut self) { self.valid = false; }

    pub fn status(&self) -> Option<&str> {
        (!self.status.is_empty())
            .then(|| std::str::from_utf8(&self.status).expect("status is sanitized ASCII"))
    }

    /// Set a first-row message. Control codes and non-ASCII text are discarded.
    /// The full message is retained so a resize can reveal its clipped suffix.
    pub fn set_status(
        &mut self,
        status: Option<&str>,
    ) {
        let sanitized: Vec<u8> = status
            .unwrap_or_default()
            .bytes()
            .filter(|byte| (b' '..=b'~').contains(byte))
            .take(usize::from(u16::MAX))
            .collect();
        if self.status != sanitized {
            self.status = sanitized;
            self.status_dirty = true;
        }
    }

    /// Clears `buffer` and encodes changes from the last complete frame.
    /// Incomplete frames are ignored, preserving the previous screen state.
    pub fn encode(
        &mut self,
        cells: &[Cell],
        buffer: &mut Vec<u8>,
    ) {
        buffer.clear();
        let width = usize::from(self.width);
        if width == 0 || self.height == 0 || cells.len() < self.previous.len() {
            return;
        }
        let status_width = self.status.len().min(width);

        // Reset these every frame: terminal state may change between writes.
        let mut foreground = None;
        let mut background = None;
        for y in 0..self.height {
            let offset = usize::from(y) * width;
            let mut contiguous = false;
            for x in 0..self.width {
                let index = offset + usize::from(x);
                let cell = cells[index];
                if y == 0 && usize::from(x) < status_width {
                    // Keep the live surface under the message without drawing
                    // over it. Removal explicitly restores these cells below.
                    self.previous[index] = cell;
                    contiguous = false;
                    continue;
                }
                let restore_status = y == 0 && usize::from(x) < self.painted_status_width;
                if self.valid && self.previous[index] == cell && !restore_status {
                    contiguous = false;
                    continue;
                }
                begin_frame(buffer, self.clear_pending);
                // Every row starts with an absolute move, cancelling pending
                // autowrap after writing the preceding row's final column.
                if !contiguous {
                    buffer.extend_from_slice(b"\x1b[");
                    push_decimal(buffer, y + 1);
                    buffer.push(b';');
                    push_decimal(buffer, x + 1);
                    buffer.push(b'H');
                }
                let foreground_changed = foreground != Some(cell.0);
                let background_changed = background != Some(cell.1);
                if foreground_changed || background_changed {
                    buffer.extend_from_slice(b"\x1b[");
                    if foreground_changed {
                        buffer.extend_from_slice(b"38;2;");
                        push_rgb(buffer, cell.0);
                        foreground = Some(cell.0);
                    }
                    if background_changed {
                        if foreground_changed {
                            buffer.push(b';');
                        }
                        buffer.extend_from_slice(b"48;2;");
                        push_rgb(buffer, cell.1);
                        background = Some(cell.1);
                    }
                    buffer.push(b'm');
                }
                buffer.extend_from_slice(HALF_BLOCK);
                self.previous[index] = cell;
                contiguous = true;
            }
        }
        if status_width > 0 && (self.status_dirty || !self.valid) {
            begin_frame(buffer, self.clear_pending);
            buffer.extend_from_slice(b"\x1b[1;1H\x1b[38;2;212;225;255;48;2;16;19;32m");
            buffer.extend_from_slice(&self.status[..status_width]);
        }
        if !buffer.is_empty() {
            buffer.extend_from_slice(SYNC_END);
        }
        self.valid = true;
        self.clear_pending = false;
        self.status_dirty = false;
        self.painted_status_width = status_width;
    }
}

fn begin_frame(
    buffer: &mut Vec<u8>,
    clear: bool,
) {
    if buffer.is_empty() {
        buffer.extend_from_slice(SYNC_BEGIN);
        if clear {
            buffer.extend_from_slice(b"\x1b[2J");
        }
    }
}

fn push_rgb(
    buffer: &mut Vec<u8>,
    (red, green, blue): (u8, u8, u8),
) {
    push_decimal(buffer, u16::from(red));
    buffer.push(b';');
    push_decimal(buffer, u16::from(green));
    buffer.push(b';');
    push_decimal(buffer, u16::from(blue));
}

fn push_decimal(
    buffer: &mut Vec<u8>,
    mut value: u16,
) {
    let mut digits = [0; 5];
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    buffer.extend_from_slice(&digits[start..]);
}

#[cfg(test)]
mod tests {
    use super::{
        Cell,
        TerminalOutput,
    };

    const BLACK: Cell = ((0, 0, 0), (0, 0, 0));
    const COLOR: Cell = ((17, 101, 255), (8, 64, 193));
    const BEGIN: &str = "\x1b[?2026h";
    const END: &str = "\x1b[?2026l";

    fn encoded(
        output: &mut TerminalOutput,
        cells: &[Cell],
    ) -> String {
        let mut bytes = Vec::new();
        output.encode(cells, &mut bytes);
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn initial_black_frame_replaces_existing_terminal_content() {
        let mut output = TerminalOutput::new(2, 1);
        let frame = encoded(&mut output, &[BLACK; 2]);
        assert!(frame.starts_with(BEGIN));
        assert!(frame.ends_with(END));
        assert!(frame.contains("\x1b[1;1H"));
        assert_eq!(frame.matches('▀').count(), 2);
        assert!(frame.contains("38;2;0;0;0"));
        assert!(frame.contains("48;2;0;0;0"));
    }

    #[test]
    fn unchanged_frame_is_empty_and_reuses_callers_buffer() {
        let mut output = TerminalOutput::new(2, 1);
        encoded(&mut output, &[COLOR; 2]);
        let mut buffer = b"stale output".to_vec();
        output.encode(&[COLOR; 2], &mut buffer);
        assert!(buffer.is_empty());
    }

    #[test]
    fn separated_changes_move_cursor_over_unchanged_cells() {
        let mut output = TerminalOutput::new(4, 1);
        encoded(&mut output, &[BLACK; 4]);
        let frame = encoded(&mut output, &[COLOR, BLACK, COLOR, COLOR]);
        assert!(frame.contains("\x1b[1;1H"));
        assert!(frame.contains("\x1b[1;3H"));
        assert!(!frame.contains("\x1b[1;4H"));
        assert_eq!(frame.matches('▀').count(), 3);
        assert_eq!(frame.matches("38;2;").count(), 1);
    }

    #[test]
    fn next_row_moves_explicitly_after_last_column() {
        let mut output = TerminalOutput::new(2, 2);
        let frame = encoded(&mut output, &[COLOR; 4]);
        assert!(frame.contains("▀▀\x1b[2;1H▀▀"));
    }

    #[test]
    fn resize_clears_inside_sync_and_repaints_every_cell() {
        let mut output = TerminalOutput::new(2, 2);
        encoded(&mut output, &[COLOR; 4]);
        output.resize(3, 1);
        let frame = encoded(&mut output, &[COLOR; 3]);
        assert!(frame.starts_with(&format!("{BEGIN}\x1b[2J")));
        assert_eq!(frame.matches('▀').count(), 3);
        assert!(encoded(&mut output, &[COLOR; 3]).is_empty());
    }

    #[test]
    fn invalidate_repaints_an_unchanged_frame() {
        let mut output = TerminalOutput::new(2, 1);
        encoded(&mut output, &[COLOR; 2]);
        output.invalidate();
        assert_eq!(encoded(&mut output, &[COLOR; 2]).matches('▀').count(), 2);
    }

    #[test]
    fn contiguous_colors_use_one_cursor_move_and_preserve_full_rgb() {
        let mut output = TerminalOutput::new(80, 1);
        let frame = encoded(&mut output, &[COLOR; 80]);
        assert_eq!(frame.matches('H').count(), 1);
        assert_eq!(frame.matches("38;2;17;101;255").count(), 1);
        assert_eq!(frame.matches("48;2;8;64;193").count(), 1);
        assert!(
            frame.len() < 320,
            "contiguous row took {} bytes",
            frame.len()
        );
    }

    #[test]
    fn zero_size_emits_nothing_and_can_grow_again() {
        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let mut output = TerminalOutput::new(width, height);
            assert!(encoded(&mut output, &[]).is_empty());
            output.resize(1, 1);
            assert_eq!(encoded(&mut output, &[BLACK]).matches('▀').count(), 1);
        }
    }

    /// Apply the supported ANSI subset to a screen, including delayed autowrap.
    fn apply_frame(
        frame: &str,
        screen: &mut [Cell],
        width: usize,
    ) {
        let bytes = frame.as_bytes();
        let mut cursor = (0, 0);
        let mut color = ((9, 9, 9), (9, 9, 9));
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'\x1b' {
                assert_eq!(bytes[index + 1], b'[');
                let start = index + 2;
                index = start;
                while !(b'@'..=b'~').contains(&bytes[index]) {
                    index += 1;
                }
                let parameters = &frame[start..index];
                let command = bytes[index];
                index += 1;
                if parameters == "?2026" {
                    assert!(command == b'h' || command == b'l');
                    continue;
                }
                let values: Vec<usize> =
                    parameters.split(';').map(|s| s.parse().unwrap()).collect();
                match command {
                    b'H' => cursor = (values[1] - 1, values[0] - 1),
                    b'J' => {
                        assert_eq!(values, [2]);
                        screen.fill(BLACK);
                    },
                    b'm' => {
                        for rgb in values.as_chunks::<5>().0 {
                            assert_eq!(rgb[1], 2);
                            let value = (rgb[2] as u8, rgb[3] as u8, rgb[4] as u8);
                            match rgb[0] {
                                38 => color.0 = value,
                                48 => color.1 = value,
                                _ => panic!("unsupported color: {rgb:?}"),
                            }
                        }
                    },
                    _ => panic!("unsupported CSI: {command}"),
                }
            } else {
                assert!(frame[index..].starts_with('▀'));
                if cursor.0 == width {
                    cursor = (0, cursor.1 + 1);
                }
                screen[cursor.1 * width + cursor.0] = color;
                cursor.0 += 1;
                index += '▀'.len_utf8();
            }
        }
    }

    #[test]
    fn decoded_screen_matches_sparse_rgb_updates_and_resizes() {
        let mut output = TerminalOutput::new(1, 1);
        for (width, height) in [(13, 4), (4, 13), (1, 1), (27, 2)] {
            output.resize(width, height);
            let mut cells = vec![BLACK; usize::from(width) * usize::from(height)];
            let mut screen = vec![COLOR; cells.len()];
            for frame_index in 0..32 {
                for (index, cell) in cells.iter_mut().enumerate() {
                    if (index + frame_index) % 5 == 0 {
                        cell.0 = ((index * 17) as u8, frame_index as u8, 255);
                    }
                    if (index + frame_index) % 7 == 0 {
                        cell.1 = (frame_index as u8, (index * 31) as u8, 0);
                    }
                }
                let frame = encoded(&mut output, &cells);
                apply_frame(&frame, &mut screen, usize::from(width));
                assert_eq!(screen, cells, "frame {frame_index} at {width}x{height}");
            }
        }
    }

    #[test]
    fn status_is_clipped_to_first_row_inside_synchronized_frame() {
        let mut output = TerminalOutput::new(8, 2);
        encoded(&mut output, &[COLOR; 16]);
        output.set_status(Some("Quality: High"));
        let frame = encoded(&mut output, &[COLOR; 16]);
        assert!(frame.starts_with(BEGIN));
        assert!(frame.ends_with(END));
        assert_eq!(frame.matches(BEGIN).count(), 1);
        assert_eq!(frame.matches(END).count(), 1);
        assert!(frame.contains("\x1b[1;1H"));
        assert!(frame.contains("Quality:"));
        assert!(!frame.contains("High"));
        assert_eq!(frame.matches('▀').count(), 0);
        assert!(encoded(&mut output, &[COLOR; 16]).is_empty());
    }

    #[test]
    fn clearing_status_restores_only_covered_cells_even_on_static_surface() {
        let mut output = TerminalOutput::new(20, 2);
        encoded(&mut output, &[COLOR; 40]);
        output.set_status(Some("Quality: High"));
        encoded(&mut output, &[COLOR; 40]);
        output.set_status(None);
        let frame = encoded(&mut output, &[COLOR; 40]);
        assert_eq!(frame.matches('▀').count(), "Quality: High".len());
        assert_eq!(frame.matches('H').count(), 1);
        assert!(!frame.contains("\x1b[2J"));
        assert!(encoded(&mut output, &[COLOR; 40]).is_empty());
    }

    #[test]
    fn shortened_status_restores_the_old_tail_without_repainting_other_rows() {
        let mut output = TerminalOutput::new(20, 2);
        output.set_status(Some("Quality: High"));
        encoded(&mut output, &[COLOR; 40]);
        output.set_status(Some("Auto"));
        let frame = encoded(&mut output, &[COLOR; 40]);
        assert_eq!(frame.matches('▀').count(), 9);
        assert!(frame.contains("\x1b[1;5H"));
        assert!(frame.contains("Auto"));
        assert!(!frame.contains("\x1b[2;"));
        output.set_status(None);
        assert_eq!(encoded(&mut output, &[COLOR; 40]).matches('▀').count(), 4);
    }

    #[test]
    fn hidden_surface_changes_are_restored_when_status_disappears() {
        let mut output = TerminalOutput::new(20, 2);
        let mut cells = [COLOR; 40];
        output.set_status(Some("Quality: High"));
        encoded(&mut output, &cells);
        cells[0] = BLACK;
        cells[5] = ((255, 128, 0), (1, 2, 3));
        assert!(encoded(&mut output, &cells).is_empty());
        output.set_status(None);
        let frame = encoded(&mut output, &cells);
        let mut screen = [COLOR; 40];
        apply_frame(&frame, &mut screen, 20);
        assert_eq!(screen, cells);
    }

    #[test]
    fn resize_reclips_status_and_clear_restores_new_width() {
        let mut output = TerminalOutput::new(20, 2);
        output.set_status(Some("Quality: High"));
        encoded(&mut output, &[COLOR; 40]);
        output.resize(3, 2);
        let frame = encoded(&mut output, &[COLOR; 6]);
        assert!(frame.contains("Qua"));
        assert!(!frame.contains("Quality"));
        output.set_status(None);
        assert_eq!(encoded(&mut output, &[COLOR; 6]).matches('▀').count(), 3);
        output.resize(0, 0);
        output.set_status(Some("Low"));
        assert!(encoded(&mut output, &[]).is_empty());
        output.resize(10, 1);
        assert!(encoded(&mut output, &[COLOR; 10]).contains("Low"));
    }

    #[test]
    fn status_accepts_only_printable_ascii_and_cannot_inject_terminal_controls() {
        let mut output = TerminalOutput::new(80, 1);
        encoded(&mut output, &[COLOR; 80]);
        output.set_status(Some("Quality:\x1b[2J\nHigh\t🔥"));
        let frame = encoded(&mut output, &[COLOR; 80]);
        assert!(frame.contains("Quality:[2JHigh"));
        assert!(!frame.contains("\x1b[2J"));
        assert!(!frame.contains('\n'));
        assert!(!frame.contains('\t'));
        assert!(!frame.contains('🔥'));
        output.set_status(Some("Quality:[2JHigh"));
        assert!(encoded(&mut output, &[COLOR; 80]).is_empty());
    }
}
