# Acid Rain

Acid Rain is a terminal-based screensaver that emulates the effect of acid raindrops falling and rippling on a watery surface. It's inspired by the classic `cmatrix` animation and provides a visually engaging experience for users who spend a lot of time in the terminal.

## Installation

### From Crates.io

You can install Acid Rain directly from crates.io using Cargo, Rust's package manager:

```sh
cargo install acid-rain
```

### Debian/Ubuntu (apt-get)

Acid Rain can be installed on Debian-based systems using `apt-get`. Please note that the package might not be available in the official repository and may require adding a custom repository:

```sh
sudo apt-get update
sudo apt-get install acid-rain
```

### macOS (Homebrew)

For macOS users, Acid Rain can be installed using Homebrew. The formula might need to be tapped from a custom repository:

```sh
brew tap acid-rain/formulae
brew install acid-rain
```

## Usage

After installation, you can run Acid Rain by simply typing:

```sh
acid-rain
```

Use the following keyboard shortcuts to interact with the screensaver:

- `ESC` or `Ctrl+C`: Exit the screensaver

## Features

- Realistic raindrop ripple effect
- Customizable raindrop frequency and color settings
- Support for terminal resizing
- Low CPU usage

## Dependencies

Acid Rain is built using Rust and depends on several crates listed in the `Cargo.toml` file. The key dependencies include `crossterm` for terminal handling, `rand` for generating random raindrops, and `ndarray` for managing the water surface state.

## License

Acid Rain is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions to Acid Rain are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## Credits

Acid Rain was created by [yampolson](https://github.com/Yamp). Special thanks to all the contributors who have helped improve the project.