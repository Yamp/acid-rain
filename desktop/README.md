# Acid Rain Desktop

The graphical edition renders the shared water simulation with wgpu: DirectX 12
on Windows, Vulkan on Linux and Metal on macOS. It runs independently of the
terminal renderer. The original Swift/Metal application remains in `native/`.

## Release downloads

Choose the archive for Windows x86-64, Linux x86-64, macOS Apple Silicon (arm64),
or macOS Intel (x86-64). Each archive includes the desktop and terminal executables,
launchers, documentation, license and SHA-256 checksums. A Rust installation is not
required. GPU shaders are embedded in the desktop executable.
Windows release executables link the MSVC runtime statically and do not require
installing the Visual C++ redistributable.

- Windows: extract the ZIP and double-click `acid-rain-desktop.exe`. The optional
  `Start Desktop.cmd` helper starts the graphical app and closes its command window.
- Linux: extract the tarball and run `./start-desktop.sh` inside the extracted directory.
- macOS: open `Acid Rain Desktop.app`. The Apple Silicon archive also includes the
  original `Acid Rain Metal.app`. Applications are ad-hoc signed, not notarized.

The Linux binary is built on Ubuntu 22.04 and requires glibc 2.35 or newer, a Vulkan
driver, `libxkbcommon`, and the runtime libraries for X11 or Wayland. Install the
Vulkan driver supplied for your GPU; software Vulkan is useful for validation but
does not represent normal hardware performance. On macOS the minimum version is 14.

`h` shows the in-app controls. `acid-rain-desktop --help` lists command-line options.
Choose a backend explicitly with `--backend vulkan`, `--backend dx12`, or
`--backend metal`; `--backend auto` uses the platform's preferred available backend.

## Build from source

Install a current stable Rust toolchain. From the repository root:

```sh
cargo check --locked --workspace --all-targets
cargo build --locked --release -p acid-rain-desktop
cargo run --locked --release -p acid-rain-desktop --
```

On Ubuntu, the build dependencies are:

```sh
sudo apt-get install build-essential pkg-config libx11-dev libxi-dev libxcursor-dev \
  libxrandr-dev libxinerama-dev libwayland-dev libxkbcommon-dev libegl1-mesa-dev libvulkan-dev
```

Windows builds use the MSVC Rust toolchain and Visual Studio C++ Build Tools.
macOS builds use Xcode Command Line Tools. Portable releases do not use
`-C target-cpu=native`; that option is only suitable for binaries built for your own CPU.
Build the original native app with `ACID_RAIN_PORTABLE=1 ./native/build.sh` when packaging it.

## Rendering checks

Headless mode executes the actual graphics pipeline and writes a PNG and JSON report:

```sh
cargo check --locked -p acid-rain-desktop --all-targets
cargo run --locked --release -p acid-rain-desktop -- --headless --frames 120 \
  --warmup 20 --size 1280x800 --output artifacts/desktop.png --report artifacts/desktop.json
```

`--software` requests a software adapter. CI requires numerical GPU tests and a
captured frame under both Linux Mesa Vulkan and Windows WARP DirectX 12. Missing
software adapters fail the required numerical tests. Software-renderer timings
must not be reported as hardware GPU FPS.

`--no-gpu-timestamps` preserves the visual rendering graph, uses queue completion
for automatic quality control, and reports GPU intervals as `null`.

Every platform runs locked CPU tests and WGSL validation. Tagged releases run the
same checks before publishing four platform archives, `validation.zip`, and a
top-level `SHA256SUMS`. The workflow rejects mismatched package versions and verifies
the archive hashes after artifact download.
