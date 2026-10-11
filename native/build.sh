#!/bin/bash
set -euo pipefail
native_dir="$(cd "$(dirname "$0")" && pwd)"
app_dir="$native_dir/build/Acid Rain.app"
native_target_dir="${CARGO_TARGET_DIR:-$native_dir/../target}"
native_core_lib="$native_target_dir/release/libacid_rain_native_core.a"
mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources" "$native_dir/build"
export MACOSX_DEPLOYMENT_TARGET=14.0
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-mmacosx-version-min=14.0"
if [[ "${ACID_RAIN_PORTABLE:-0}" == 1 ]]; then
    if [[ "$RUSTFLAGS ${CARGO_ENCODED_RUSTFLAGS:-}" == *target-cpu=native* ]]; then
        printf 'Portable builds cannot contain target-cpu=native in Rust flags.\n' >&2
        exit 1
    fi
elif [[ "$RUSTFLAGS" != *target-cpu* ]]; then
    export RUSTFLAGS="$RUSTFLAGS -C target-cpu=native"
fi
cargo check --locked --manifest-path "$native_dir/core/Cargo.toml" --target-dir "$native_target_dir" --all-targets
cargo build --locked --manifest-path "$native_dir/core/Cargo.toml" --target-dir "$native_target_dir" --release
xcrun swiftc "$native_dir/Sources/RenderPolicy.swift" "$native_dir/Tests/RenderPolicyTests.swift" -o "$native_dir/build/render-policy-tests"
"$native_dir/build/render-policy-tests"
xcrun swiftc "$native_dir/Sources/RenderPolicy.swift" "$native_dir/Sources/TemporalPolicy.swift" "$native_dir/Tests/TemporalPolicyTests.swift" -o "$native_dir/build/temporal-policy-tests"
"$native_dir/build/temporal-policy-tests"
xcrun swiftc -import-objc-header "$native_dir/include/AcidRainCore.h" \
    "$native_dir/Sources/CoreBridge.swift" "$native_dir/Tests/CoreClockTests.swift" \
    "$native_core_lib" \
    -framework QuartzCore -framework Security -liconv -lresolv -o "$native_dir/build/core-clock-tests"
"$native_dir/build/core-clock-tests"
xcrun swiftc -import-objc-header "$native_dir/include/AcidRainCore.h" "$native_dir/Sources/Camera.swift" "$native_dir/Tests/CameraTests.swift" -o "$native_dir/build/camera-tests"
"$native_dir/build/camera-tests"
xcrun swiftc "$native_dir/Sources/KeyboardInput.swift" "$native_dir/Tests/KeyboardInputTests.swift" -o "$native_dir/build/keyboard-tests"
"$native_dir/build/keyboard-tests"
cp "$native_dir/Info.plist" "$app_dir/Contents/Info.plist"
cat "$native_dir/Sources/Cloud.metal" "$native_dir/Sources/Caustics.metal" "$native_dir/Sources/BedLife.metal" "$native_dir/Sources/Water.metal" > "$native_dir/build/Water.metal"
cp "$native_dir/build/Water.metal" "$app_dir/Contents/Resources/Water.metal"
if xcrun -sdk macosx metal -c "$native_dir/build/Water.metal" -o "$native_dir/build/Water.air" > "$native_dir/build/metal-compile.log" 2>&1; then
    xcrun -sdk macosx metallib "$native_dir/build/Water.air" -o "$app_dir/Contents/Resources/Water.metallib"
    printf 'Metal shaders compiled ahead of time.\n'
else
    if ! grep -Eq 'missing Metal Toolchain|cannot execute tool|unable to find utility' "$native_dir/build/metal-compile.log"; then
        cat "$native_dir/build/metal-compile.log" >&2
        exit 1
    fi
    rm -f "$app_dir/Contents/Resources/Water.metallib"
    printf 'Offline Metal compiler unavailable; using the macOS runtime Metal compiler.\n'
    printf 'Compiler details: %s\n' "$native_dir/build/metal-compile.log"
fi
xcrun swiftc -O -whole-module-optimization -target "$(uname -m)-apple-macos14.0" \
    -import-objc-header "$native_dir/include/AcidRainCore.h" \
    "$native_dir"/Sources/*.swift \
    "$native_core_lib" \
    -framework AppKit -framework Metal -framework MetalKit -framework QuartzCore \
    -framework ImageIO -framework UniformTypeIdentifiers -framework Security -liconv -lresolv \
    -o "$app_dir/Contents/MacOS/AcidRain"
codesign --force --sign - --timestamp=none "$app_dir" >/dev/null
printf '\nBuilt: %s\nRun: %s/run.sh\n' "$app_dir" "$native_dir"
