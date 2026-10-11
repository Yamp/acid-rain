#!/bin/bash
set -euo pipefail
native_dir="$(cd "$(dirname "$0")" && pwd)"
app_dir="$native_dir/build/Acid Rain.app"
native_executable="$app_dir/Contents/MacOS/AcidRain"
needs_build=false
if [[ ! -x "$native_executable" ]]; then
    needs_build=true
elif [[ -n "$(find "$native_dir/Sources" "$native_dir/include" "$native_dir/Tests" "$native_dir/core/src" "$native_dir/../src" -type f -newer "$native_executable" -print -quit)" ]]; then
    needs_build=true
else
    for input in "$native_dir/build.sh" "$native_dir/Info.plist" "$native_dir/core/Cargo.toml" "$native_dir/core/Cargo.lock" "$native_dir/../Cargo.toml" "$native_dir/../Cargo.lock"; do
        if [[ "$input" -nt "$native_executable" ]]; then needs_build=true; break; fi
    done
fi
if [[ "$needs_build" == true ]] || [[ "${1:-}" == "--build" ]]; then
    if [[ "${1:-}" == "--build" ]]; then shift; fi
    "$native_dir/build.sh"
fi
exec "$app_dir/Contents/MacOS/AcidRain" "$@"
