#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
cargo check --locked
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
cargo build --locked --release
exec ./target/release/acid-rain "$@"
