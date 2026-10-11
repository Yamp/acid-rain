# wgpu-core D3D12 resource-state backport

`wgpu-core/` contains the complete official crates.io package **27.0.3**, with
exactly one source change: the first transition of the internal default external
texture parameter buffer uses `COPY_DST` instead of `COPY_SRC`. The following
command copies data into that buffer, so it must be in the destination state.
The original transition produces DirectX validation errors 538 and 527.

This is the exact fix from [upstream PR 8760](https://github.com/gfx-rs/wgpu/pull/8760),
commit `bdf7d9bb5d0f8d0c4d58f7e1ec3e214cece199a3`, backported without changing
wgpu APIs, shader compilation, or rendering quality. It fixes a confirmed state
error; it does not by itself establish the cause of any other device-loss error.

The upstream archive SHA-256 is
`27a75de515543b1897b26119f93731b385a19aea165a1ec5f0e3acecc229cae7`.
The source and license hashes are in `wgpu-core-provenance.json`; the one-line
patch is in `wgpu-core-backport.patch`. Both original licenses remain in the
crate directory. All 81 package files were compared against the checksum-verified
archive, allowing only that patch.

There is no corrected 27.x release. The fix is included in 29.0.0, but not 28.0.0
or 28.0.1. Remove this local Cargo patch when the desktop dependencies are upgraded
together to an upstream version containing the correction.
