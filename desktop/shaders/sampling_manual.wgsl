// Portable linear interpolation for FP32 state. No float32-filtering feature required.
// Texel centres and boundary addressing match the hardware samplers exactly.
@group(1) @binding(0) var linearClampSampler: sampler;
@group(1) @binding(1) var skyRepeatSampler: sampler;
@group(1) @binding(2) var mirroredSampler: sampler;
fn mirror_index(i: vec2i, size: vec2i) -> vec2i {
    let period = 2 * size;
    let p = ((i % period) + period) % period;
    return min(p, period - 1 - p);
}
fn repeat_u_index(i: vec2i, size: vec2i) -> vec2i {
    return vec2i(((i.x % size.x) + size.x) % size.x, clamp(i.y, 0, size.y - 1));
}
fn load_addressed2(tex: texture_2d<f32>, i: vec2i, size: vec2i, mode: u32) -> vec4f {
    var p = clamp(i, vec2i(0), size - 1);
    if mode == 1u { p = repeat_u_index(i, size); }
    if mode == 2u { p = mirror_index(i, size); }
    return textureLoad(tex, p, 0);
}
fn sample_linear2(tex: texture_2d<f32>, uv: vec2f, mode: u32) -> vec4f {
    let size = vec2i(textureDimensions(tex));
    let p = uv * vec2f(size) - 0.5;
    let cell = vec2i(floor(p));
    let f = fract(p);
    return mix(mix(load_addressed2(tex, cell, size, mode),
                   load_addressed2(tex, cell + vec2i(1, 0), size, mode), f.x),
               mix(load_addressed2(tex, cell + vec2i(0, 1), size, mode),
                   load_addressed2(tex, cell + vec2i(1, 1), size, mode), f.x), f.y);
}
fn sample_clamp2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return sample_linear2(tex, clamp(uv, vec2f(0), vec2f(1)), 0u);
}
fn sample_repeat2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    // Reduce the periodic coordinate before converting it to integer texel addresses.
    return sample_linear2(tex, vec2f(fract(uv.x), clamp(uv.y, 0.0, 1.0)), 1u);
}
fn sample_mirror2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return sample_linear2(tex, uv - 2.0 * floor(uv * 0.5), 2u);
}
fn sample_clamp3(tex: texture_3d<f32>, uv: vec3f) -> vec4f {
    let size = vec3i(textureDimensions(tex));
    let p = clamp(uv, vec3f(0), vec3f(1)) * vec3f(size) - 0.5;
    let cell = vec3i(floor(p));
    let f = fract(p);
    let lo = clamp(cell, vec3i(0), size - 1);
    let hi = clamp(cell + 1, vec3i(0), size - 1);
    let z0 = mix(mix(textureLoad(tex, vec3i(lo.x, lo.y, lo.z), 0),
                     textureLoad(tex, vec3i(hi.x, lo.y, lo.z), 0), f.x),
                 mix(textureLoad(tex, vec3i(lo.x, hi.y, lo.z), 0),
                     textureLoad(tex, vec3i(hi.x, hi.y, lo.z), 0), f.x), f.y);
    let z1 = mix(mix(textureLoad(tex, vec3i(lo.x, lo.y, hi.z), 0),
                     textureLoad(tex, vec3i(hi.x, lo.y, hi.z), 0), f.x),
                 mix(textureLoad(tex, vec3i(lo.x, hi.y, hi.z), 0),
                     textureLoad(tex, vec3i(hi.x, hi.y, hi.z), 0), f.x), f.y);
    return mix(z0, z1, f.z);
}
