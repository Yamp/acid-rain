// Same interface as the portable textureLoad path; enabled only with FLOAT32_FILTERABLE.
@group(1) @binding(0) var linearClampSampler: sampler;
@group(1) @binding(1) var skyRepeatSampler: sampler;
@group(1) @binding(2) var mirroredSampler: sampler;
fn sample_clamp2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return textureSampleLevel(tex, linearClampSampler, uv, 0.0);
}
fn sample_repeat2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return textureSampleLevel(tex, skyRepeatSampler, uv, 0.0);
}
fn sample_mirror2(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return textureSampleLevel(tex, mirroredSampler, uv, 0.0);
}
fn sample_clamp3(tex: texture_3d<f32>, uv: vec3f) -> vec4f {
    return textureSampleLevel(tex, linearClampSampler, uv, 0.0);
}
