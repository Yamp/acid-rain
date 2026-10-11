// Shared, 16-byte aligned host/shader contract. Keep in sync with ARUniforms/ARDrop.
struct Uniforms {
    viewport: vec4f, time: vec4f, camera: vec4f, optics: vec4f,
    gain: vec4f, light: vec4f, world: vec4f, post: vec4f,
    eye: vec4f, forward: vec4f, right: vec4f, up: vec4f,
    illumination: vec4f,
}
struct ARDrop {
    position: vec4f, pigment: vec4f, dynamics: vec4f, identity: vec4u,
}
fn finite1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite2(v: vec2f) -> bool { return all((bitcast<vec2u>(v) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }
fn finite3(v: vec3f) -> bool { return all((bitcast<vec3u>(v) & vec3u(0x7f800000u)) != vec3u(0x7f800000u)); }
fn finite4(v: vec4f) -> bool { return all((bitcast<vec4u>(v) & vec4u(0x7f800000u)) != vec4u(0x7f800000u)); }
