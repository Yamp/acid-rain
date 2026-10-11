// Compose sampling + caustics_common before this module. Forward transport
// maps transmitted beam power into a normalized spectral footprint on the bed.
struct CausticsUniforms {
    light: vec4f,
    domain: vec4f,
    grid: vec4u,
}

@group(0) @binding(0) var<uniform> causticsUniforms: CausticsUniforms;
@group(0) @binding(1) var causticsWater: texture_2d<f32>;
@group(0) @binding(2) var causticsDepth: texture_2d<f32>;
@group(0) @binding(3) var causticsMappingOut: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(4) var causticsMappingRead: texture_2d_array<f32>;

@compute @workgroup_size(8, 8, 1)
fn causticsMap(@builtin(global_invocation_id) p: vec3u) {
    let u = causticsUniforms;
    if p.x > u.grid.x || p.y > u.grid.x || p.z >= 3u { return; }
    let source = vec2f(p.xy) * (u.domain.x / f32(max(1u, u.grid.x)));
    let field = causticsField(causticsWater, source, u.domain.x);
    let n = normalize(vec3f(-field.yz, 1.0));
    let l = normalize(u.light.xyz);
    let incidence = dot(n, l);
    let ior = select(select(1.337, 1.333, p.z == 1u), 1.331, p.z == 0u);
    let ray = refract(-l, n, 1.0 / ior);
    var hit = vec4f(source, field.x, -1.0);
    if incidence > 0.0 {
        hit = causticsBedHit(causticsWater, causticsDepth, vec3f(source, field.x), ray);
    }
    let distance = hit.w;
    if distance <= 0.0 {
        textureStore(causticsMappingOut, p.xy, i32(p.z), vec4f(source, 0.0, 0.0));
        return;
    }
    let incidentFluxDensity = max(incidence, 0.0) / n.z * (1.0 - causticsFresnel(incidence, ior));
    let angularSigma = distance * max(u.light.w, 0.0) / (2.0 * ior);
    textureStore(causticsMappingOut, p.xy, i32(p.z),
                 vec4f(hit.xy, incidentFluxDensity, angularSigma * angularSigma));
}

struct CausticsVertexOut {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) center: vec2f,
    @location(1) @interpolate(flat) inverseCovariance: vec3f,
    @location(2) @interpolate(flat) amplitude: f32,
    @location(3) @interpolate(flat) channel: u32,
}

@vertex
fn causticsVertex(@builtin(vertex_index) id: u32,
                  @builtin(instance_index) channel: u32) -> CausticsVertexOut {
    let u = causticsUniforms;
    let cells = max(1u, u.grid.x);
    let cell = id / 6u;
    let p = vec2u(cell % cells, cell / cells);
    var a = textureLoad(causticsMappingRead, p, i32(channel), 0);
    var b = textureLoad(causticsMappingRead, p + vec2u(1u, 0u), i32(channel), 0);
    var c = textureLoad(causticsMappingRead, p + vec2u(0u, 1u), i32(channel), 0);
    var d = textureLoad(causticsMappingRead, p + vec2u(1u, 1u), i32(channel), 0);
    let fluxSum = a.z + b.z + c.z + d.z;
    let center = (a.xy * a.z + b.xy * b.z + c.xy * c.z + d.xy * d.z) / max(fluxSum, 1e-12);
    // Invalid neighbor coordinates must not relocate the remaining valid flux.
    if a.z <= 0.0 { a = vec4f(center, a.zw); }
    if b.z <= 0.0 { b = vec4f(center, b.zw); }
    if c.z <= 0.0 { c = vec4f(center, c.zw); }
    if d.z <= 0.0 { d = vec4f(center, d.zw); }
    let dx = 0.5 * (b.xy - a.xy + d.xy - c.xy);
    let dy = 0.5 * (c.xy - a.xy + d.xy - b.xy);
    let pixel = u.domain.z / f32(min(u.grid.y, u.grid.z));
    let filtering = max(0.5, u.domain.w) * pixel;
    let variance = filtering * filtering
        + (a.w * a.z + b.w * b.z + c.w * c.z + d.w * d.z) / max(fluxSum, 1e-12);
    let xx = (dx.x * dx.x + dy.x * dy.x) / 12.0 + variance;
    let xy = (dx.x * dx.y + dy.x * dy.y) / 12.0;
    let yy = (dx.y * dx.y + dy.y * dy.y) / 12.0 + variance;
    let determinant = max(xx * yy - xy * xy, variance * variance);
    let inverse = vec3f(yy, -xy, xx) / determinant;
    let sourceArea = pow(u.domain.x / f32(cells), 2.0);
    let flux = 0.25 * fluxSum * sourceArea;
    let amplitude = flux / (6.28318530718 * sqrt(determinant) * (1.0 - exp(-4.5)));
    let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
    let receiver = center + corners[id % 6u] * 3.0 * sqrt(vec2f(xx, yy));
    let uv = (receiver - vec2f(u.domain.y)) / u.domain.z;
    return CausticsVertexOut(vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0),
                            center, inverse, amplitude, channel);
}

@fragment
fn causticsFragment(fragmentInput: CausticsVertexOut) -> @location(0) vec4f {
    let u = causticsUniforms;
    let receiver = vec2f(u.domain.y) + fragmentInput.position.xy / vec2f(u.grid.yz) * u.domain.z;
    let delta = receiver - fragmentInput.center;
    let q = fragmentInput.inverseCovariance.x * delta.x * delta.x
        + 2.0 * fragmentInput.inverseCovariance.y * delta.x * delta.y
        + fragmentInput.inverseCovariance.z * delta.y * delta.y;
    if q > 9.0 { discard; }
    var result = vec4f(0.0);
    result[fragmentInput.channel] = fragmentInput.amplitude * exp(-0.5 * q);
    return result;
}
