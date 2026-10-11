#include <metal_stdlib>
using namespace metal;

// Three-dimensional tracer transport under a prescribed wake closure. The water
// surface remains the shared shallow-water model; this is not a 3-D NS solver.
// Concentration is pigment mass per cubic metre, so sum(c * cellVolume) is mass.
struct CloudUniforms {
    float4 step;    // dt<=.008, physical time, bleach rate, this pass's drop count
    float4 rebase;  // integer XY voxel shift, clear volume, inject new sources
    float4 domain;  // XY extent metres, lower Z, vertical extent metres
    float4 options;// world origin X mod 2pi, wake strength, diffusion m²/s, origin Y mod 2pi
};
struct CloudDrop {
    float4 position;
    float4 pigment;
    float4 dynamics;
    uint4 identity;
};

constant sampler cloudFieldSampler(coord::normalized, address::clamp_to_edge, filter::linear);

static uint3 cloudSize(texture3d<float, access::read> volume) {
    return uint3(volume.get_width(), volume.get_height(), volume.get_depth());
}
static float3 cloudCell(constant CloudUniforms &u, uint3 size) {
    return float3(u.domain.xy, u.domain.w) / float3(size);
}
static float3 cloudPosition(uint3 index, uint3 size, constant CloudUniforms &u) {
    return (float3(index) + .5f) * cloudCell(u, size) + float3(0, 0, u.domain.z);
}
static float4 cloudField(texture2d<float> field, float2 p, constant CloudUniforms &u) {
    return field.sample(cloudFieldSampler, p / u.domain.xy + .5f / float2(field.get_width(), field.get_height()));
}
static bool cloudFluid(float3 p, texture2d<float> water, texture2d<float> flow, constant CloudUniforms &u) {
    float surface = cloudField(water, p.xy, u).x;
    float depth = max(.01f, cloudField(flow, p.xy, u).w);
    return p.z <= surface && p.z >= surface - depth;
}
static float3 cloudSourceCenter(CloudDrop drop) {
    return drop.position.xyz - float3(0, 0, .035f + 1.5f * drop.position.w);
}
static float3 cloudSigma(CloudDrop drop) {
    return max(float3(drop.dynamics.xx, drop.dynamics.y), float3(.006f));
}
static float cloudGaussian(float3 position, CloudDrop drop) {
    float3 q = (position - cloudSourceCenter(drop)) / cloudSigma(drop);
    if (any(abs(q) > 4.f)) return 0;
    return exp(-.5f * dot(q, q));
}

// One bounded reduction per new impact, not per voxel. Normalization uses the
// exact same finite support and fluid mask as injection, including clipped edges.
kernel void cloudNormalize(constant CloudUniforms &u [[buffer(0)]],
                           device const CloudDrop *drops [[buffer(1)]],
                           device float4 *normalization [[buffer(2)]],
                           texture3d<float, access::read> input [[texture(0)]],
                           texture2d<float> flow [[texture(3)]],
                           texture2d<float> water [[texture(4)]],
                           uint index [[thread_position_in_grid]]) {
    if (index >= uint(u.step.w)) return;
    CloudDrop drop = drops[index];
    uint3 size = cloudSize(input);
    float3 cell = cloudCell(u, size);
    float3 center = cloudSourceCenter(drop) - float3(0, 0, u.domain.z);
    float3 support = 4.f * cloudSigma(drop);
    int3 lo = max(int3(0), int3(ceil((center - support) / cell - .5f)));
    int3 hi = min(int3(size) - 1, int3(floor((center + support) / cell - .5f)));
    float integral = 0;
    if (drop.identity.z == 1) {
        for (int z = lo.z; z <= hi.z; ++z) {
            for (int y = lo.y; y <= hi.y; ++y) {
                for (int x = lo.x; x <= hi.x; ++x) {
                    float3 p = cloudPosition(uint3(x, y, z), size, u);
                    if (cloudFluid(p, water, flow, u)) integral += cloudGaussian(p, drop);
                }
            }
        }
    }
    integral *= cell.x * cell.y * cell.z;
    normalization[index] = float4(integral > 1e-20f ? 1.f / integral : 0.f, integral, 0, 0);
}

kernel void cloudPrepare(constant CloudUniforms &u [[buffer(0)]],
                         device const CloudDrop *drops [[buffer(1)]],
                         device const float4 *normalization [[buffer(2)]],
                         texture3d<float, access::read> input [[texture(0)]],
                         texture3d<float, access::write> output [[texture(1)]],
                         texture2d<float> flow [[texture(3)]],
                         texture2d<float> water [[texture(4)]],
                         uint3 index [[thread_position_in_grid]]) {
    uint3 size = cloudSize(input);
    if (any(index >= size)) return;
    int3 oldIndex = int3(index) + int3(int2(round(u.rebase.xy)), 0);
    float3 pigment = 0;
    if (u.rebase.z < .5f && all(oldIndex >= 0) && all(oldIndex < int3(size))) pigment = input.read(uint3(oldIndex)).rgb;
    if (u.rebase.w > .5f) {
        float3 p = cloudPosition(index, size, u);
        if (cloudFluid(p, water, flow, u)) {
            for (uint i = 0; i < uint(u.step.w); ++i) {
                CloudDrop drop = drops[i];
                if (drop.identity.z != 1) continue;
                float source = cloudGaussian(p, drop) * normalization[i].x * max(0.f, drop.pigment.w);
                pigment += max(float3(0), drop.pigment.rgb) * source;
            }
        }
    }
    output.write(float4(pigment, 0), index);
}

// Analytic axisymmetric ring from psi=A*r² exp(-r²/a²-(z-zc)²/b²).
// Before the boundary/CFL limits this velocity is analytically divergence-free.
static float3 cloudRing(float3 p, CloudDrop drop, float time) {
    float age = time - drop.dynamics.z;
    if (age < 0 || age > 2.5f) return 0;
    float ageRoot = sqrt(age + .025f);
    // A sub-voxel wake can advect a blob but cannot resolve its rolled rim.
    // Keep the coherent core at least 4–5 voxels across on the bounded grid.
    float a = max(.07f, 1.4f * drop.dynamics.x) + .03f * ageRoot;
    float b = max(.045f, drop.dynamics.y) + .018f * ageRoot;
    float speed = clamp(drop.dynamics.w, 0.f, .45f);
    float3 center = cloudSourceCenter(drop);
    center.z -= .30f * speed * age / (1.f + .28f * age);
    float3 delta = p - center;
    float radial2 = dot(delta.xy, delta.xy);
    float q = radial2 / (a * a) + delta.z * delta.z / (b * b);
    if (q > 18.f) return 0;
    float A = -.5f * speed * exp(-age * .30f);
    float E = exp(-q);
    float2 radialVelocity = (2.f * A * delta.z / (b * b) * E) * delta.xy;
    float verticalVelocity = 2.f * A * (1.f - radial2 / (a * a)) * E;
    return float3(radialVelocity, verticalVelocity);
}

kernel void cloudVelocity(constant CloudUniforms &u [[buffer(0)]],
                          device const CloudDrop *drops [[buffer(1)]],
                          texture3d<float, access::read> input [[texture(0)]],
                          texture3d<float, access::write> velocity [[texture(2)]],
                          texture2d<float> flow [[texture(3)]],
                          texture2d<float> water [[texture(4)]],
                          uint3 index [[thread_position_in_grid]]) {
    uint3 size = cloudSize(input);
    if (any(index >= size)) return;
    float3 p = cloudPosition(index, size, u);
    if (!cloudFluid(p, water, flow, u)) { velocity.write(float4(0), index); return; }
    float4 ambient = cloudField(flow, p.xy, u);
    float3 v = float3(ambient.xy, 0);
    uint count = min(24u, uint(u.step.w));
    for (uint i = 0; i < count; ++i) {
        if (drops[i].identity.z == 1) v += u.options.y * cloudRing(p, drops[i], u.step.y);
    }
    // Two oblique Taylor-Green curls supply coherent strain and stretched folds
    // without random voxel noise. Amplitude is small relative to the impact.
    float k = 24.f;
    // Integer spatial frequencies share a 2pi world period. Wrapped origin
    // preserves phase across window rebases without accumulating float drift.
    float3 worldPoint = p + float3(u.options.x, u.options.w, 0);
    float3 phase = k * worldPoint + float3(.19f, -.13f, .11f) * u.step.y;
    float3 sine = sin(phase), cosine = cos(phase);
    v += .025f * u.options.y * float3(sine.x * cosine.y * cosine.z,
                                     -cosine.x * sine.y * cosine.z, 0);
    float3 phase2 = 37.f * worldPoint + float3(-.07f, .09f, .05f) * u.step.y + 1.73f;
    float3 sine2 = sin(phase2), cosine2 = cos(phase2);
    v += .015f * u.options.y * float3(0, sine2.y * cosine2.z * cosine2.x,
                                     -cosine2.y * sine2.z * cosine2.x);
    // The explicit positivity guarantee assumes |v_i| <= .35 m/s. Clamping is
    // a numerical safety limit, not a claim of incompressible Navier-Stokes.
    velocity.write(float4(clamp(v, -.35f, .35f), 1), index);
}

static float3 cloudMinmod(float3 a, float3 b) {
    return select(float3(0), sign(a) * min(abs(a), abs(b)), a * b > 0);
}
static float3 cloudRead(texture3d<float, access::read> field, int3 position, uint3 size) {
    return field.read(uint3(clamp(position, int3(0), int3(size) - 1))).rgb;
}
static float3 cloudSlope(texture3d<float, access::read> field, int3 p, int3 axis, uint3 size) {
    float3 c = cloudRead(field, p, size);
    return cloudMinmod(c - cloudRead(field, p - axis, size), cloudRead(field, p + axis, size) - c);
}

// One unique expression for a face, used by each neighboring cell, so summed
// fluxes cancel. Minmod bounds outgoing reconstruction by 1.5*c; CFL plus
// diffusion stays < .5 at the supported grid/time step, preserving positivity.
static float3 cloudFlux(texture3d<float, access::read> field,
                        texture3d<float, access::read> velocity,
                        int3 left, int3 axis, uint axisIndex, uint3 size,
                        float spacing, float diffusion) {
    int3 right = left + axis;
    if (any(left < 0) || any(right >= int3(size))) return 0;
    float4 vLeft = velocity.read(uint3(left)), vRight = velocity.read(uint3(right));
    if (vLeft.w < .5f || vRight.w < .5f) return 0;
    float faceVelocity = .5f * (vLeft[axisIndex] + vRight[axisIndex]);
    float3 cLeft = field.read(uint3(left)).rgb, cRight = field.read(uint3(right)).rgb;
    float3 reconstructed = faceVelocity >= 0 ? cLeft + .5f * cloudSlope(field, left, axis, size)
                                             : cRight - .5f * cloudSlope(field, right, axis, size);
    return faceVelocity * reconstructed - diffusion / spacing * (cRight - cLeft);
}

kernel void cloudAdvect(constant CloudUniforms &u [[buffer(0)]],
                        texture3d<float, access::read> input [[texture(0)]],
                        texture3d<float, access::write> output [[texture(1)]],
                        texture3d<float, access::read> velocity [[texture(2)]],
                        uint3 index [[thread_position_in_grid]]) {
    uint3 size = cloudSize(input);
    if (any(index >= size)) return;
    float4 original = input.read(index);
    if (u.step.x <= 0) { output.write(original, index); return; }
    float dt = min(.008f, u.step.x);
    float diffusion = clamp(u.options.z, 0.f, 1e-5f);
    float3 cell = cloudCell(u, size), concentration = original.rgb;
    int3 p = int3(index);
    for (uint axisIndex = 0; axisIndex < 3; ++axisIndex) {
        int3 axis = 0; axis[axisIndex] = 1;
        float3 outgoing = cloudFlux(input, velocity, p, axis, axisIndex, size, cell[axisIndex], diffusion);
        float3 incoming = cloudFlux(input, velocity, p - axis, axis, axisIndex, size, cell[axisIndex], diffusion);
        concentration -= dt / cell[axisIndex] * (outgoing - incoming);
    }
    float fade = exp(-.08664339757f * dt * max(0.f, u.step.z)); // ln(2)/8
    // No nonnegative clamp: loss of positivity would expose a violated CFL
    // invariant in the GPU checks instead of silently manufacturing mass.
    output.write(float4(concentration * fade, 0), index);
}
