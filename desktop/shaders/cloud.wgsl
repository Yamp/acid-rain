// Conservative three-dimensional CMY tracer under a prescribed wake closure.
// This is not a three-dimensional Navier-Stokes solver. Concentration has units
// of pigment mass / m^3; integrated mass is sum(concentration * voxel volume).
// Compose after the shared sample_clamp2 helpers (manual or hardware filtering).
//
// Group 0 binding contract (one explicit layout for all four entry points):
// 0: 80-byte uniform; 1: complete read-only CloudDrop storage; 2: complete RW
// vec4<f32> normalization storage; 3: sampled FP32 input volume; 4: write-only
// rgba32float output volume; 5: sampled FP32 velocity; 6: flow; 7: water.
// cloudVelocity writes binding 4: bind the velocity destination there and bind
// another read-only texture at unused binding 5, avoiding read/write aliasing.
// Every dispatch needs distinct uniform bytes until the submission completes.
struct CloudDispatchUniforms {
    step: vec4<f32>,       // dt <= .008, physical time, bleach rate, drop count
    rebase: vec4<f32>,     // exact integer XY voxel shift, clear, inject
    domain: vec4<f32>,     // XY extent in m, lower Z, vertical extent in m
    options: vec4<f32>,    // origin X mod 2pi, wake strength, diffusion, origin Y mod 2pi
    dispatchData: vec4<u32>, // sourceBase, normalizationBase, reserved, reserved
}

struct CloudDrop {
    position: vec4<f32>,
    pigment: vec4<f32>,
    dynamics: vec4<f32>,
    identity: vec4<u32>,
}

@group(0) @binding(0) var<uniform> cloudUniforms: CloudDispatchUniforms;
@group(0) @binding(1) var<storage, read> cloudDrops: array<CloudDrop>;
@group(0) @binding(2) var<storage, read_write> cloudNormalization: array<vec4<f32>>;
@group(0) @binding(3) var cloudInput: texture_3d<f32>;
@group(0) @binding(4) var cloudOutput: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var cloudVelocityInput: texture_3d<f32>;
@group(0) @binding(6) var cloudFlow: texture_2d<f32>;
@group(0) @binding(7) var cloudWater: texture_2d<f32>;

fn cloudDropCount() -> u32 {
    let available = arrayLength(&cloudDrops);
    return min(u32(max(cloudUniforms.step.w, 0.0)),
               available - min(cloudUniforms.dispatchData.x, available));
}

fn cloudSourceCount() -> u32 {
    let available = arrayLength(&cloudNormalization);
    return min(cloudDropCount(), available - min(cloudUniforms.dispatchData.y, available));
}

fn cloudCell(size: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(cloudUniforms.domain.xy, cloudUniforms.domain.w) / vec3<f32>(size);
}

fn cloudPosition(index: vec3<u32>, size: vec3<u32>) -> vec3<f32> {
    return (vec3<f32>(index) + vec3<f32>(0.5)) * cloudCell(size)
        + vec3<f32>(0.0, 0.0, cloudUniforms.domain.z);
}

fn cloudField(field: texture_2d<f32>, position: vec2<f32>) -> vec4<f32> {
    let uv = position / cloudUniforms.domain.xy + vec2<f32>(0.5) / vec2<f32>(textureDimensions(field));
    return sample_clamp2(field, uv);
}

fn cloudFluid(position: vec3<f32>) -> bool {
    let surface = cloudField(cloudWater, position.xy).x;
    let depth = max(0.01, cloudField(cloudFlow, position.xy).w);
    return position.z <= surface && position.z >= surface - depth;
}

fn cloudSourceCenter(drop: CloudDrop) -> vec3<f32> {
    return drop.position.xyz - vec3<f32>(0.0, 0.0, 0.035 + 1.5 * drop.position.w);
}

fn cloudSigma(drop: CloudDrop) -> vec3<f32> {
    return max(vec3<f32>(drop.dynamics.xx, drop.dynamics.y), vec3<f32>(0.006));
}

fn cloudGaussian(position: vec3<f32>, drop: CloudDrop) -> f32 {
    let q = (position - cloudSourceCenter(drop)) / cloudSigma(drop);
    if any(abs(q) > vec3<f32>(4.0)) {
        return 0.0;
    }
    return exp(-0.5 * dot(q, q));
}

// A bounded reduction per new impact. Injection uses the identical voxel
// support and fluid mask, including sources clipped against the domain edges.
@compute @workgroup_size(1)
fn cloudNormalize(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= cloudSourceCount() {
        return;
    }
    let drop = cloudDrops[cloudUniforms.dispatchData.x + index];
    let size = textureDimensions(cloudInput);
    let cell = cloudCell(size);
    let center = cloudSourceCenter(drop) - vec3<f32>(0.0, 0.0, cloudUniforms.domain.z);
    let support = 4.0 * cloudSigma(drop);
    let lo = max(vec3<i32>(0), vec3<i32>(ceil((center - support) / cell - vec3<f32>(0.5))));
    let hi = min(vec3<i32>(size) - vec3<i32>(1),
                 vec3<i32>(floor((center + support) / cell - vec3<f32>(0.5))));
    var integral = 0.0;
    if drop.identity.z == 1u {
        for (var z = lo.z; z <= hi.z; z += 1) {
            for (var y = lo.y; y <= hi.y; y += 1) {
                for (var x = lo.x; x <= hi.x; x += 1) {
                    let position = cloudPosition(vec3<u32>(u32(x), u32(y), u32(z)), size);
                    if cloudFluid(position) {
                        integral += cloudGaussian(position, drop);
                    }
                }
            }
        }
    }
    integral *= cell.x * cell.y * cell.z;
    var inverse = 0.0;
    if integral > 1e-20 {
        inverse = 1.0 / integral;
    }
    cloudNormalization[cloudUniforms.dispatchData.y + index] = vec4<f32>(inverse, integral, 0.0, 0.0);
}

// Exact integer world-window rebase; newly exposed voxels start empty. No
// source deduplication occurs here: the host submits each impact ID once.
@compute @workgroup_size(4, 4, 4)
fn cloudPrepare(@builtin(global_invocation_id) index: vec3<u32>) {
    let size = textureDimensions(cloudInput);
    if any(index >= size) {
        return;
    }
    let oldIndex = vec3<i32>(index) + vec3<i32>(vec2<i32>(round(cloudUniforms.rebase.xy)), 0);
    var pigment = vec3<f32>(0.0);
    if cloudUniforms.rebase.z < 0.5 && all(oldIndex >= vec3<i32>(0)) && all(oldIndex < vec3<i32>(size)) {
        pigment = textureLoad(cloudInput, oldIndex, 0).rgb;
    }
    if cloudUniforms.rebase.w > 0.5 {
        let position = cloudPosition(index, size);
        if cloudFluid(position) {
            let count = cloudSourceCount();
            for (var i = 0u; i < count; i += 1u) {
                let drop = cloudDrops[cloudUniforms.dispatchData.x + i];
                if drop.identity.z != 1u {
                    continue;
                }
                let source = cloudGaussian(position, drop)
                    * cloudNormalization[cloudUniforms.dispatchData.y + i].x
                    * max(0.0, drop.pigment.w);
                pigment += max(vec3<f32>(0.0), drop.pigment.rgb) * source;
            }
        }
    }
    textureStore(cloudOutput, vec3<i32>(index), vec4<f32>(pigment, 0.0));
}

// Analytic axisymmetric ring from psi=A*r^2 exp(-r^2/a^2-(z-zc)^2/b^2).
// Analytically divergence-free before the fluid boundary and CFL safety limits.
fn cloudRing(position: vec3<f32>, drop: CloudDrop, time: f32) -> vec3<f32> {
    let age = time - drop.dynamics.z;
    if age < 0.0 || age > 2.5 {
        return vec3<f32>(0.0);
    }
    let ageRoot = sqrt(age + 0.025);
    let a = max(0.07, 1.4 * drop.dynamics.x) + 0.03 * ageRoot;
    let b = max(0.045, drop.dynamics.y) + 0.018 * ageRoot;
    let speed = clamp(drop.dynamics.w, 0.0, 0.45);
    var center = cloudSourceCenter(drop);
    center.z -= 0.30 * speed * age / (1.0 + 0.28 * age);
    let delta = position - center;
    let radial2 = dot(delta.xy, delta.xy);
    let q = radial2 / (a * a) + delta.z * delta.z / (b * b);
    if q > 18.0 {
        return vec3<f32>(0.0);
    }
    let strength = -0.5 * speed * exp(-age * 0.30);
    let envelope = exp(-q);
    let radial = (2.0 * strength * delta.z / (b * b) * envelope) * delta.xy;
    let vertical = 2.0 * strength * (1.0 - radial2 / (a * a)) * envelope;
    return vec3<f32>(radial, vertical);
}

@compute @workgroup_size(4, 4, 4)
fn cloudVelocity(@builtin(global_invocation_id) index: vec3<u32>) {
    let size = textureDimensions(cloudInput);
    if any(index >= size) {
        return;
    }
    let position = cloudPosition(index, size);
    if !cloudFluid(position) {
        textureStore(cloudOutput, vec3<i32>(index), vec4<f32>(0.0));
        return;
    }
    let ambient = cloudField(cloudFlow, position.xy);
    var velocity = vec3<f32>(ambient.xy, 0.0);
    let count = min(24u, cloudDropCount());
    for (var i = 0u; i < count; i += 1u) {
        let drop = cloudDrops[cloudUniforms.dispatchData.x + i];
        if drop.identity.z == 1u {
            velocity += cloudUniforms.options.y * cloudRing(position, drop, cloudUniforms.step.y);
        }
    }
    // Smooth coherent strain, not random voxel noise. Integer frequencies make
    // the wrapped world origin continuous across both rebases and the 2pi wrap.
    let worldPoint = position + vec3<f32>(cloudUniforms.options.x, cloudUniforms.options.w, 0.0);
    let phase = 24.0 * worldPoint + vec3<f32>(0.19, -0.13, 0.11) * cloudUniforms.step.y;
    let sine = sin(phase);
    let cosine = cos(phase);
    velocity += 0.025 * cloudUniforms.options.y
        * vec3<f32>(sine.x * cosine.y * cosine.z, -cosine.x * sine.y * cosine.z, 0.0);
    let phase2 = 37.0 * worldPoint + vec3<f32>(-0.07, 0.09, 0.05) * cloudUniforms.step.y + vec3<f32>(1.73);
    let sine2 = sin(phase2);
    let cosine2 = cos(phase2);
    velocity += 0.015 * cloudUniforms.options.y
        * vec3<f32>(0.0, sine2.y * cosine2.z * cosine2.x, -cosine2.y * sine2.z * cosine2.x);
    // Componentwise .35m/s limit is required by the positivity CFL bound.
    textureStore(cloudOutput, vec3<i32>(index), vec4<f32>(clamp(velocity, vec3<f32>(-0.35), vec3<f32>(0.35)), 1.0));
}

fn cloudMinmod(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), sign(a) * min(abs(a), abs(b)), a * b > vec3<f32>(0.0));
}

fn cloudRead(position: vec3<i32>, size: vec3<u32>) -> vec3<f32> {
    return textureLoad(cloudInput, clamp(position, vec3<i32>(0), vec3<i32>(size) - vec3<i32>(1)), 0).rgb;
}

fn cloudSlope(position: vec3<i32>, axis: vec3<i32>, size: vec3<u32>) -> vec3<f32> {
    let concentration = cloudRead(position, size);
    return cloudMinmod(concentration - cloudRead(position - axis, size),
                       cloudRead(position + axis, size) - concentration);
}

// One expression for each shared face: opposite contributions cancel exactly
// apart from FP32 accumulation. No flux through the volume/fluid boundaries;
// world streaming exports concentration only when the window rebases.
fn cloudFlux(left: vec3<i32>, axis: vec3<i32>, axisIndex: u32, size: vec3<u32>,
             spacing: f32, diffusion: f32) -> vec3<f32> {
    let right = left + axis;
    if any(left < vec3<i32>(0)) || any(right >= vec3<i32>(size)) {
        return vec3<f32>(0.0);
    }
    let vLeft = textureLoad(cloudVelocityInput, left, 0);
    let vRight = textureLoad(cloudVelocityInput, right, 0);
    if vLeft.w < 0.5 || vRight.w < 0.5 {
        return vec3<f32>(0.0);
    }
    let faceVelocity = 0.5 * (vLeft[axisIndex] + vRight[axisIndex]);
    let cLeft = textureLoad(cloudInput, left, 0).rgb;
    let cRight = textureLoad(cloudInput, right, 0).rgb;
    var reconstructed: vec3<f32>;
    if faceVelocity >= 0.0 {
        reconstructed = cLeft + 0.5 * cloudSlope(left, axis, size);
    } else {
        reconstructed = cRight - 0.5 * cloudSlope(right, axis, size);
    }
    return faceVelocity * reconstructed - diffusion / spacing * (cRight - cLeft);
}

@compute @workgroup_size(4, 4, 4)
fn cloudAdvect(@builtin(global_invocation_id) index: vec3<u32>) {
    let size = textureDimensions(cloudInput);
    if any(index >= size) {
        return;
    }
    let original = textureLoad(cloudInput, vec3<i32>(index), 0);
    if cloudUniforms.step.x <= 0.0 {
        textureStore(cloudOutput, vec3<i32>(index), original);
        return;
    }
    let dt = min(0.008, cloudUniforms.step.x);
    let diffusion = clamp(cloudUniforms.options.z, 0.0, 1e-5);
    let cell = cloudCell(size);
    var concentration = original.rgb;
    let position = vec3<i32>(index);
    for (var axisIndex = 0u; axisIndex < 3u; axisIndex += 1u) {
        var axis = vec3<i32>(0);
        axis[axisIndex] = 1;
        let outgoing = cloudFlux(position, axis, axisIndex, size, cell[axisIndex], diffusion);
        let incoming = cloudFlux(position - axis, axis, axisIndex, size, cell[axisIndex], diffusion);
        concentration -= dt / cell[axisIndex] * (outgoing - incoming);
    }
    let fade = exp(-0.08664339757 * dt * max(0.0, cloudUniforms.step.z)); // ln(2)/8
    // Never clamp concentrations nonnegative: a violated CFL must be visible
    // to the GPU conservation/positivity checks instead of manufacturing mass.
    textureStore(cloudOutput, vec3<i32>(index), vec4<f32>(concentration * fade, 0.0));
}
