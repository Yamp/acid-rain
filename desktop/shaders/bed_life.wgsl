// Authored living material in world metres, separate from the hydraulic bed.
// Requires finite1/finite2; this module has no resource bindings or entry points.
struct BedMaterial {
    albedo: vec3f,
    emission: vec3f,
    slope: vec2f,
    roughness: f32,
}

fn bedLifeMix(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn bedLifeHash(p: vec2i, salt: u32) -> f32 {
    let h = bedLifeMix(bitcast<u32>(p.x) ^ bedLifeMix(bitcast<u32>(p.y) + 0x9e3779b9u) ^ salt);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Quintic value noise and the analytic gradient of the same relief field.
fn bedLifeNoise(p: vec2f) -> vec3f {
    let cell = vec2i(floor(p));
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - vec2f(15.0)) + vec2f(10.0));
    let dw = 30.0 * f * f * (f - vec2f(1.0)) * (f - vec2f(1.0));
    let a = bedLifeHash(cell, 0u);
    let b = bedLifeHash(cell + vec2i(1, 0), 0u);
    let c = bedLifeHash(cell + vec2i(0, 1), 0u);
    let d = bedLifeHash(cell + vec2i(1, 1), 0u);
    let lo = mix(a, b, w.x);
    let hi = mix(c, d, w.x);
    return vec3f(mix(lo, hi, w.y), mix(b - a, d - c, w.y) * dw.x, (hi - lo) * dw.y);
}

fn bedLifeSpectrum(phase: f32) -> vec3f {
    let palette = array<vec3f, 6>(
        vec3f(0.025, 0.76, 1.0), vec3f(0.28, 0.055, 1.0),
        vec3f(0.88, 0.045, 0.58), vec3f(0.34, 0.88, 0.035),
        vec3f(0.025, 0.90, 0.42), vec3f(0.12, 0.29, 1.0));
    let x = fract(phase) * 6.0;
    let i = min(5u, u32(x));
    let fraction = fract(x);
    let blend = fraction * fraction * (3.0 - 2.0 * fraction);
    return mix(palette[i], palette[(i + 1u) % 6u], blend);
}

fn animatedBed(world: vec2f, physicalTime: f32, existingMaterial: vec4f,
               pixelFootprint: f32) -> BedMaterial {
    let neutral = clamp(existingMaterial.x, 0.02, 0.92);
    var result = BedMaterial(vec3f(neutral), max(existingMaterial.yzw, vec3f(0.0)), vec2f(0.0), 0.7);
    if !finite2(world) || !finite1(physicalTime) { return result; }
    var footprint = 0.01;
    if finite1(pixelFootprint) { footprint = clamp(pixelFootprint, 0.0, 1.0); }
    let time = physicalTime * 0.24;
    let frequencies = array<f32, 3>(4.1, 8.7, 18.2);
    let weights = array<f32, 3>(0.55, 0.29, 0.16);
    var mineral = 0.5;
    var gradient = vec2f(0.0);
    for (var octave = 0u; octave < 3u; octave += 1u) {
        let frequency = frequencies[octave];
        let weight = weights[octave] * (1.0 - smoothstep(0.18, 0.48, footprint * frequency));
        let noise = bedLifeNoise(world * frequency + vec2f(3.17, -8.41) * f32(octave));
        mineral += weight * (noise.x - 0.5);
        gradient += weight * frequency * noise.yz;
    }
    let mineralColor = bedLifeSpectrum(mineral * 0.55 + 0.57);
    result.albedo = neutral * (0.35 + 0.95 * mineral) * mix(vec3f(1.0), mineralColor, 0.45);
    result.slope = 0.012 * gradient;
    result.roughness = 0.42 + 0.43 * mineral;
    let resolve = 1.0 - smoothstep(0.025, 0.11, footprint);
    let veinAA = max(0.008, footprint * length(gradient));
    let veinDistance = abs(mineral - (0.51 + 0.024 * sin(time * 0.31)));
    let veins = (1.0 - smoothstep(0.013, 0.033 + veinAA, veinDistance)) * resolve;
    let veinColor = bedLifeSpectrum(mineral * 1.8 + 0.21 + 0.025 * sin(time * 0.17));
    result.emission *= 0.72;
    result.emission += 0.17 * veins * veinColor;

    // Compact supports stay inside their world cell as the specimens crawl.
    let cellSize = 0.72;
    let address = world / cellSize;
    let cell = vec2i(floor(address));
    let seed = bedLifeHash(cell, 0x3962a51fu);
    let presence = smoothstep(0.43, 0.69, seed) * resolve;
    if presence > 0.001 {
        let phase = bedLifeHash(cell, 0x716bb12du) * 6.28318530718;
        let variety = bedLifeHash(cell, 0xab5714d3u);
        let jitter = vec2f(bedLifeHash(cell, 0x218c91e9u), variety) - vec2f(0.5);
        let center = 0.075 * jitter + 0.017 * vec2f(sin(time * 0.43 + phase), cos(time * 0.37 + phase));
        let q = fract(address) - vec2f(0.5) - center;
        let radius = max(length(q), 0.0001);
        let angle = atan2(q.y, q.x);
        let aa = max(0.0012, footprint / cellSize);
        let support = 1.0 - smoothstep(0.32, 0.42, radius);
        let boundary = 0.205 + 0.028 * sin(3.0 * angle + time + phase)
            + 0.019 * sin(5.0 * angle - 0.67 * time + phase);
        let boundaryDerivative = 0.084 * cos(3.0 * angle + time + phase)
            + 0.095 * cos(5.0 * angle - 0.67 * time + phase);
        let body = 1.0 - smoothstep(boundary - aa, boundary + aa, radius);
        let rim = 1.0 - smoothstep(0.007, 0.021 + aa, abs(radius - boundary));
        let winding = 48.0 * radius - 2.0 * angle + 2.7 * mineral - time * 0.71 + phase;
        let filamentAA = max(0.04, aa * (48.0 + 2.0 / max(radius, 0.04)));
        let filaments = (1.0 - smoothstep(0.12, 0.12 + filamentAA, abs(sin(winding))))
            * body * smoothstep(0.025, 0.06, radius);
        let curve = 0.070 * sin(10.0 * q.x + 0.74 * time + phase)
            + 0.027 * sin(24.0 * q.x - 0.39 * time + phase);
        let ribbon = (1.0 - smoothstep(0.010, 0.021 + aa, abs(q.y - curve)))
            * (1.0 - smoothstep(0.25, 0.34, abs(q.x)));
        let ribbonSkeleton = ribbon * (0.45 + 0.55 * sin(43.0 * q.x - time + phase) * sin(43.0 * q.x - time + phase));
        let ribbonKind = smoothstep(0.48, 0.65, variety);
        let tissue = mix(body, ribbon, ribbonKind);
        var luminous = mix(0.50 * rim + 0.54 * filaments, ribbonSkeleton, ribbonKind);
        let eyeA = 1.0 - smoothstep(0.007, 0.014 + aa, length(q - vec2f(-0.048, 0.043)));
        let eyeB = 1.0 - smoothstep(0.007, 0.014 + aa, length(q - vec2f(0.039, 0.055)));
        luminous += 0.65 * max(eyeA, eyeB) * body * (1.0 - ribbonKind);
        let activity = presence * support;
        let hue = bedLifeHash(cell, 0xf369175bu);
        let organismColor = bedLifeSpectrum(hue + 0.045 * sin(time * 0.23 + phase));
        result.albedo = mix(result.albedo, vec3f(0.035) + 0.38 * organismColor,
                            0.85 * activity * tissue);
        result.emission += 0.37 * activity * luminous * organismColor;
        let relativeRadius = radius / boundary;
        if relativeRadius < 1.0 {
            let angularGradient = vec2f(-q.y, q.x) / (radius * radius);
            let radiusGradient = q / (radius * boundary)
                - radius * boundaryDerivative / (boundary * boundary) * angularGradient;
            result.slope += (-0.016 * relativeRadius * (1.0 - relativeRadius * relativeRadius)
                / cellSize) * radiusGradient * presence * (1.0 - ribbonKind);
        }
    }
    result.albedo = clamp(result.albedo, vec3f(0.02), vec3f(0.92));
    let maximum = max(result.emission.r, max(result.emission.g, result.emission.b));
    result.emission *= min(1.0, 0.55 / max(maximum, 1e-7));
    result.slope = clamp(result.slope, vec2f(-0.32), vec2f(0.32));
    result.roughness = clamp(result.roughness, 0.30, 0.95);
    return result;
}
