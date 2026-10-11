#include <metal_stdlib>
using namespace metal;

// Authored living material, evaluated in world metres. It changes neither the
// hydraulic bed nor the water's energy budget: reflectance and emitted radiance
// remain separate inputs to the renderer's illumination and extinction paths.
struct BedMaterial {
    float3 albedo;
    float3 emission;
    float2 slope;
    float roughness;
};

static uint bedLifeMix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
static float bedLifeHash(int2 p, uint salt = 0u) {
    uint h = bedLifeMix(as_type<uint>(p.x) ^ bedLifeMix(as_type<uint>(p.y) + 0x9e3779b9u) ^ salt);
    return float(h >> 8) * (1.f / 16777216.f);
}

// Quintic value noise and its analytic gradient; normals use the same relief
// as the material, rather than an unrelated sparkling normal texture.
static float3 bedLifeNoise(float2 p) {
    int2 cell = int2(floor(p));
    float2 f = fract(p);
    float2 w = f * f * f * (f * (f * 6.f - 15.f) + 10.f);
    float2 dw = 30.f * f * f * (f - 1.f) * (f - 1.f);
    float a = bedLifeHash(cell), b = bedLifeHash(cell + int2(1, 0));
    float c = bedLifeHash(cell + int2(0, 1)), d = bedLifeHash(cell + int2(1, 1));
    float lo = mix(a, b, w.x), hi = mix(c, d, w.x);
    return float3(mix(lo, hi, w.y),
                  mix(b - a, d - c, w.y) * dw.x, (hi - lo) * dw.y);
}

static float3 bedLifeSpectrum(float phase) {
    // Saturated related pigments: no white stops and no isolated hot red.
    const float3 palette[6] = {
        float3(.025f, .76f, 1.f), float3(.28f, .055f, 1.f),
        float3(.88f, .045f, .58f), float3(.34f, .88f, .035f),
        float3(.025f, .90f, .42f), float3(.12f, .29f, 1.f)
    };
    float x = fract(phase) * 6.f;
    uint i = min(5u, uint(x));
    float f = fract(x); f = f * f * (3.f - 2.f * f);
    return mix(palette[i], palette[(i + 1u) % 6u], f);
}

BedMaterial animatedBed(float2 world, float physicalTime, float4 existingMaterial,
                        float pixelFootprint) {
    float neutral = clamp(existingMaterial.x, .02f, .92f);
    BedMaterial result = { float3(neutral), max(existingMaterial.yzw, 0.f), float2(0), .7f };
    if (!all(isfinite(world)) || !isfinite(physicalTime)) return result;
    float footprint = isfinite(pixelFootprint) ? clamp(pixelFootprint, 0.f, 1.f) : .01f;
    float time = physicalTime * .24f;

    // Three explicitly filtered octaves: finer relief disappears continuously
    // when a refracted ray footprint can no longer resolve it.
    const float frequencies[3] = { 4.1f, 8.7f, 18.2f };
    const float weights[3] = { .55f, .29f, .16f };
    float mineral = .5f;
    float2 gradient = 0;
    for (uint octave = 0; octave < 3; ++octave) {
        float frequency = frequencies[octave];
        float weight = weights[octave] * (1.f - smoothstep(.18f, .48f, footprint * frequency));
        float3 noise = bedLifeNoise(world * frequency + float2(3.17f, -8.41f) * float(octave));
        mineral += weight * (noise.x - .5f);
        gradient += weight * frequency * noise.yz;
    }
    float3 mineralColor = bedLifeSpectrum(mineral * .55f + .57f);
    result.albedo = neutral * (.35f + .95f * mineral) * mix(float3(1), mineralColor, .45f);
    result.slope = .012f * gradient;
    result.roughness = .42f + .43f * mineral;

    float resolve = 1.f - smoothstep(.025f, .11f, footprint);
    float veinAA = max(.008f, footprint * length(gradient));
    float veinDistance = abs(mineral - (.51f + .024f * sin(time * .31f)));
    float veins = (1.f - smoothstep(.013f, .033f + veinAA, veinDistance)) * resolve;
    float3 veinColor = bedLifeSpectrum(mineral * 1.8f + .21f + .025f * sin(time * .17f));
    result.emission *= .72f;
    result.emission += .17f * veins * veinColor;

    // Each specimen has compact support wholly inside its world cell. Crawling
    // never swaps cell identities, crosses a seam, or follows the camera. Empty
    // regions leave room for dark minerals and the original procedural bed.
    const float cellSize = .72f;
    float2 address = world / cellSize;
    int2 cell = int2(floor(address));
    float seed = bedLifeHash(cell, 0x3962a51fu);
    float presence = smoothstep(.43f, .69f, seed) * resolve;
    if (presence > .001f) {
        float phase = bedLifeHash(cell, 0x716bb12du) * 6.28318530718f;
        float variety = bedLifeHash(cell, 0xab5714d3u);
        float2 jitter = float2(bedLifeHash(cell, 0x218c91e9u), variety) - .5f;
        float2 center = .075f * jitter + .017f * float2(sin(time * .43f + phase), cos(time * .37f + phase));
        float2 q = fract(address) - .5f - center;
        float radius = max(length(q), .0001f);
        float angle = atan2(q.y, q.x);
        float aa = max(.0012f, footprint / cellSize);
        float support = 1.f - smoothstep(.32f, .42f, radius);
        float boundary = .205f + .028f * sin(3.f * angle + time + phase)
                                   + .019f * sin(5.f * angle - .67f * time + phase);
        float boundaryDerivative = .084f * cos(3.f * angle + time + phase)
                                   + .095f * cos(5.f * angle - .67f * time + phase);
        float body = 1.f - smoothstep(boundary - aa, boundary + aa, radius);
        float rim = 1.f - smoothstep(.007f, .021f + aa, abs(radius - boundary));
        float winding = 48.f * radius - 2.f * angle + 2.7f * mineral - time * .71f + phase;
        float filamentAA = max(.04f, aa * (48.f + 2.f / max(radius, .04f)));
        float filaments = (1.f - smoothstep(.12f, .12f + filamentAA, abs(sin(winding))))
            * body * smoothstep(.025f, .06f, radius);

        float curve = .070f * sin(10.f * q.x + .74f * time + phase)
                    + .027f * sin(24.f * q.x - .39f * time + phase);
        float ribbon = (1.f - smoothstep(.010f, .021f + aa, abs(q.y - curve)))
                     * (1.f - smoothstep(.25f, .34f, abs(q.x)));
        float ribbonSkeleton = ribbon * (.45f + .55f * sin(43.f * q.x - time + phase) * sin(43.f * q.x - time + phase));
        float ribbonKind = smoothstep(.48f, .65f, variety);
        float tissue = mix(body, ribbon, ribbonKind);
        float luminous = mix(.50f * rim + .54f * filaments, ribbonSkeleton, ribbonKind);
        // Paired luminous vacuoles suggest living forms without filling the bed
        // with icons. They remain embedded in the same material silhouette.
        float eyeA = 1.f - smoothstep(.007f, .014f + aa, length(q - float2(-.048f, .043f)));
        float eyeB = 1.f - smoothstep(.007f, .014f + aa, length(q - float2(.039f, .055f)));
        luminous += .65f * max(eyeA, eyeB) * body * (1.f - ribbonKind);
        float activity = presence * support;
        float hue = bedLifeHash(cell, 0xf369175bu);
        float3 organismColor = bedLifeSpectrum(hue + .045f * sin(time * .23f + phase));
        result.albedo = mix(result.albedo, .035f + .38f * organismColor,
                            .85f * activity * tissue);
        result.emission += .37f * activity * luminous * organismColor;

        // A 4 mm soft body relief. This is the analytic derivative of
        // h=.004*(1-(r/R(theta))²)² inside the amoeba, tapered away for ribbons.
        float relativeRadius = radius / boundary;
        if (relativeRadius < 1.f) {
            float2 angularGradient = float2(-q.y, q.x) / (radius * radius);
            float2 radiusGradient = q / (radius * boundary)
                - radius * boundaryDerivative / (boundary * boundary) * angularGradient;
            result.slope += (-.016f * relativeRadius * (1.f - relativeRadius * relativeRadius)
                / cellSize) * radiusGradient * presence * (1.f - ribbonKind);
        }
    }
    result.albedo = clamp(result.albedo, .02f, .92f);
    // A shared scalar cap retains the spectrum, avoiding a white RGB plateau.
    float maximum = max(result.emission.r, max(result.emission.g, result.emission.b));
    result.emission *= min(1.f, .55f / max(maximum, 1e-7f));
    result.slope = clamp(result.slope, -.32f, .32f);
    result.roughness = clamp(result.roughness, .30f, .95f);
    return result;
}
