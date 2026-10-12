#include <metal_stdlib>
using namespace metal;

// Keep this layout identical to AcidRainCore.h (13 aligned float4s).
struct Uniforms {
    float4 viewport, time, camera, optics, gain, light, world, post;
    float4 eye, forward, right, up, illumination;
};
struct VertexOut { float4 position [[position]]; float2 uv; };
constant float PI = 3.14159265358979323846f;
constant float3 LUMA = float3(.2126f, .7152f, .0722f);
constant float3 IOR = float3(1.331f, 1.333f, 1.337f);
constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler skySampler(coord::normalized, s_address::repeat,
                             t_address::clamp_to_edge, filter::linear);
constexpr sampler mirrorSampler(coord::normalized, address::mirrored_repeat, filter::linear);

vertex VertexOut waterVertex(uint id [[vertex_id]]) {
    float2 p = float2(id == 1 ? 3.f : -1.f, id == 2 ? 3.f : -1.f);
    return { float4(p, 0, 1), float2(p.x * .5f + .5f, .5f - p.y * .5f) };
}

// Core fields live at lattice vertices i/N, unlike screen/material texel centres.
float4 field(texture2d<float> tex, float2 metres) {
    float2 n = float2(tex.get_width(), tex.get_height());
    return tex.sample(linearClamp, metres * .5f + .5f / n, level(0));
}
// Positive cubic B-spline reconstruction: four bilinear taps represent the
// separable 4x4 kernel. Partition of unity preserves radiance and affine slopes;
// continuous first derivatives remove magnified environment texel bends.
float4 opticalCubicAxis(float f) {
    float a = 1-f, f2 = f*f, f3 = f2*f;
    float4 w = float4(a*a*a, 3*f3-6*f2+4, -3*f3+3*f2+3*f+1, f3)/6;
    float2 g = float2(w.x+w.y,w.z+w.w);
    return float4(w.y/g.x-1, w.w/g.y+1, g);
}
float4 opticalCubicSample(texture2d<float> tex, sampler sampling, float2 uv) {
    float2 size = float2(tex.get_width(),tex.get_height());
    float2 p = uv*size-.5f, cell = floor(p);
    float4 x = opticalCubicAxis(p.x-cell.x), y = opticalCubicAxis(p.y-cell.y);
    float2 lo = (cell+float2(x.x,y.x)+.5f)/size;
    float2 hi = (cell+float2(x.y,y.y)+.5f)/size;
    return mix(mix(tex.sample(sampling,lo,level(0)),
                   tex.sample(sampling,float2(hi.x,lo.y),level(0)),x.w),
               mix(tex.sample(sampling,float2(lo.x,hi.y),level(0)),
                   tex.sample(sampling,hi,level(0)),x.w),y.w);
}
float3 surfaceNormal(texture2d<float> water, float2 metres) {
    float2 slope = waterSurface(water,metres).yz;
    return normalize(float3(-slope,1));
}
float3 environment(texture2d<float> sky, float3 direction) {
    direction = normalize(direction);
    // Longitude is arbitrary at either pole; avoid atan2(0,0) reaching floor.
    float longitude = dot(direction.xy,direction.xy) > 1e-12f ? atan2(direction.y,direction.x) : 0.f;
    float2 uv = float2(longitude / (2 * PI),
                       acos(clamp(direction.z, -1.f, 1.f)) / PI);
    return opticalCubicSample(sky, skySampler, uv).rgb;
}
float3 pigmentTau(float3 mass) {
    mass = max(mass, 0.f);
    return .85f * float3(dot(mass, float3(2, .30f, .25f)),
                         dot(mass, float3(.65f, 1.8f, .65f)),
                         dot(mass, float3(.30f, .45f, 1.8f)));
}
float3 fresnel(float cosine) {
    float c = clamp(abs(cosine), .00001f, 1.f);
    float3 t = sqrt(max(1.f - (1.f - c * c) / (IOR * IOR), 0.f));
    float3 p = (IOR * c - t) / (IOR * c + t);
    float3 s = (c - IOR * t) / (c + IOR * t);
    return .5f * (p * p + s * s);
}
float3 coating(float3 base, float cosine, float height, constant Uniforms &u) {
    float thickness = 475 + 220 * sin(u.time.x * .15f * .31f * .35f + height * 18);
    float ct = sqrt(max(0.f, 1 - (1 - cosine * cosine) / (1.46f * 1.46f)));
    float path = 2 * 1.46f * thickness * ct;
    float r01 = (1 - 1.46f) / (1 + 1.46f);
    float r12 = (1.46f - 1.333f) / (1.46f + 1.333f);
    float3 airy = 0;
    // Three broad-band quadrature nodes suppress unnaturally sharp spectral lines.
    for (int j = -1; j <= 1; ++j) {
        float3 wavelengths = float3(640, 545, 450) + float(j) * 38;
        float3 interference = 2 * r01 * r12 * cos(2 * PI * path / wavelengths);
        airy += (j == 0 ? .5f : .25f) *
            (r01 * r01 + r12 * r12 + interference) /
            (1 + r01 * r01 * r12 * r12 + interference);
    }
    return saturate(base + (1 - base) * (airy - .0204f) * min(u.light.w, 2.f) * .65f);
}

float heightResidual(texture2d<float> water, float3 eye, float3 ray, float distance) {
    float3 p = eye + ray * distance;
    return p.z - field(water, p.xy).x;
}
struct WaterIntersection { float3 point; float2 slope; float height, distance; };
WaterIntersection refineWaterIntersectionCached(texture2d<float> water,float3 eye,float3 ray,
    float3 seed,WaterSurfacePatch patch,float2 bracket) {
    float distance=max(0.f,dot(seed-eye,ray));
    float3 point=eye+ray*distance;
    float cellDistance=1.f/max(length(ray.xy*patch.scale),1e-5f);
    for (uint iteration=0;iteration<4;++iteration) {
        int2 cell=int2(floor(point.xy*patch.scale));
        if (any(cell!=patch.cell)) patch=waterSurfacePatch(water,point.xy);
        float3 surface=waterSurfaceValue(patch,point.xy);
        float residual=point.z-surface.x;
        float derivative=ray.z-dot(surface.yz,ray.xy);
        if (abs(residual)<1e-7f || abs(derivative)<1e-5f) return {point,surface.yz,surface.x,distance};
        float next=max(0.f,distance-clamp(residual/derivative,-cellDistance,cellDistance));
        if (next<bracket.x || next>bracket.y) return {point,surface.yz,surface.x,distance};
        distance=next;
        point=eye+ray*distance;
    }
    // The final Newton update may have changed cells. Return the derivative
    // at that exact point, reusing the coefficients only when still valid.
    if (any(int2(floor(point.xy*patch.scale))!=patch.cell)) patch=waterSurfacePatch(water,point.xy);
    float3 surface=waterSurfaceValue(patch,point.xy);
    return {point,surface.yz,surface.x,distance};
}
WaterIntersection refineWaterIntersectionData(texture2d<float> water,float3 eye,float3 ray,float3 seed) {
    float3 point=eye+ray*max(0.f,dot(seed-eye,ray));
    return refineWaterIntersectionCached(water,eye,ray,seed,waterSurfacePatch(water,point.xy),float2(0,1e30f));
}
WaterIntersection cubicWaterBracket(texture2d<float> water,float3 eye,float3 ray,float near,float far) {
    float3 left=eye+ray*near,right=eye+ray*far;
    float3 ls=waterSurface(water,left.xy),rs=waterSurface(water,right.xy);
    float a=left.z-ls.x,b=right.z-rs.x;
    WaterIntersection best=abs(a)<abs(b) ? WaterIntersection{left,ls.yz,ls.x,near} : WaterIntersection{right,rs.yz,rs.x,far};
    float error=min(abs(a),abs(b));
    // Callers provide finite fields and a verified cubic bracket. An invalid
    // bracket returns the best endpoint rather than inventing a sign change.
    if (!isfinite(a) || !isfinite(b) || (a>0)==(b>0)) return best;
    WaterSurfacePatch patch=waterSurfacePatch(water,left.xy);
    for (uint iteration=0;iteration<28;++iteration) {
        if (error<=1e-7f) break;
        float middle=.5f*(near+far);
        if (middle==near || middle==far) break;
        float3 point=eye+ray*middle;
        if (any(int2(floor(point.xy*patch.scale))!=patch.cell)) patch=waterSurfacePatch(water,point.xy);
        float3 surface=waterSurfaceValue(patch,point.xy);
        float residual=point.z-surface.x;
        if (abs(residual)<error) { best={point,surface.yz,surface.x,middle}; error=abs(residual); }
        if ((residual>0)==(a>0)) { near=middle; a=residual; }
        else { far=middle; b=residual; }
    }
    return best;
}
WaterIntersection acceptWaterIntersection(texture2d<float> water,float3 eye,float3 ray,
    WaterIntersection candidate,float near,float far) {
    // Retain the computed parameter: reconstructing it with dot(point-eye,ray)
    // can move a converged endpoint outside its bracket by a rounding ULP.
    if (abs(candidate.point.z-candidate.height)<=1e-6f && candidate.distance>=near && candidate.distance<=far) return candidate;
    return cubicWaterBracket(water,eye,ray,near,far);
}
float3 refineWaterIntersection(texture2d<float> water,float3 eye,float3 ray,float3 seed) {
    return refineWaterIntersectionData(water,eye,ray,seed).point;
}
WaterIntersection intersectWaterData(texture2d<float> water, float3 eye, float3 ray,
                                     constant Uniforms &u) {
    float near = max(0.f, (u.world.z + .00001f - eye.z) / ray.z);
    float far = (u.world.y - .00001f - eye.z) / ray.z;
    float2 bracket=float2(near,far);
    float a = heightResidual(water, eye, ray, near);
    float b = heightResidual(water, eye, ray, far);
    // Frustum fitting guarantees the primary bracket lies entirely in the patch.
    for (uint i = 0; i < 12; ++i) {
        float t = clamp(a / max(a - b, 1e-9f), .05f, .95f);
        float m = mix(near, far, t);
        float value = heightResidual(water, eye, ray, m);
        if (value > 0) { near = m; a = value; }
        else { far = m; b = value; }
    }
    float3 seed=eye+ray*mix(near,far,saturate(a/max(a-b,1e-9f)));
    float seedDistance=max(0.f,dot(seed-eye,ray));
    float3 point=eye+ray*seedDistance;
    WaterSurfacePatch patch=waterSurfacePatch(water,point.xy);
    float seedResidual=point.z-waterSurfaceValue(patch,point.xy).x;
    // A known inside point prevents Newton from jumping through this crest to
    // an unrelated later root. Preserve the original global bracket, not the
    // interval narrowed against the different bilinear seed surface.
    if (seedDistance>bracket.x && seedDistance<bracket.y) {
        if (seedResidual>0) bracket.x=seedDistance; else bracket.y=seedDistance;
    }
    WaterIntersection result=refineWaterIntersectionCached(water,eye,ray,seed,patch,bracket);
    return acceptWaterIntersection(water,eye,ray,result,bracket.x,bracket.y);
}
float3 intersectWater(texture2d<float> water,float3 eye,float3 ray,constant Uniforms &u) {
    return intersectWaterData(water,eye,ray,u).point;
}

float3 extinction(constant Uniforms &u) {
    return float3(1.2f, 1.6f, 1.2f) * u.optics.y + .015f;
}
float3 cloudIllumination(constant Uniforms &u) {
    // Optional observation lighting leaves the transport and optical mass alone.
    return u.illumination.w > .5f ? float3(1.f, .98f, 1.12f) : u.illumination.rgb;
}
float3 pigmentScattering(float3 concentration) {
    // Colored elastic scattering by suspended pigment, separate from absorption.
    float3 c = max(concentration, 0.f);
    return .32f * float3(dot(c, float3(.03f, .90f, 1.10f)),
                         dot(c, float3(.85f, .04f, .65f)),
                         dot(c, float3(1.10f, .50f, .04f)));
}
float3 cloudSample(texture3d<float> volume, float3 p) {
    float3 uv = float3(p.xy * .5f, (p.z + .8f) / .9f);
    if (any(uv < 0.f) || any(uv > 1.f)) return 0;
    return max(volume.sample(linearClamp, uv, level(0)).rgb, 0.f);
}
struct CloudOptics { float3 transmittance, radiance, column; };
CloudOptics cloudTransfer(texture3d<float> volume, float3 origin, float3 direction,
                          float distance, constant Uniforms &u, uint steps) {
    CloudOptics result = { float3(1), float3(0), float3(0) };
    steps = clamp(steps, 1u, 64u);
    float ds = max(distance, 0.f) / float(steps);
    // Midpoint quadrature with exact Beer integration inside each segment. This
    // is ambient single scattering, not a multiple-scattering/emission shortcut.
    for (uint i = 0; i < steps; ++i) {
        float3 c = cloudSample(volume, origin + direction * ((float(i) + .5f) * ds));
        float3 scattering = pigmentScattering(c) + .015f;
        float3 sigma = extinction(u) + pigmentTau(c) + pigmentScattering(c);
        float3 tau = sigma * ds;
        float3 segment = exp(-tau);
        float3 removed = select(1 - segment, tau * (1 - .5f * tau + tau * tau / 6), tau < .001f);
        result.radiance += result.transmittance * scattering / max(sigma, 1e-7f)
            * removed * cloudIllumination(u);
        result.transmittance *= segment;
        result.column += c * ds;
    }
    return result;
}
float3 bedRadiance(texture2d<float> water, texture2d<float> dye, texture2d<float> bed,
                   texture2d<float> lightField, texture3d<float> volume,
                   texture2d<float> sky, texture2d<float> caustics,
                   float3 point, float3 normal, float3 ray, constant Uniforms &u,
                   constant float4 &bedWorld, float footprint, bool primaryFootprint, uint cloudSteps) {
    float3 result = 0;
    float3 sigma = extinction(u);
    float3 centralRay = refract(ray, normal, 1.f / IOR.y);
    float4 centralHit = causticsBedHit(water, dye, point, centralRay);
    float2 centralLocation = centralHit.xy;
    float2 centralDisplacement = centralLocation - point.xy;
    centralLocation = point.xy + centralDisplacement * min(1.f, 1.5f /
        max(max(abs(centralDisplacement.x), abs(centralDisplacement.y)), 1e-6f));
    float3 hydraulicNormal = float3(0, 0, 1);
    if (primaryFootprint) {
        // Refraction can magnify a pixel far more than the water intersection.
        // Derivatives are evaluated only on the uniform primary path; secondary
        // bounces pass their conservative coarser footprint explicitly.
        footprint = max(footprint, max(length(dfdx(centralLocation)), length(dfdy(centralLocation))));
        float3 crossed = cross(dfdx(centralHit.xyz), dfdy(centralHit.xyz));
        if (dot(crossed, crossed) > 1e-20f) {
            hydraulicNormal = normalize(crossed);
            if (hydraulicNormal.z < 0) hydraulicNormal = -hydraulicNormal;
        }
    }
    // Invalid rays retain the finite source point returned by causticsBedHit
    // until the entire primary quad has evaluated its derivatives.
    if (centralHit.w <= 0) return 0;
    float centralLength = centralHit.w;
    float depth = max(.015f, point.z - centralHit.z);
    CloudOptics cloud = cloudTransfer(volume, point, centralRay, centralLength, u, cloudSteps);
    float3 columnExtinction = pigmentTau(cloud.column) + pigmentScattering(cloud.column);
    float4 base = bed.sample(linearClamp, (centralLocation + 1.5f) / 5.f, level(0));
    BedMaterial living = { float3(base.x), base.yzw, float2(0), .7f };
    bool livingEnabled = bedWorld.w > .5f && u.illumination.w < .5f;
    if (livingEnabled) {
        // The fine authored material shares the central ray footprint, like the
        // sub-voxel volume integral. The streamed bed keeps independent RGB rays.
        living = animatedBed(bedWorld.xy + centralLocation, bedWorld.z, base,
                             footprint / max(.2f, -centralRay.z));
    }
    float3 l = normalize(u.light.xyz);
    float3 downLight = refract(-l, float3(0, 0, 1), 1.f / IOR.y);
    float3 bedNormal = normalize(float3(hydraulicNormal.xy - living.slope * hydraulicNormal.z,
                                       hydraulicNormal.z));
    float lightCosine = max(.001f, -downLight.z);
    float bumpCosine = max(0.f, dot(bedNormal, -downLight));
    // Irradiance is tabulated per horizontal square metre. Convert it to the
    // unresolved material facet, whose relief does not alter the hydraulic bed.
    float facet = hydraulicNormal.z * bumpCosine / max(.001f, dot(hydraulicNormal, -downLight));
    float3 directSource = environment(sky, l) * (.18f / PI);
    for (uint c = 0; c < 3; ++c) {
        float3 transmitted = refract(ray, normal, 1.f / IOR[c]);
        // Back-facing facets cannot transmit to the local planar bed.
        if (transmitted.z >= -.001f) continue;
        float4 bedHit = c == 1 ? centralHit : causticsBedHit(water, dye, point, transmitted);
        if (bedHit.w <= 0) continue;
        float2 displacement = bedHit.xy - point.xy;
        float scale = min(1.f, 1.5f / max(max(abs(displacement.x), abs(displacement.y)), 1e-6f));
        float2 location = point.xy + displacement * scale;
        float2 uv = (location + 1.5f) / 5.f;
        float4 material = bed.sample(linearClamp, uv, level(0));
        float3 incident = lightField.sample(linearClamp, uv, level(0)).rgb;
        float albedo = material.x;
        float emission = material[c + 1];
        if (livingEnabled) {
            albedo = clamp(material.x * living.albedo[c] / max(base.x, .02f), .02f, .92f);
            emission = clamp(material[c + 1] + living.emission[c] - base[c + 1], 0.f, .55f);
        }
        if (u.illumination.w > .5f) {
            albedo = .012f + .018f * material.x;
            emission *= .02f;
            incident = 1;
        }
        // The three IORs retain independent bed parallax. Their volume paths
        // share a midpoint quadrature because broadband dispersion is sub-voxel.
        float viewTransmission = cloud.transmittance[c]
            * exp(-sigma[c] * (bedHit.w - centralLength));
        // Ambient illumination uses a bounded local-column closure for the
        // incoming path; full 3D multiple scattering is deliberately not claimed.
        float lightPath = 1.18f;
        float lightTau = sigma[c] * depth * lightPath + columnExtinction[c] * (-centralRay.z) * lightPath;
        float reflected = albedo * cloudIllumination(u)[c] * incident[c] * viewTransmission * exp(-lightTau);
        if (livingEnabled) {
            float flatFlux = max(0.f, l.z) * (1 - fresnel(l.z)[c]);
            float focused = caustics.sample(linearClamp, (location + .5f) / 3.f, level(0))[c];
            float coverage = causticsCoverage(location, depth, l);
            float irradiance = mix(flatFlux, max(0.f, focused), coverage);
            float incomingTau = (sigma[c] * depth + columnExtinction[c] * (-centralRay.z)) / lightCosine;
            reflected += albedo * directSource[c] * irradiance * facet
                       * exp(-incomingTau) * viewTransmission;
        }
        float emitted = emission * (u.time.z * .24f) * viewTransmission;
        result[c] = reflected + emitted;
    }
    return result + cloud.radiance;
}

bool secondarySurfaceHit(texture2d<float> water, float3 start, float3 ray,
                         thread WaterIntersection &hit, constant Uniforms &u) {
    // An upward ray can still meet a steeper crest. Only its position above
    // the conservative cubic-height envelope proves that no hit is possible.
    if (start.z > u.world.z + .002f && ray.z >= 0) return false;
    float near = .001f;
    // Quadratic spacing resolves nearby folds while keeping a bounded work budget.
    for (uint i = 1; i <= 14; ++i) {
        float far = .001f + 1.8f * float(i * i) / 196.f;
        float3 p = start + ray * far;
        if (any(p.xy < 0.f) || any(p.xy > 2.f) || (p.z > u.world.z + .001f && ray.z > 0))
            return false;
        if (p.z < waterSurface(water,p.xy).x) {
            float3 nearPoint=start+ray*near;
            if (nearPoint.z-waterSurface(water,nearPoint.xy).x<0) {
                near=0;
                if (start.z-waterSurface(water,start.xy).x<0) return false;
            }
            for (uint k = 0; k < 7; ++k) {
                float mid = .5f * (near + far);
                float3 midPoint=start+ray*mid;
                if (midPoint.z-waterSurface(water,midPoint.xy).x > 0) near = mid;
                else far = mid;
            }
            float3 seed=start+ray*(.5f*(near+far));
            float3 point=start+ray*max(0.f,dot(seed-start,ray));
            WaterSurfacePatch patch=waterSurfacePatch(water,point.xy);
            WaterIntersection candidate=refineWaterIntersectionCached(water,start,ray,seed,patch,float2(near,far));
            hit=acceptWaterIntersection(water,start,ray,candidate,near,far);
            return true;
        }
        near = far;
    }
    return false;
}
bool secondaryHit(texture2d<float> water,float3 start,float3 ray,thread float3 &hit,constant Uniforms &u) {
    WaterIntersection result;
    bool found=secondarySurfaceHit(water,start,ray,result,u);
    if (found) hit=result.point;
    return found;
}
float3 reflectedRadiance(texture2d<float> water, texture2d<float> dye,
                         texture2d<float> bed, texture2d<float> sky,
                         texture2d<float> lightField, texture3d<float> volume, texture2d<float> caustics,
                         float3 point, float3 normal, float3 ray, constant Uniforms &u,
                         constant float4 &bedWorld, float footprint) {
    float3 position = point + normal * .0003f;
    float3 direction = reflect(ray, normal);
    float3 result = 0, throughput = 1;
    uint limit = uint(clamp(u.post.y, 0.f, 2.f));
    for (uint bounce = 0; bounce < limit; ++bounce) {
        WaterIntersection intersection;
        if (!secondarySurfaceHit(water, position, direction, intersection, u)) break;
        float3 hit=intersection.point;
        float3 n=normalize(float3(-intersection.slope,1));
        float3 f = fresnel(-dot(direction, n));
        // Radiance / n² is invariant along a lossless refracted ray. Caustic
        // flux has no eta² factor; underwater radiance exits into air here.
        result += throughput * (1 - f) / (IOR * IOR) * bedRadiance(water, dye, bed, lightField, volume,
            sky, caustics, hit, n, direction, u, bedWorld, footprint * 1.7f, false, 12);
        throughput *= f;
        direction = reflect(direction, n);
        position = hit + n * .0003f;
    }
    // Keep the unresolved radiance tail; a bounce budget must not turn folds black.
    return result + throughput * environment(sky, direction);
}

float3 directHighlight(float3 n, float3 view, float3 light) {
    if (dot(n,view) <= 0 || dot(n,light) <= 0) return 0;
    float ndv = max(.001f,dot(n,view)), ndl = max(.001f,dot(n,light));
    float3 halfway = normalize(light+view);
    float ndh = max(0.f,dot(n,halfway));
    float a2 = .0025f;
    float denom = ndh*ndh*(a2-1)+1;
    float d = a2/max(PI*denom*denom,1e-8f);
    float smith = 2*ndv*ndl/max(ndl*sqrt(a2+(1-a2)*ndv*ndv)
        +ndv*sqrt(a2+(1-a2)*ndl*ndl),1e-7f);
    return fresnel(dot(view,halfway))*d*smith*.18f/(4*ndv);
}
float3 directHighlightPixel(float3 n, float3 view, float3 light,
                            float3 nx, float3 ny, float3 vx, float3 vy) {
    // Adapt each axis to the angular footprint / physical alpha=.05. The
    // positive Gauss weights integrate pixel area without roughness inflation.
    // Four BRDF evaluations for ordinary pixels, at most 256 for steep folds;
    // no additional water intersections or volume samples.
    float2 width=float2(length(nx)+.5f*length(vx),length(ny)+.5f*length(vy));
    uint2 count=uint2(width.x>.2f ? 16u : (width.x>.1f ? 8u : (width.x>.04f ? 4u : 2u)),
                     width.y>.2f ? 16u : (width.y>.1f ? 8u : (width.y>.04f ? 4u : 2u)));
    constexpr float nodes[30]={-.2886751346f,.2886751346f,
        -.4305681558f,-.1699905218f,.1699905218f,.4305681558f,
        -.4801449282f,-.3983332387f,-.262766205f,-.0917173212f,
         .0917173212f,.262766205f,.3983332387f,.4801449282f,
        -.4947004675f,-.4722875115f,-.4328156012f,-.3777022042f,
        -.3089381222f,-.2290083888f,-.1408017754f,-.0475062549f,
         .0475062549f,.1408017754f,.2290083888f,.3089381222f,
         .3777022042f,.4328156012f,.4722875115f,.4947004675f};
    constexpr float weights[30]={.5f,.5f,
        .1739274226f,.3260725774f,.3260725774f,.1739274226f,
        .0506142681f,.1111905172f,.1568533229f,.1813418917f,
        .1813418917f,.1568533229f,.1111905172f,.0506142681f,
        .0135762297f,.031126762f,.0475792558f,.0623144856f,
        .0747979944f,.0845782597f,.0913017075f,.0947253052f,
        .0947253052f,.0913017075f,.0845782597f,.0747979944f,
        .0623144856f,.0475792558f,.031126762f,.0135762297f};
    uint2 base=count-2;
    float3 result = 0;
    for (uint y=0; y<count.y; ++y) for (uint x=0; x<count.x; ++x) {
        float2 offset=float2(nodes[base.x+x],nodes[base.y+y]);
        result+=directHighlight(normalize(n+nx*offset.x+ny*offset.y),
            normalize(view+vx*offset.x+vy*offset.y),light)*weights[base.x+x]*weights[base.y+y];
    }
    return result;
}

fragment float4 waterFragment(VertexOut in [[stage_in]],
    constant Uniforms &u [[buffer(0)]], constant float4 &bedWorld [[buffer(2)]],
    texture2d<float> water [[texture(0)]], texture2d<float> dye [[texture(1)]],
    texture2d<float> bed [[texture(2)]], texture2d<float> sky [[texture(3)]],
    texture2d<float> lightField [[texture(4)]], texture3d<float> volume [[texture(5)]],
    texture2d<float> caustics [[texture(6)]]) {
    float2 screen = float2(in.uv.x * 2 - 1, 1 - in.uv.y * 2);
    float3 ray = normalize(u.forward.xyz + screen.x * u.right.w * u.right.xyz
                         + screen.y * u.up.w * u.up.xyz);
    WaterIntersection intersection=intersectWaterData(water,u.eye.xyz,ray,u);
    float3 point=intersection.point;
    float footprint = max(length(dfdx(point.xy)), length(dfdy(point.xy)));
    float4 surface = field(water, point.xy);
    float3 n = normalize(float3(-intersection.slope,1));
    float3 view = -ray;
    if (dot(n, view) < .001f) n = normalize(n + view * (.001f - dot(n, view)));
    float ndv = max(.001f, dot(n, view));
    float3 f = coating(fresnel(ndv), ndv, surface.x, u);
    float3 reflected = reflectedRadiance(water, dye, bed, sky, lightField, volume, caustics,
        point, n, ray, u, bedWorld, footprint);
    uint cloudSteps = u.post.y > 1.5f ? 32u : (u.post.y > .5f ? 24u : 16u);
    float3 transmitted = bedRadiance(water, dye, bed, lightField, volume, sky, caustics,
        point, n, ray, u, bedWorld, footprint, true, cloudSteps);

    float3 l = normalize(u.light.xyz);
    float3 emitter = environment(sky, l);
    float3 specular = emitter*directHighlightPixel(n,view,l,dfdx(n),dfdy(n),dfdx(view),dfdy(view));

    float3 radiance = f * reflected + specular + (1 - f) * transmitted / (IOR * IOR);
    float foam = saturate(surface.w);
    float3 foamLight = cloudIllumination(u) * exp(-pigmentTau(cloudSample(volume, point - float3(0,0,.02f))) * .02f);
    radiance = mix(radiance, foamLight, foam * .82f);
    return float4(clamp(radiance, 0.f, 64000.f), 1);
}

// A small instanced draw makes each airborne drop a recognizable glass sphere,
// without adding a loop over every drop to the full-screen water pass.
struct VisibleDrop { float4 position, pigment, dynamics; uint4 identity; };
struct DropletOut {
    float4 position [[position]];
    float4 sphere [[flat]];
    float3 pigment [[flat]];
};
vertex DropletOut dropletVertex(uint id [[vertex_id]], uint instance [[instance_id]],
                                constant Uniforms &u [[buffer(0)]],
                                device const VisibleDrop *drops [[buffer(1)]]) {
    const float2 corners[6] = { float2(-1,-1), float2(1,-1), float2(-1,1),
                               float2(-1,1), float2(1,-1), float2(1,1) };
    VisibleDrop d = drops[instance];
    float3 p = d.position.xyz + d.position.w * 1.12f
        * (u.right.xyz * corners[id].x + u.up.xyz * corners[id].y);
    float3 relative = p - u.eye.xyz;
    float distance = dot(relative, u.forward.xyz);
    float4 clip = float4(dot(relative, u.right.xyz) / u.right.w,
                         dot(relative, u.up.xyz) / u.up.w, distance * .5f, distance);
    return { clip, d.position, d.pigment.rgb };
}
fragment float4 dropletFragment(DropletOut in [[stage_in]],
                                constant Uniforms &u [[buffer(0)]],
                                texture2d<float> water [[texture(0)]],
                                texture2d<float> sky [[texture(3)]]) {
    float2 screen = float2(in.position.x * u.viewport.z * 2 - 1,
                           1 - in.position.y * u.viewport.w * 2);
    float3 ray = normalize(u.forward.xyz + screen.x * u.right.w * u.right.xyz
                            + screen.y * u.up.w * u.up.xyz);
    float3 relative = u.eye.xyz - in.sphere.xyz;
    float b = dot(relative, ray);
    float h = b * b - dot(relative, relative) + in.sphere.w * in.sphere.w;
    if (h <= 0) discard_fragment();
    float chord = 2 * sqrt(h);
    float distance = -b - .5f * chord;
    if (distance <= 0) discard_fragment();
    float3 point = u.eye.xyz + ray * distance;
    if (point.z < field(water, point.xy).x) discard_fragment();
    float3 n = normalize(point - in.sphere.xyz);
    float3 f = fresnel(-dot(ray, n));
    float3 absorption = pigmentTau(in.pigment * 65);
    float3 scattering = pigmentScattering(in.pigment * 65);
    float3 sigma = absorption + scattering;
    float3 transmission = exp(-sigma * chord);
    float3 reflected = environment(sky, reflect(ray, n));
    float3 refracted = environment(sky, refract(ray, n, 1.f / 1.333f));
    float3 radiance = f * reflected + (1 - f) * (refracted * transmission
        + cloudIllumination(u) * scattering / max(sigma, 1e-7f) * (1 - transmission));
    float3 halfway = normalize(normalize(u.light.xyz) - ray);
    float highlight = pow(max(0.f, dot(n, halfway)), 140.f) * 3;
    radiance += fresnel(dot(-ray, halfway)) * environment(sky, u.light.xyz) * highlight;
    float edge = smoothstep(0.f, max(fwidth(h), 1e-9f), h);
    return float4(clamp(radiance, 0.f, 64000.f), edge);
}

// One GPU thread reduces a sparse, deterministic luminance grid. It introduces
// no CPU readback or synchronization; the next pass consumes the adapted pupil.
kernel void adaptExposure(texture2d<float, access::sample> current [[texture(0)]],
                          texture2d<float, access::read> previous [[texture(1)]],
                          texture2d<float, access::write> result [[texture(2)]],
                          constant Uniforms &u [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float sum = 0;
    for (uint y = 0; y < 18; ++y)
        for (uint x = 0; x < 32; ++x) {
            float3 rgb = current.sample(linearClamp, (float2(x, y) + .5f) / float2(32, 18), level(0)).rgb;
            sum += log(max(.001f, dot(rgb, LUMA)));
        }
    float target = u.illumination.w > .5f ? .85f : clamp(.25f / exp(sum / 576), .4f, 6.4f);
    float old = previous.read(uint2(0)).r;
    if (!isfinite(old) || old < .4f) old = 1;
    float rate = target < old ? 4.5f : 1.1f;
    float value = mix(old, target, 1 - exp(-rate * clamp(u.time.y, 0.f, .1f)));
    result.write(float4(value, 0, 0, 1), uint2(0));
}

float3 toneMap(float3 hdr, float exposure, constant Uniforms &u) {
    float3 gain = u.gain.rgb / max(dot(u.gain.rgb, LUMA), 1e-6f);
    float3 rgb = max(hdr, 0.f) * gain * exposure * u.optics.z;
    float y = dot(rgb, LUMA);
    float saturation = u.illumination.w > .5f ? 1.15f : clamp(u.optics.w, 0.f, 2.f);
    rgb = y + (rgb - y) * saturation;
    float minimum = min(rgb.r, min(rgb.g, rgb.b));
    if (minimum < 0) rgb = max(0.f, y + (rgb - y) * y / max(y - minimum, 1e-7f));
    float p = max(rgb.r, max(rgb.g, rgb.b));
    if (p < 1e-10f) return 0;
    float tone = min(.9995f, p * (2.51f * p + .03f) / (p * (2.43f * p + .59f) + .14f));
    tone = clamp(tone + (u.gain.w - 1) * tone * (1 - tone) * (tone - .18f), 0.f, .9995f);
    return rgb * (tone / p);
}

// Exact finite-volume restriction of piecewise-constant radiance. Each output
// covers its complete source-pixel footprint, including fractional edge areas.
// Across the image the work is O(source pixels + output pixels), even at Native
// resolutions; no fixed tap count can discard small highlights between samples.
float3 retinaAreaAverage(texture2d<float,access::read> current, uint2 p, uint2 outputSize) {
    uint2 size = uint2(current.get_width(),current.get_height());
    float2 scale = float2(size)/float2(outputSize);
    float2 lo = float2(p)*scale;
    float2 hi = min(float2(p+1)*scale,float2(size));
    uint2 first = min(uint2(floor(lo)),size-1);
    uint2 end = min(uint2(ceil(hi)),size);
    float3 sum = 0;
    for (uint y = first.y; y < end.y; ++y) {
        float wy = max(0.f,min(hi.y,float(y+1))-max(lo.y,float(y)));
        for (uint x = first.x; x < end.x; ++x) {
            float wx = max(0.f,min(hi.x,float(x+1))-max(lo.x,float(x)));
            sum += max(current.read(uint2(x,y)).rgb,0.f)*(wx*wy);
        }
    }
    return sum/max((hi.x-lo.x)*(hi.y-lo.y),1e-12f);
}

struct RetinaUpdateParameters { float4 attack, release; };

float3 retinaIntegrate(float3 now, float3 old, float attack, float release) {
    if (!all(isfinite(old))) return now;
    // Shared RGB response preserves the captured spectrum. Convex integration
    // cannot exceed the input/history envelope or feed subjective color back.
    float alpha = dot(now,LUMA) > dot(old,LUMA) ? attack : release;
    return mix(old,now,saturate(alpha));
}

// Reservoirs contain only unwarped scene-linear HDR; distortions are output-only.
// attack.xyz/release.xyz are wall-time coefficients; attack.w is prior-valid.
// All three layers share the exact area restriction in this single dispatch.
kernel void retinaHistoryUpdate(texture2d<float,access::read> current [[texture(0)]],
    texture2d<float,access::read> oldFast [[texture(1)]],
    texture2d<float,access::read> oldMiddle [[texture(2)]],
    texture2d<float,access::read> oldSlow [[texture(3)]],
    texture2d<float,access::write> nextFast [[texture(4)]],
    texture2d<float,access::write> nextMiddle [[texture(5)]],
    texture2d<float,access::write> nextSlow [[texture(6)]],
    constant RetinaUpdateParameters &state [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    uint2 size = uint2(nextFast.get_width(),nextFast.get_height());
    if (any(p >= size)) return;
    float3 now = retinaAreaAverage(current,p,size);
    if (state.attack.w <= 0) {
        nextFast.write(float4(now,1),p); nextMiddle.write(float4(now,1),p); nextSlow.write(float4(now,1),p);
        return;
    }
    nextFast.write(float4(retinaIntegrate(now,oldFast.read(p).rgb,state.attack.x,state.release.x),1),p);
    nextMiddle.write(float4(retinaIntegrate(now,oldMiddle.read(p).rgb,state.attack.y,state.release.y),1),p);
    nextSlow.write(float4(retinaIntegrate(now,oldSlow.read(p).rgb,state.attack.z,state.release.z),1),p);
}

float retinaHistoryChromaRetention(float3 now, float3 old) {
    float peak = max(now.x,max(now.y,now.z)), oldPeak = max(old.x,max(old.y,old.z));
    float saturation = (peak-min(now.x,min(now.y,now.z)))/max(peak,1e-6f);
    float oldSaturation = (oldPeak-min(old.x,min(old.y,old.z)))/max(oldPeak,1e-6f);
    // Bright neutral history must not bleach currently visible pigment. Dark
    // current still accepts afterimages, and black history cannot add a veil.
    float protection = smoothstep(.04f,.55f,saturation)*smoothstep(.005f,.045f,peak)
        *smoothstep(.015f,.06f,oldPeak/max(peak,.001f));
    float required = max(saturation,.001f);
    return mix(1.f,smoothstep(.25f*required,.8f*required,oldSaturation),protection);
}

float3 historySample(texture2d<float> current, texture2d<float> previous,
                     float2 uv, constant Uniforms &u) {
    float3 now = current.sample(mirrorSampler, uv, level(0)).rgb;
    if (u.post.z <= 0) return now;
    float3 old = previous.sample(mirrorSampler, uv, level(0)).rgb;
    if (!all(isfinite(old))) return now;
    float chroma = dot(normalize(now + .001f), normalize(old + .001f));
    return mix(now, old, saturate(u.post.z) * smoothstep(.6f, .95f, chroma)
        *retinaHistoryChromaRetention(now,old));
}

// state = (active effect clock, subjective intensity 0..2, reservoirs valid, 0).
// The host freezes this clock and the reservoirs while paused or showing help.
float retinaDose(constant Uniforms &u, float4 state) {
    if (u.post.x < .5f || u.time.z <= 0 || state.y <= 0) return 0;
    float t = state.x;
    float envelope = .72f + .17f*sin(t*.19f+.7f) + .11f*sin(t*.071f-1.2f);
    return clamp(state.y,0.f,2.f)*clamp(u.time.z*.25f,0.f,2.f)*envelope;
}

// Return logical display pixels. In particular, the configured radius is a
// strict vector-length ceiling, not an average later exceeded by modulation.
float2 retinaChannelOffset(float2 uv, uint channel, constant Uniforms &u,
                           float4 state, float dose) {
    float phase = float(channel)*(2*PI/3);
    float t = state.x;
    float field = .19f*sin(uv.y*6+t*.09f) + .11f*sin(uv.x*5-t*.043f);
    float angle = t*(.13f+.009f*float(channel))+phase+field;
    float breathing = .675f+.325f*sin(t*.12f+phase+.23f*sin(uv.x*4+uv.y*3));
    float radius = max(u.time.w,0.f)*breathing*smoothstep(0.f,1.f,dose);
    return float2(cos(angle),sin(angle))*radius;
}

float3 retinaTemporal(texture2d<float> current, texture2d<float> previous,
    texture2d<float> fast, texture2d<float> middle, texture2d<float> slow,
    float2 uv, constant Uniforms &u, float4 state, float dose) {
    float3 now = historySample(current,previous,uv,u);
    if (state.z <= 0 || dose <= 0) return now;
    float2 pixel = u.viewport.zw*u.post.w;
    float t = state.x;
    float2 bend = float2(sin(uv.y*7+t*.17f),cos(uv.x*6-t*.13f));
    float amplitude = (.5f+.45f*max(u.time.w,0.f))*min(dose,2.f);
    float3 a = fast.sample(mirrorSampler,uv+pixel*amplitude*(.35f*bend+float2(sin(t*.31f),cos(t*.23f))*.35f),level(0)).rgb;
    float3 b = middle.sample(mirrorSampler,uv+pixel*amplitude*(.60f*bend+float2(cos(t*.19f+.9f),sin(t*.27f))*.6f),level(0)).rgb;
    float3 c = slow.sample(mirrorSampler,uv+pixel*amplitude*(-.8f*bend+float2(sin(t*.11f+2),cos(t*.17f))*.75f),level(0)).rgb;
    if (!all(isfinite(a))) a = now;
    if (!all(isfinite(b))) b = now;
    if (!all(isfinite(c))) c = now;
    float3 direction = normalize(now+.001f);
    float3 affinity = smoothstep(.35f,.92f,float3(dot(direction,normalize(a+.001f)),
        dot(direction,normalize(b+.001f)),dot(direction,normalize(c+.001f))));
    affinity *= float3(retinaHistoryChromaRetention(now,a),retinaHistoryChromaRetention(now,b),
                       retinaHistoryChromaRetention(now,c));
    float3 weights = float3(.12f,.11f,.09f)*min(dose,1.7f)*affinity;
    // Sum <= .544: every old layer has a nonnegative, bounded contribution.
    return now*(1-dot(weights,float3(1)))+a*weights.x+b*weights.y+c*weights.z;
}

float3 retinaChroma(float3 color, float contrast, float2 uv, float4 state, float dose) {
    float y = dot(color,LUMA);
    if (y <= 1e-8f) return color;
    float3 chroma = color-y;
    float t = state.x;
    float2 p = (uv-.5f)*8;
    float field = sin(p.x+sin(p.y*1.37f+t*.11f))*cos(p.y*.91f-t*.087f);
    field += .35f*sin(p.x*2.3f-p.y*1.7f+t*.061f);
    float amount = min(dose,2.f)*saturate(contrast*4);
    float angle = .25f*amount*field;
    // Rotate only in the zero-luminance plane: grey cannot acquire a tint.
    float3 axis = normalize(LUMA);
    float3 rotated = (chroma*cos(angle)+cross(axis,chroma)*sin(angle))
        *(1+.08f*amount*sin(field*1.8f+t*.07f));
    float minimum = min(rotated.x,min(rotated.y,rotated.z));
    float scale = minimum < -y ? y/max(-minimum,1e-8f) : 1;
    return max(y+rotated*scale,0.f);
}

float3 retinaDiffraction(texture2d<float> current, float3 center, float2 uv,
                        constant Uniforms &u, float4 state, float dose,
                        thread float &contrast) {
    float2 pixel = u.viewport.zw*u.post.w;
    float3 inner = 0, outer = 0;
    float angle = .24f+.07f*sin(state.x*.083f);
    // Eight shared RGB taps, not three complete per-channel post pipelines.
    // Two symmetric rings interpolate positive spectral PSFs. Their second
    // moments scale with wavelength squared, the diffraction radius law.
    for (uint i = 0; i < 4; ++i) {
        float a = angle+float(i)*PI*.5f;
        float2 direction = float2(cos(a),sin(a))*pixel;
        float3 near = current.sample(mirrorSampler,uv+direction*3.5f,level(0)).rgb;
        float3 far = current.sample(mirrorSampler,uv+direction*10.5f,level(0)).rgb;
        inner += near*.25f;
        outer += far*.25f;
    }
    float3 wavelengths = float3(650,550,460)/550;
    float3 outerWeight = (36*wavelengths*wavelengths-12.25f)/98;
    float3 scattered = mix(inner,outer,outerWeight);
    float strength = .13f*min(dose,1.7f);
    contrast = length(center-inner)/max(.05f,length(center)+length(inner));
    // Linear positive normalized kernels preserve each spectral integral.
    // High-energy glints dominate naturally; no white light is added. Applying
    // a threshold after bilinear sampling would silently destroy glint energy.
    return mix(center,scattered,strength);
}

fragment float4 postFragment(VertexOut in [[stage_in]],
    constant Uniforms &u [[buffer(0)]], constant float4 &retinaState [[buffer(1)]],
    texture2d<float> current [[texture(0)]], texture2d<float> previous [[texture(1)]],
    texture2d<float,access::read> exposure [[texture(2)]],
    texture2d<float> fast [[texture(3)]], texture2d<float> middle [[texture(4)]],
    texture2d<float> slow [[texture(5)]]) {
    float pupil = exposure.read(uint2(0)).r;
    float dose = retinaDose(u,retinaState);
    float3 rgb;
    if (dose <= 0) rgb = historySample(current,previous,in.uv,u);
    else {
        float2 pixel = u.viewport.zw*u.post.w;
        for (uint c = 0; c < 3; ++c) {
            float2 uv = in.uv+retinaChannelOffset(in.uv,c,u,retinaState,dose)*pixel;
            rgb[c] = retinaTemporal(current,previous,fast,middle,slow,uv,u,retinaState,dose)[c];
        }
        float contrast;
        rgb = retinaDiffraction(current,rgb,in.uv,u,retinaState,dose,contrast);
        rgb = retinaChroma(rgb,contrast,in.uv,retinaState,dose);
    }
    // Recombine scene-linear channels first, tone map once. The sRGB render
    // attachment performs the single transfer to encoded light.
    return float4(clamp(toneMap(rgb,pupil,u),0.f,1.f),1);
}
