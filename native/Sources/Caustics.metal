#include <metal_stdlib>
using namespace metal;

// Forward flux transport. Each source cell carries transmitted beam power;
// its refracted Jacobian shapes a normalized finite footprint on the bed.
// Irradiance is flux / receiver horizontal area, not an artistic brightness map.
// References: pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Radiometry and
// pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission .
struct CausticsUniforms {
    float4 light;  // normalized direction toward emitter; angular radius radians
    float4 domain; // source extent, receiver minimum, receiver extent, filter sigma pixels
    uint4 grid;    // source cells per axis, receiver width, receiver height, reserved
};
constexpr sampler causticsFieldSampler(coord::normalized, address::clamp_to_edge, filter::linear);

static float4 causticsField(texture2d<float> field, float2 position, float extent) {
    return field.sample(causticsFieldSampler, position / extent
        + .5f / float2(field.get_width(), field.get_height()), level(0));
}
static float causticsFresnel(float cosine, float ior) {
    float c = clamp(cosine, 0.f, 1.f);
    float t = sqrt(max(0.f, 1 - (1 - c*c) / (ior*ior)));
    float rs = (c - ior*t) / max(c + ior*t, 1e-7f);
    float rp = (ior*c - t) / max(ior*c + t, 1e-7f);
    return .5f * (rs*rs + rp*rp);
}
static float causticsBed(texture2d<float> water, texture2d<float> depth, float2 point, float extent) {
    return causticsField(water, point, extent).x - max(.001f, causticsField(depth, point, extent).a);
}
static float causticsBedResidual(texture2d<float> water, texture2d<float> depth,
                                float3 origin, float3 ray, float distance, float extent) {
    float3 p = origin + distance * ray;
    return p.z - causticsBed(water, depth, p.xy, extent);
}
// Bounded local bathymetry solve. Outside the surface patch its edge bed height
// continues as a local plane; receiver coverage fades these boundary rays.
static float causticsBedDistance(texture2d<float> water, texture2d<float> depth,
                                 float3 origin, float3 ray, float localDepth, float extent) {
    float near = 0, a = localDepth;
    float far = max(.001f, localDepth / max(-ray.z, .001f));
    float b = causticsBedResidual(water, depth, origin, ray, far, extent);
    for (uint i = 0; i < 4 && b > 0; ++i) {
        far *= 2;
        b = causticsBedResidual(water, depth, origin, ray, far, extent);
    }
    if (b > .001f) return -1; // no bounded receiver hit; power escapes this cache
    for (uint i = 0; i < 10; ++i) {
        if (abs(b) < .00001f) return far;
        float t = mix(near, far, clamp(a / max(a-b, 1e-8f), .03f, .97f));
        float residual = causticsBedResidual(water, depth, origin, ray, t, extent);
        if (abs(residual) < .00001f) return t;
        if (residual > 0) { near = t; a = residual; }
        else { far = t; b = residual; }
    }
    return .5f * (near + far);
}

// Shared with the view/refraction renderer. Directions must be normalized.
// xyz is the bed hit; w is its physical path length, <=0 on a missing hit.
float4 causticsBedHit(texture2d<float> water, texture2d<float> depth,
                     float3 source, float3 direction) {
    if (direction.z >= -.001f) return float4(source,-1);
    float localDepth = source.z-causticsBed(water,depth,source.xy,2.f);
    if (localDepth <= 0) return float4(source,0);
    float distance = causticsBedDistance(water,depth,source,direction,localDepth,2.f);
    return float4(source+direction*max(distance,0.f),distance);
}

kernel void causticsMap(texture2d<float> water [[texture(0)]], texture2d<float> depth [[texture(1)]],
                        texture2d_array<float,access::write> output [[texture(2)]],
                        constant CausticsUniforms &u [[buffer(0)]], uint3 p [[thread_position_in_grid]]) {
    if (p.x > u.grid.x || p.y > u.grid.x || p.z >= 3) return;
    float2 source = float2(p.xy) * (u.domain.x / float(max(1u,u.grid.x)));
    float4 field = causticsField(water, source, u.domain.x);
    float3 n = normalize(float3(-field.yz, 1));
    float3 l = normalize(u.light.xyz);
    float incidence = dot(n,l);
    float ior = p.z == 0 ? 1.331f : (p.z == 1 ? 1.333f : 1.337f);
    float3 ray = refract(-l,n,1.f/ior);
    float4 hit = incidence > 0 ? causticsBedHit(water,depth,float3(source,field.x),ray) : float4(source,field.x,-1);
    float distance = hit.w;
    if (distance <= 0) { output.write(float4(source,0,0),p.xy,p.z); return; }
    float2 receiver = hit.xy;
    // dA_surface = dA_horizontal / n.z; cos incidence belongs to incoming power.
    float incidentFluxDensity = max(incidence,0.f) / n.z * (1-causticsFresnel(incidence,ior));
    // Match the second moment of a finite angular disk by a Gaussian footprint.
    // Full angular quadrature is deliberately avoided; this is source filtering.
    float angularSigma = distance * max(u.light.w,0.f) / (2*ior);
    output.write(float4(receiver,incidentFluxDensity,angularSigma*angularSigma),p.xy,p.z);
}

struct CausticsVertexOut {
    float4 position [[position]];
    float2 center [[flat]];
    float3 inverseCovariance [[flat]];
    float amplitude [[flat]];
    uint channel [[flat]];
};

vertex CausticsVertexOut causticsVertex(uint id [[vertex_id]], uint channel [[instance_id]],
                                       constant CausticsUniforms &u [[buffer(0)]],
                                       texture2d_array<float,access::read> mapping [[texture(0)]]) {
    uint cells = max(1u,u.grid.x), cell = id/6;
    uint2 p = uint2(cell % cells,cell / cells);
    float4 a = mapping.read(p,channel), b = mapping.read(p+uint2(1,0),channel);
    float4 c = mapping.read(p+uint2(0,1),channel), d = mapping.read(p+uint2(1,1),channel);
    float fluxSum = a.z+b.z+c.z+d.z;
    float2 center = (a.xy*a.z+b.xy*b.z+c.xy*c.z+d.xy*d.z)/max(fluxSum,1e-12f);
    // Missing/downward-invalid rays carry zero flux. Their placeholder source
    // coordinates must not drag a neighboring valid caustic back onto the surface.
    if (a.z <= 0) a.xy = center;
    if (b.z <= 0) b.xy = center;
    if (c.z <= 0) c.xy = center;
    if (d.z <= 0) d.xy = center;
    float2 dx = .5f*(b.xy-a.xy+d.xy-c.xy);
    float2 dy = .5f*(c.xy-a.xy+d.xy-b.xy);
    float pixel = u.domain.z / float(min(u.grid.y,u.grid.z));
    float filtering = max(.5f,u.domain.w)*pixel;
    float variance = filtering*filtering + (a.w*a.z+b.w*b.z+c.w*c.z+d.w*d.z)/max(fluxSum,1e-12f);
    // Covariance of a uniformly illuminated mapped source square: J J^T / 12.
    float xx = (dx.x*dx.x+dy.x*dy.x)/12 + variance;
    float xy = (dx.x*dx.y+dy.x*dy.y)/12;
    float yy = (dx.y*dx.y+dy.y*dy.y)/12 + variance;
    float determinant = max(xx*yy-xy*xy,variance*variance);
    float3 inverse = float3(yy,-xy,xx)/determinant;
    float sourceArea = pow(u.domain.x/float(cells),2.f);
    float flux = .25f*fluxSum*sourceArea;
    // Circular 3-sigma truncation: normalization retains its full source flux.
    float amplitude = flux / (6.28318530718f*sqrt(determinant)*(1-exp(-4.5f)));
    const float2 corners[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),
                               float2(-1,1),float2(1,-1),float2(1,1)};
    float2 receiver = center + corners[id%6] * 3 * sqrt(float2(xx,yy));
    float2 uv = (receiver-u.domain.y)/u.domain.z;
    return {float4(uv.x*2-1,1-uv.y*2,0,1),center,inverse,amplitude,channel};
}

fragment float4 causticsFragment(CausticsVertexOut in [[stage_in]],
                                 constant CausticsUniforms &u [[buffer(0)]]) {
    float2 receiver = u.domain.y + in.position.xy / float2(u.grid.yz) * u.domain.z;
    float2 delta = receiver-in.center;
    float q = in.inverseCovariance.x*delta.x*delta.x
        + 2*in.inverseCovariance.y*delta.x*delta.y + in.inverseCovariance.z*delta.y*delta.y;
    if (q > 9) discard_fragment();
    float4 result = 0;
    result[in.channel] = in.amplitude*exp(-.5f*q);
    return result;
}

// Receiver coverage is geometry, not irradiance: deriving it from accumulated
// brightness would fill real dark caustic troughs and destroy energy contrast.
// Water.metal blends toward flat Snell/Fresnel illumination beyond this guard.
float causticsCoverage(float2 receiver, float depth, float3 light) {
    float3 l = normalize(light);
    float cosT = sqrt(max(.001f,1-dot(l.xy,l.xy)/(1.333f*1.333f)));
    float2 source = receiver + l.xy*max(depth,0.f)/(1.333f*cosT);
    float edge = min(min(source.x,source.y),min(2-source.x,2-source.y));
    return smoothstep(.04f,.20f,edge);
}
