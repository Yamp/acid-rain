// Faithful native water optics. Common types/sampling, caustics_common and
// bed_life are prepended by the host. All returned values are scene-linear HDR.
@group(0) @binding(0) var<uniform> waterUniforms: Uniforms;
@group(0) @binding(1) var<uniform> bedWorld: vec4f;
@group(0) @binding(2) var waterTex: texture_2d<f32>;
@group(0) @binding(3) var dyeTex: texture_2d<f32>;
@group(0) @binding(4) var bedTex: texture_2d<f32>;
@group(0) @binding(5) var skyTex: texture_2d<f32>;
@group(0) @binding(6) var bedLightTex: texture_2d<f32>;
@group(0) @binding(7) var cloudTex: texture_3d<f32>;
@group(0) @binding(8) var causticsTex: texture_2d<f32>;

const PI: f32 = 3.14159265358979323846;
const IOR: vec3f = vec3f(1.331, 1.333, 1.337);
struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}
@vertex fn waterVertex(@builtin(vertex_index) id: u32) -> VertexOut {
    let p = vec2f(select(-1.0, 3.0, id == 1u), select(-1.0, 3.0, id == 2u));
    return VertexOut(vec4f(p, 0.0, 1.0), vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}
fn field(tex: texture_2d<f32>, metres: vec2f) -> vec4f {
    return sample_clamp2(tex, metres * 0.5 + vec2f(0.5) / vec2f(textureDimensions(tex)));
}
// Positive cubic B-spline: four linear taps, no negative lobes or new scene rays.
// Used for the low-resolution environment; height uses interpolating geometry.
fn opticalCubicAxis(f: f32) -> vec4f {
    let a=1.0-f; let f2=f*f; let f3=f2*f;
    let w=vec4f(a*a*a,3.0*f3-6.0*f2+4.0,-3.0*f3+3.0*f2+3.0*f+1.0,f3)/6.0;
    let g=vec2f(w.x+w.y,w.z+w.w);
    return vec4f(w.y/g.x-1.0,w.w/g.y+1.0,g);
}
fn opticalCubicTap(tex: texture_2d<f32>, uv: vec2f, repeatU: bool) -> vec4f {
    if (repeatU) { return sample_repeat2(tex,uv); }
    return sample_clamp2(tex,uv);
}
fn opticalCubicSample(tex: texture_2d<f32>, uv: vec2f, repeatU: bool) -> vec4f {
    let size=vec2f(textureDimensions(tex));
    let p=uv*size-vec2f(0.5); let cell=floor(p);
    let x=opticalCubicAxis(p.x-cell.x); let y=opticalCubicAxis(p.y-cell.y);
    let lo=(cell+vec2f(x.x,y.x)+vec2f(0.5))/size;
    let hi=(cell+vec2f(x.y,y.y)+vec2f(0.5))/size;
    return mix(mix(opticalCubicTap(tex,lo,repeatU),
                   opticalCubicTap(tex,vec2f(hi.x,lo.y),repeatU),x.w),
               mix(opticalCubicTap(tex,vec2f(lo.x,hi.y),repeatU),
                   opticalCubicTap(tex,hi,repeatU),x.w),y.w);
}
fn surfaceNormal(water: texture_2d<f32>, metres: vec2f) -> vec3f {
    let slope=waterSurface(water,metres,2.0).yz;
    return normalize(vec3f(-slope,1.0));
}
fn environment(sky: texture_2d<f32>, incoming: vec3f) -> vec3f {
    let direction = normalize(incoming);
    var longitude=0.0;
    if (dot(direction.xy,direction.xy) > 1e-12) { longitude=atan2(direction.y,direction.x); }
    let uv = vec2f(longitude / (2.0 * PI),
                  acos(clamp(direction.z, -1.0, 1.0)) / PI);
    return opticalCubicSample(sky,uv,true).rgb;
}
fn pigmentTau(mass: vec3f) -> vec3f {
    let m = max(mass, vec3f(0.0));
    return 0.85 * vec3f(dot(m, vec3f(2.0, 0.30, 0.25)),
                       dot(m, vec3f(0.65, 1.8, 0.65)),
                       dot(m, vec3f(0.30, 0.45, 1.8)));
}
fn fresnel(cosine: f32) -> vec3f {
    let c = clamp(abs(cosine), 0.00001, 1.0);
    let t = sqrt(max(vec3f(1.0) - vec3f(1.0 - c*c) / (IOR*IOR), vec3f(0.0)));
    let p = (IOR*c-t)/(IOR*c+t);
    let s = (vec3f(c)-IOR*t)/(vec3f(c)+IOR*t);
    return 0.5*(p*p+s*s);
}
fn coating(base: vec3f, cosine: f32, height: f32, u: Uniforms) -> vec3f {
    let thickness = 475.0 + 220.0*sin(u.time.x*0.15*0.31*0.35 + height*18.0);
    let ct = sqrt(max(0.0, 1.0-(1.0-cosine*cosine)/(1.46*1.46)));
    let path = 2.0*1.46*thickness*ct;
    let r01 = (1.0-1.46)/(1.0+1.46);
    let r12 = (1.46-1.333)/(1.46+1.333);
    var airy = vec3f(0.0);
    for (var j = -1; j <= 1; j += 1) {
        let wavelengths = vec3f(640.0,545.0,450.0) + vec3f(f32(j)*38.0);
        let interference = 2.0*r01*r12*cos(vec3f(2.0*PI*path)/wavelengths);
        airy += select(0.25,0.5,j == 0) *
            (vec3f(r01*r01+r12*r12)+interference) /
            (vec3f(1.0+r01*r01*r12*r12)+interference);
    }
    return clamp(base+(vec3f(1.0)-base)*(airy-vec3f(0.0204))*min(u.light.w,2.0)*0.65,
                 vec3f(0.0),vec3f(1.0));
}
fn heightResidual(water: texture_2d<f32>, eye: vec3f, ray: vec3f, distance: f32) -> f32 {
    let p = eye+ray*distance;
    return p.z-field(water,p.xy).x;
}
struct WaterIntersection { point: vec3f, slope: vec2f, height: f32, distance: f32 }
fn refineWaterIntersectionCached(water: texture_2d<f32>, eye: vec3f, ray: vec3f,
    seed: vec3f, initialPatch: WaterSurfacePatch, bracket: vec2f) -> WaterIntersection {
    var distance=max(0.0,dot(seed-eye,ray)); var point=eye+ray*distance;
    var surfacePatch=initialPatch;
    let cellDistance=1.0/max(length(ray.xy*surfacePatch.scale),1e-5);
    for (var iteration=0u;iteration<4u;iteration+=1u) {
        let cell=vec2i(floor(point.xy*surfacePatch.scale));
        if (any(cell!=surfacePatch.cell)) { surfacePatch=waterSurfacePatch(water,point.xy,2.0); }
        let surface=waterSurfaceValue(surfacePatch,point.xy);
        let residual=point.z-surface.x; let derivative=ray.z-dot(surface.yz,ray.xy);
        if (abs(residual)<1e-7 || abs(derivative)<1e-5) { return WaterIntersection(point,surface.yz,surface.x,distance); }
        let next=max(0.0,distance-clamp(residual/derivative,-cellDistance,cellDistance));
        if (next<bracket.x || next>bracket.y) { return WaterIntersection(point,surface.yz,surface.x,distance); }
        distance=next;
        point=eye+ray*distance;
    }
    // The last update can cross a cell; its gradient must match the final point.
    if (any(vec2i(floor(point.xy*surfacePatch.scale))!=surfacePatch.cell)) {
        surfacePatch=waterSurfacePatch(water,point.xy,2.0);
    }
    let surface=waterSurfaceValue(surfacePatch,point.xy);
    return WaterIntersection(point,surface.yz,surface.x,distance);
}
fn refineWaterIntersectionData(water: texture_2d<f32>, eye: vec3f, ray: vec3f, seed: vec3f) -> WaterIntersection {
    let point=eye+ray*max(0.0,dot(seed-eye,ray));
    return refineWaterIntersectionCached(water,eye,ray,seed,waterSurfacePatch(water,point.xy,2.0),vec2f(0.0,1e30));
}
fn cubicWaterBracket(water: texture_2d<f32>, eye: vec3f, ray: vec3f,
    nearInput: f32, farInput: f32) -> WaterIntersection {
    var near=nearInput; var far=farInput;
    let left=eye+ray*near; let right=eye+ray*far;
    let ls=waterSurface(water,left.xy,2.0); let rs=waterSurface(water,right.xy,2.0);
    var a=left.z-ls.x; var b=right.z-rs.x;
    var best=WaterIntersection(right,rs.yz,rs.x,far);
    if (abs(a)<abs(b)) { best=WaterIntersection(left,ls.yz,ls.x,near); }
    var error=min(abs(a),abs(b));
    if (!finite1(a) || !finite1(b) || (a>0.0)==(b>0.0)) { return best; }
    var surfacePatch=waterSurfacePatch(water,left.xy,2.0);
    for (var iteration=0u;iteration<28u;iteration+=1u) {
        if (error<=1e-7) { break; }
        let middle=0.5*(near+far);
        if (middle==near || middle==far) { break; }
        let point=eye+ray*middle;
        if (any(vec2i(floor(point.xy*surfacePatch.scale))!=surfacePatch.cell)) {
            surfacePatch=waterSurfacePatch(water,point.xy,2.0);
        }
        let surface=waterSurfaceValue(surfacePatch,point.xy);
        let residual=point.z-surface.x;
        if (abs(residual)<error) { best=WaterIntersection(point,surface.yz,surface.x,middle); error=abs(residual); }
        if ((residual>0.0)==(a>0.0)) { near=middle; a=residual; }
        else { far=middle; b=residual; }
    }
    return best;
}
fn acceptWaterIntersection(water: texture_2d<f32>, eye: vec3f, ray: vec3f,
    candidate: WaterIntersection, near: f32, far: f32) -> WaterIntersection {
    // Preserve the exact computed parameter at converged bracket endpoints.
    if (abs(candidate.point.z-candidate.height)<=1e-6 && candidate.distance>=near && candidate.distance<=far) { return candidate; }
    return cubicWaterBracket(water,eye,ray,near,far);
}
fn refineWaterIntersection(water: texture_2d<f32>, eye: vec3f, ray: vec3f, seed: vec3f) -> vec3f {
    return refineWaterIntersectionData(water,eye,ray,seed).point;
}
fn intersectWaterData(water: texture_2d<f32>, eye: vec3f, ray: vec3f, u: Uniforms) -> WaterIntersection {
    var near = max(0.0,(u.world.z+0.00001-eye.z)/ray.z);
    var far = (u.world.y-0.00001-eye.z)/ray.z;
    var bracket=vec2f(near,far);
    var a = heightResidual(water,eye,ray,near);
    var b = heightResidual(water,eye,ray,far);
    for (var i = 0u; i < 12u; i += 1u) {
        let t = clamp(a/max(a-b,1e-9),0.05,0.95);
        let m = mix(near,far,t);
        let value = heightResidual(water,eye,ray,m);
        if (value > 0.0) { near=m; a=value; } else { far=m; b=value; }
    }
    let seed=eye+ray*mix(near,far,clamp(a/max(a-b,1e-9),0.0,1.0));
    let seedDistance=max(0.0,dot(seed-eye,ray));
    let point=eye+ray*seedDistance;
    let surfacePatch=waterSurfacePatch(water,point.xy,2.0);
    let seedResidual=point.z-waterSurfaceValue(surfacePatch,point.xy).x;
    if (seedDistance>bracket.x && seedDistance<bracket.y) {
        if (seedResidual>0.0) { bracket.x=seedDistance; } else { bracket.y=seedDistance; }
    }
    let result=refineWaterIntersectionCached(water,eye,ray,seed,surfacePatch,bracket);
    return acceptWaterIntersection(water,eye,ray,result,bracket.x,bracket.y);
}
fn intersectWater(water: texture_2d<f32>, eye: vec3f, ray: vec3f, u: Uniforms) -> vec3f {
    return intersectWaterData(water,eye,ray,u).point;
}
fn extinction(u: Uniforms) -> vec3f {
    return vec3f(1.2,1.6,1.2)*u.optics.y+vec3f(0.015);
}
fn cloudIllumination(u: Uniforms) -> vec3f {
    return select(u.illumination.rgb,vec3f(1.0,0.98,1.12),u.illumination.w > 0.5);
}
fn pigmentScattering(concentration: vec3f) -> vec3f {
    let c = max(concentration,vec3f(0.0));
    return 0.32*vec3f(dot(c,vec3f(0.03,0.90,1.10)),
                     dot(c,vec3f(0.85,0.04,0.65)),
                     dot(c,vec3f(1.10,0.50,0.04)));
}
fn cloudSample(volume: texture_3d<f32>, p: vec3f) -> vec3f {
    let uv = vec3f(p.xy*0.5,(p.z+0.8)/0.9);
    if (any(uv < vec3f(0.0)) || any(uv > vec3f(1.0))) { return vec3f(0.0); }
    return max(sample_clamp3(volume,uv).rgb,vec3f(0.0));
}
struct CloudOptics { transmittance: vec3f, radiance: vec3f, column: vec3f }
fn cloudTransfer(volume: texture_3d<f32>, origin: vec3f, direction: vec3f,
                 distance: f32, u: Uniforms, requestedSteps: u32) -> CloudOptics {
    var result = CloudOptics(vec3f(1.0),vec3f(0.0),vec3f(0.0));
    let steps = clamp(requestedSteps,1u,64u);
    let ds = max(distance,0.0)/f32(steps);
    for (var i = 0u; i < steps; i += 1u) {
        let c = cloudSample(volume,origin+direction*((f32(i)+0.5)*ds));
        let scattering = pigmentScattering(c)+vec3f(0.015);
        let sigma = extinction(u)+pigmentTau(c)+pigmentScattering(c);
        let tau = sigma*ds;
        let segment = exp(-tau);
        let removed = select(vec3f(1.0)-segment,
            tau*(vec3f(1.0)-0.5*tau+tau*tau/6.0),tau < vec3f(0.001));
        result.radiance += result.transmittance*scattering/max(sigma,vec3f(1e-7))
            *removed*cloudIllumination(u);
        result.transmittance *= segment;
        result.column += c*ds;
    }
    return result;
}
fn bedLocation(point: vec3f, hit: vec4f) -> vec2f {
    let displacement = hit.xy-point.xy;
    return point.xy+displacement*min(1.0,1.5/max(max(abs(displacement.x),abs(displacement.y)),1e-6));
}
// Derivatives are supplied by the uniform primary fragment path. Secondary
// rays use a conservative footprint and cannot reach any derivative builtin.
fn bedRadiance(water: texture_2d<f32>, dye: texture_2d<f32>, bed: texture_2d<f32>,
    lightField: texture_2d<f32>, volume: texture_3d<f32>, sky: texture_2d<f32>,
    caustics: texture_2d<f32>, point: vec3f, normal: vec3f, ray: vec3f,
    u: Uniforms, world: vec4f, footprint: f32, hydraulicNormal: vec3f,
    centralHit: vec4f, cloudSteps: u32) -> vec3f {
    if (centralHit.w <= 0.0) { return vec3f(0.0); }
    var result = vec3f(0.0);
    let sigma = extinction(u);
    let centralRay = refract(ray,normal,1.0/IOR.y);
    let centralLocation = bedLocation(point,centralHit);
    let centralLength = centralHit.w;
    let depth = max(0.015,point.z-centralHit.z);
    let cloud = cloudTransfer(volume,point,centralRay,centralLength,u,cloudSteps);
    let columnExtinction = pigmentTau(cloud.column)+pigmentScattering(cloud.column);
    let base = sample_clamp2(bed,(centralLocation+vec2f(1.5))/5.0);
    var living = BedMaterial(vec3f(base.x),base.yzw,vec2f(0.0),0.7);
    let livingEnabled = world.w > 0.5 && u.illumination.w < 0.5;
    if (livingEnabled) {
        living = animatedBed(world.xy+centralLocation,world.z,base,
                             footprint/max(0.2,-centralRay.z));
    }
    let l = normalize(u.light.xyz);
    let downLight = refract(-l,vec3f(0.0,0.0,1.0),1.0/IOR.y);
    let bedNormal = normalize(vec3f(hydraulicNormal.xy-living.slope*hydraulicNormal.z,hydraulicNormal.z));
    let lightCosine = max(0.001,-downLight.z);
    let bumpCosine = max(0.0,dot(bedNormal,-downLight));
    let facet = hydraulicNormal.z*bumpCosine/max(0.001,dot(hydraulicNormal,-downLight));
    let directSource = environment(sky,l)*(0.18/PI);
    for (var c = 0u; c < 3u; c += 1u) {
        let transmitted = refract(ray,normal,1.0/IOR[c]);
        if (transmitted.z >= -0.001) { continue; }
        var bedHit = centralHit;
        if (c != 1u) { bedHit = causticsBedHit(water,dye,point,transmitted); }
        if (bedHit.w <= 0.0) { continue; }
        let location = bedLocation(point,bedHit);
        let uv = (location+vec2f(1.5))/5.0;
        let material = sample_clamp2(bed,uv);
        var incident = sample_clamp2(lightField,uv).rgb;
        var albedo = material.x;
        var emission = material[c+1u];
        if (livingEnabled) {
            albedo=clamp(material.x*living.albedo[c]/max(base.x,0.02),0.02,0.92);
            emission=clamp(material[c+1u]+living.emission[c]-base[c+1u],0.0,0.55);
        }
        if (u.illumination.w > 0.5) {
            albedo=0.012+0.018*material.x; emission *= 0.02; incident=vec3f(1.0);
        }
        let viewTransmission=cloud.transmittance[c]*exp(-sigma[c]*(bedHit.w-centralLength));
        let lightPath=1.18;
        let lightTau=sigma[c]*depth*lightPath+columnExtinction[c]*(-centralRay.z)*lightPath;
        var reflected=albedo*cloudIllumination(u)[c]*incident[c]*viewTransmission*exp(-lightTau);
        if (livingEnabled) {
            let flatFlux=max(0.0,l.z)*(1.0-fresnel(l.z)[c]);
            let focused=sample_clamp2(caustics,(location+vec2f(0.5))/3.0)[c];
            let coverage=causticsCoverage(location,depth,l);
            let irradiance=mix(flatFlux,max(0.0,focused),coverage);
            let incomingTau=(sigma[c]*depth+columnExtinction[c]*(-centralRay.z))/lightCosine;
            reflected += albedo*directSource[c]*irradiance*facet*exp(-incomingTau)*viewTransmission;
        }
        let emitted=emission*(u.time.z*0.24)*viewTransmission;
        result[c]=reflected+emitted;
    }
    return result+cloud.radiance;
}
struct SecondaryIntersection { hit: vec3f, found: bool, slope: vec2f }
fn secondaryHit(water: texture_2d<f32>, start: vec3f, ray: vec3f, u: Uniforms) -> SecondaryIntersection {
    let miss=SecondaryIntersection(start,false,vec2f(0.0));
    // Upward rays may hit steeper crests; only the height envelope proves a miss.
    if (start.z > u.world.z+0.002 && ray.z >= 0.0) { return miss; }
    var near=0.001;
    for (var i=1u; i <= 14u; i += 1u) {
        var far=0.001+1.8*f32(i*i)/196.0;
        let p=start+ray*far;
        if (any(p.xy < vec2f(0.0)) || any(p.xy > vec2f(2.0)) ||
            (p.z > u.world.z+0.001 && ray.z > 0.0)) { return miss; }
        if (p.z < waterSurface(water,p.xy,2.0).x) {
            let nearPoint=start+ray*near;
            if (nearPoint.z-waterSurface(water,nearPoint.xy,2.0).x<0.0) {
                near=0.0;
                if (start.z-waterSurface(water,start.xy,2.0).x<0.0) { return miss; }
            }
            for (var k=0u; k < 7u; k += 1u) {
                let mid=0.5*(near+far);
                let midPoint=start+ray*mid;
                if (midPoint.z-waterSurface(water,midPoint.xy,2.0).x > 0.0) { near=mid; } else { far=mid; }
            }
            let seed=start+ray*(0.5*(near+far));
            let point=start+ray*max(0.0,dot(seed-start,ray));
            let surfacePatch=waterSurfacePatch(water,point.xy,2.0);
            let candidate=refineWaterIntersectionCached(water,start,ray,seed,surfacePatch,vec2f(near,far));
            let hit=acceptWaterIntersection(water,start,ray,candidate,near,far);
            return SecondaryIntersection(hit.point,true,hit.slope);
        }
        near=far;
    }
    return miss;
}
fn reflectedRadiance(water: texture_2d<f32>, dye: texture_2d<f32>, bed: texture_2d<f32>,
    sky: texture_2d<f32>, lightField: texture_2d<f32>, volume: texture_3d<f32>,
    caustics: texture_2d<f32>, point: vec3f, normal: vec3f, ray: vec3f,
    u: Uniforms, world: vec4f, footprint: f32) -> vec3f {
    var position=point+normal*0.0003;
    var direction=reflect(ray,normal);
    var result=vec3f(0.0);
    var throughput=vec3f(1.0);
    let limit=u32(clamp(u.post.y,0.0,2.0));
    for (var bounce=0u; bounce < limit; bounce += 1u) {
        let intersection=secondaryHit(water,position,direction,u);
        if (!intersection.found) { break; }
        let hit=intersection.hit;
        let n=normalize(vec3f(-intersection.slope,1.0));
        let f=fresnel(-dot(direction,n));
        let bedHit=causticsBedHit(water,dye,hit,refract(direction,n,1.0/IOR.y));
        result += throughput*(vec3f(1.0)-f)/(IOR*IOR)*bedRadiance(water,dye,bed,lightField,
            volume,sky,caustics,hit,n,direction,u,world,footprint*1.7,vec3f(0.0,0.0,1.0),bedHit,12u);
        throughput *= f;
        direction=reflect(direction,n);
        position=hit+n*0.0003;
    }
    return result+throughput*environment(sky,direction);
}
fn directHighlight(n: vec3f, view: vec3f, light: vec3f) -> vec3f {
    if (dot(n,view) <= 0.0 || dot(n,light) <= 0.0) { return vec3f(0.0); }
    let ndv=max(0.001,dot(n,view)); let ndl=max(0.001,dot(n,light));
    let halfway=normalize(light+view);
    let ndh=max(0.0,dot(n,halfway)); let a2=0.0025;
    let denom=ndh*ndh*(a2-1.0)+1.0;
    let d=a2/max(PI*denom*denom,1e-8);
    let smith=2.0*ndv*ndl/max(ndl*sqrt(a2+(1.0-a2)*ndv*ndv)
        +ndv*sqrt(a2+(1.0-a2)*ndl*ndl),1e-7);
    return fresnel(dot(view,halfway))*d*smith*0.18/(4.0*ndv);
}
fn directHighlightPixel(n: vec3f, view: vec3f, light: vec3f,
    nx: vec3f, ny: vec3f, vx: vec3f, vy: vec3f) -> vec3f {
    // Axis-adaptive positive quadrature; four ordinary / at most256 local BRDF
    // evaluations. Physical alpha stays .05; no additional scene rays.
    let width=vec2f(length(nx)+0.5*length(vx),length(ny)+0.5*length(vy));
    let count=vec2u(select(select(select(2u,4u,width.x>0.04),8u,width.x>0.1),16u,width.x>0.2),
                     select(select(select(2u,4u,width.y>0.04),8u,width.y>0.1),16u,width.y>0.2));
    let nodes=array<f32,30>(-0.2886751346,0.2886751346,
        -0.4305681558,-0.1699905218,0.1699905218,0.4305681558,
        -0.4801449282,-0.3983332387,-0.262766205,-0.0917173212,
         0.0917173212,0.262766205,0.3983332387,0.4801449282,
        -0.4947004675,-0.4722875115,-0.4328156012,-0.3777022042,
        -0.3089381222,-0.2290083888,-0.1408017754,-0.0475062549,
         0.0475062549,0.1408017754,0.2290083888,0.3089381222,
         0.3777022042,0.4328156012,0.4722875115,0.4947004675);
    let weights=array<f32,30>(0.5,0.5,
        0.1739274226,0.3260725774,0.3260725774,0.1739274226,
        0.0506142681,0.1111905172,0.1568533229,0.1813418917,
        0.1813418917,0.1568533229,0.1111905172,0.0506142681,
        0.0135762297,0.031126762,0.0475792558,0.0623144856,
        0.0747979944,0.0845782597,0.0913017075,0.0947253052,
        0.0947253052,0.0913017075,0.0845782597,0.0747979944,
        0.0623144856,0.0475792558,0.031126762,0.0135762297);
    let base=count-vec2u(2);
    var result=vec3f(0.0);
    for (var y=0u;y<count.y;y+=1u) {
        for (var x=0u;x<count.x;x+=1u) {
            let offset=vec2f(nodes[base.x+x],nodes[base.y+y]);
            result+=directHighlight(normalize(n+nx*offset.x+ny*offset.y),
                normalize(view+vx*offset.x+vy*offset.y),light)*weights[base.x+x]*weights[base.y+y];
        }
    }
    return result;
}
@fragment fn waterFragment(input: VertexOut) -> @location(0) vec4f {
    let u=waterUniforms;
    let screen=vec2f(input.uv.x*2.0-1.0,1.0-input.uv.y*2.0);
    let ray=normalize(u.forward.xyz+screen.x*u.right.w*u.right.xyz+screen.y*u.up.w*u.up.xyz);
    let intersection=intersectWaterData(waterTex,u.eye.xyz,ray,u);
    let point=intersection.point;
    let footprint=max(length(dpdx(point.xy)),length(dpdy(point.xy)));
    let surface=field(waterTex,point.xy);
    var n=normalize(vec3f(-intersection.slope,1.0));
    let view=-ray;
    if (dot(n,view) < 0.001) { n=normalize(n+view*(0.001-dot(n,view))); }
    // All derivatives are evaluated on this unconditional primary path.
    let dx=dpdx(n); let dy=dpdy(n);
    let vx=dpdx(view); let vy=dpdy(view);
    let centralHit=causticsBedHit(waterTex,dyeTex,point,refract(ray,n,1.0/IOR.y));
    let centralLocation=bedLocation(point,centralHit);
    let bedFootprint=max(footprint,max(length(dpdx(centralLocation)),length(dpdy(centralLocation))));
    let crossed=cross(dpdx(centralHit.xyz),dpdy(centralHit.xyz));
    var hydraulicNormal=vec3f(0.0,0.0,1.0);
    if (dot(crossed,crossed) > 1e-20) {
        hydraulicNormal=normalize(crossed);
        if (hydraulicNormal.z < 0.0) { hydraulicNormal = -hydraulicNormal; }
    }
    let ndv=max(0.001,dot(n,view));
    let f=coating(fresnel(ndv),ndv,surface.x,u);
    let reflected=reflectedRadiance(waterTex,dyeTex,bedTex,skyTex,bedLightTex,cloudTex,
        causticsTex,point,n,ray,u,bedWorld,footprint);
    let cloudSteps=select(select(16u,24u,u.post.y > 0.5),32u,u.post.y > 1.5);
    let transmitted=bedRadiance(waterTex,dyeTex,bedTex,bedLightTex,cloudTex,skyTex,
        causticsTex,point,n,ray,u,bedWorld,bedFootprint,hydraulicNormal,centralHit,cloudSteps);
    let l=normalize(u.light.xyz);
    let emitter=environment(skyTex,l);
    let specular=emitter*directHighlightPixel(n,view,l,dx,dy,vx,vy);
    var radiance=f*reflected+specular+(vec3f(1.0)-f)*transmitted/(IOR*IOR);
    let foam=clamp(surface.w,0.0,1.0);
    let foamLight=cloudIllumination(u)*exp(-pigmentTau(cloudSample(cloudTex,point-vec3f(0.0,0.0,0.02)))*0.02);
    radiance=mix(radiance,foamLight,foam*0.82);
    return vec4f(clamp(radiance,vec3f(0.0),vec3f(64000.0)),1.0);
}
