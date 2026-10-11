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
fn environment(sky: texture_2d<f32>, incoming: vec3f) -> vec3f {
    let direction = normalize(incoming);
    let uv = vec2f(atan2(direction.y, direction.x) / (2.0 * PI),
                  acos(clamp(direction.z, -1.0, 1.0)) / PI);
    return sample_repeat2(sky, uv).rgb;
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
fn intersectWater(water: texture_2d<f32>, eye: vec3f, ray: vec3f, u: Uniforms) -> vec3f {
    var near = max(0.0,(u.world.z+0.00001-eye.z)/ray.z);
    var far = (u.world.y-0.00001-eye.z)/ray.z;
    var a = heightResidual(water,eye,ray,near);
    var b = heightResidual(water,eye,ray,far);
    for (var i = 0u; i < 12u; i += 1u) {
        let t = clamp(a/max(a-b,1e-9),0.05,0.95);
        let m = mix(near,far,t);
        let value = heightResidual(water,eye,ray,m);
        if (value > 0.0) { near=m; a=value; } else { far=m; b=value; }
    }
    return eye+ray*mix(near,far,clamp(a/max(a-b,1e-9),0.0,1.0));
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
struct SecondaryIntersection { hit: vec3f, found: bool }
fn secondaryHit(water: texture_2d<f32>, start: vec3f, ray: vec3f, u: Uniforms) -> SecondaryIntersection {
    let miss=SecondaryIntersection(start,false);
    if (ray.z > 0.45 || (start.z > u.world.z+0.002 && ray.z >= 0.0)) { return miss; }
    var near=0.001;
    for (var i=1u; i <= 14u; i += 1u) {
        var far=0.001+1.8*f32(i*i)/196.0;
        let p=start+ray*far;
        if (any(p.xy < vec2f(0.0)) || any(p.xy > vec2f(2.0)) ||
            (p.z > u.world.z+0.001 && ray.z > 0.0)) { return miss; }
        if (p.z < field(water,p.xy).x) {
            for (var k=0u; k < 7u; k += 1u) {
                let mid=0.5*(near+far);
                if (heightResidual(water,start,ray,mid) > 0.0) { near=mid; } else { far=mid; }
            }
            return SecondaryIntersection(start+ray*(0.5*(near+far)),true);
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
        let surface=field(water,hit.xy);
        let n=normalize(vec3f(-surface.yz,1.0));
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
@fragment fn waterFragment(input: VertexOut) -> @location(0) vec4f {
    let u=waterUniforms;
    let screen=vec2f(input.uv.x*2.0-1.0,1.0-input.uv.y*2.0);
    let ray=normalize(u.forward.xyz+screen.x*u.right.w*u.right.xyz+screen.y*u.up.w*u.up.xyz);
    let point=intersectWater(waterTex,u.eye.xyz,ray,u);
    let footprint=max(length(dpdx(point.xy)),length(dpdy(point.xy)));
    let surface=field(waterTex,point.xy);
    var n=normalize(vec3f(-surface.yz,1.0));
    let view=-ray;
    if (dot(n,view) < 0.001) { n=normalize(n+view*(0.001-dot(n,view))); }
    // All derivatives are evaluated on this unconditional primary path.
    let dx=dpdx(n); let dy=dpdy(n);
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
    let halfVector=normalize(l+view);
    let ndl=max(0.001,dot(n,l));
    let ndh=max(0.0,dot(n,halfVector));
    let a2=clamp(0.0025+0.5*(dot(dx,dx)+dot(dy,dy)),0.0025,0.25);
    let denom=ndh*ndh*(a2-1.0)+1.0;
    let d=a2/max(PI*denom*denom,1e-8);
    let smith=2.0*ndv*ndl/max(ndl*sqrt(a2+(1.0-a2)*ndv*ndv)+ndv*sqrt(a2+(1.0-a2)*ndl*ndl),1e-7);
    let emitter=environment(skyTex,l);
    let specular=fresnel(dot(view,halfVector))*emitter*d*smith*0.18/(4.0*ndv);
    var radiance=f*reflected+specular+(vec3f(1.0)-f)*transmitted/(IOR*IOR);
    let foam=clamp(surface.w,0.0,1.0);
    let foamLight=cloudIllumination(u)*exp(-pigmentTau(cloudSample(cloudTex,point-vec3f(0.0,0.0,0.02)))*0.02);
    radiance=mix(radiance,foamLight,foam*0.82);
    return vec4f(clamp(radiance,vec3f(0.0),vec3f(64000.0)),1.0);
}
