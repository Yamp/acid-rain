// Instanced glass spheres; shared types/sampling are prepended by the host.
@group(0) @binding(0) var<uniform> dropUniforms: Uniforms;
@group(0) @binding(1) var<storage,read> visibleDrops: array<ARDrop>;
@group(0) @binding(2) var dropWater: texture_2d<f32>;
@group(0) @binding(3) var dropSky: texture_2d<f32>;
const DROP_PI: f32=3.14159265358979323846;
const DROP_IOR: vec3f=vec3f(1.331,1.333,1.337);
struct DropletOut {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) sphere: vec4f,
    @location(1) @interpolate(flat) pigment: vec3f,
}
fn dropFresnel(cosine: f32) -> vec3f {
    let c=clamp(abs(cosine),0.00001,1.0);
    let t=sqrt(max(vec3f(1.0)-vec3f(1.0-c*c)/(DROP_IOR*DROP_IOR),vec3f(0.0)));
    let p=(DROP_IOR*c-t)/(DROP_IOR*c+t);
    let s=(vec3f(c)-DROP_IOR*t)/(vec3f(c)+DROP_IOR*t);
    return 0.5*(p*p+s*s);
}
fn dropPigmentTau(mass: vec3f) -> vec3f {
    let m=max(mass,vec3f(0.0));
    return 0.85*vec3f(dot(m,vec3f(2.0,0.30,0.25)),dot(m,vec3f(0.65,1.8,0.65)),dot(m,vec3f(0.30,0.45,1.8)));
}
fn dropScattering(concentration: vec3f) -> vec3f {
    let c=max(concentration,vec3f(0.0));
    return 0.32*vec3f(dot(c,vec3f(0.03,0.90,1.10)),dot(c,vec3f(0.85,0.04,0.65)),dot(c,vec3f(1.10,0.50,0.04)));
}
fn dropEnvironment(incoming: vec3f) -> vec3f {
    let direction=normalize(incoming);
    return sample_repeat2(dropSky,vec2f(atan2(direction.y,direction.x)/(2.0*DROP_PI),
        acos(clamp(direction.z,-1.0,1.0))/DROP_PI)).rgb;
}
@vertex fn dropletVertex(@builtin(vertex_index) id: u32,
    @builtin(instance_index) instance: u32) -> DropletOut {
    let corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(-1.0,1.0),
        vec2f(-1.0,1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0));
    let u=dropUniforms;
    let d=visibleDrops[instance];
    let p=d.position.xyz+d.position.w*1.12*(u.right.xyz*corners[id].x+u.up.xyz*corners[id].y);
    let relative=p-u.eye.xyz;
    let distance=dot(relative,u.forward.xyz);
    let clip=vec4f(dot(relative,u.right.xyz)/u.right.w,
        dot(relative,u.up.xyz)/u.up.w,distance*0.5,distance);
    return DropletOut(clip,d.position,d.pigment.rgb);
}
@fragment fn dropletFragment(input: DropletOut) -> @location(0) vec4f {
    let u=dropUniforms;
    let screen=vec2f(input.position.x*u.viewport.z*2.0-1.0,1.0-input.position.y*u.viewport.w*2.0);
    let ray=normalize(u.forward.xyz+screen.x*u.right.w*u.right.xyz+screen.y*u.up.w*u.up.xyz);
    let relative=u.eye.xyz-input.sphere.xyz;
    let b=dot(relative,ray);
    let h=b*b-dot(relative,relative)+input.sphere.w*input.sphere.w;
    // Derivatives must execute before any nonuniform discard in WGSL.
    let edgeWidth=max(fwidth(h),1e-9);
    if (h <= 0.0) { discard; }
    let chord=2.0*sqrt(max(h,0.0));
    let distance=-b-0.5*chord;
    if (distance <= 0.0) { discard; }
    let point=u.eye.xyz+ray*distance;
    let waterHeight=sample_clamp2(dropWater,point.xy*0.5+vec2f(0.5)/vec2f(textureDimensions(dropWater))).x;
    if (point.z < waterHeight) { discard; }
    let n=normalize(point-input.sphere.xyz);
    let f=dropFresnel(-dot(ray,n));
    let absorption=dropPigmentTau(input.pigment*65.0);
    let scattering=dropScattering(input.pigment*65.0);
    let sigma=absorption+scattering;
    let transmission=exp(-sigma*chord);
    let reflected=dropEnvironment(reflect(ray,n));
    let refracted=dropEnvironment(refract(ray,n,1.0/1.333));
    let illumination=select(u.illumination.rgb,vec3f(1.0,0.98,1.12),u.illumination.w > 0.5);
    var radiance=f*reflected+(vec3f(1.0)-f)*(refracted*transmission+
        illumination*scattering/max(sigma,vec3f(1e-7))*(vec3f(1.0)-transmission));
    let halfway=normalize(normalize(u.light.xyz)-ray);
    let highlight=pow(max(0.0,dot(n,halfway)),140.0)*3.0;
    radiance += dropFresnel(dot(-ray,halfway))*dropEnvironment(u.light.xyz)*highlight;
    return vec4f(clamp(radiance,vec3f(0.0),vec3f(64000.0)),smoothstep(0.0,edgeWidth,h));
}
