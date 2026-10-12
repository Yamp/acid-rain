// Reservoirs contain only raw, unwarped scene-linear HDR. Distortions never feed
// back into the reservoirs; the output attachment performs the sole sRGB encode.
@group(0) @binding(0) var<uniform> retinaUniforms: Uniforms;
@group(0) @binding(1) var<uniform> retinaState: vec4f;
@group(0) @binding(2) var retinaCurrent: texture_2d<f32>;
@group(0) @binding(3) var retinaPrevious: texture_2d<f32>;
@group(0) @binding(4) var retinaExposure: texture_2d<f32>;
@group(0) @binding(5) var retinaFast: texture_2d<f32>;
@group(0) @binding(6) var retinaMiddle: texture_2d<f32>;
@group(0) @binding(7) var retinaSlow: texture_2d<f32>;
@group(0) @binding(8) var retinaNext: texture_storage_2d<rgba16float,write>;
@group(0) @binding(9) var retinaNextMiddle: texture_storage_2d<rgba16float,write>;
@group(0) @binding(10) var retinaNextSlow: texture_storage_2d<rgba16float,write>;
struct RetinaUpdateParameters { attack: vec4f, release: vec4f }
@group(0) @binding(11) var<uniform> retinaUpdate: RetinaUpdateParameters;
const RETINA_PI: f32=3.14159265358979323846;
const LUMA: vec3f=vec3f(0.2126,0.7152,0.0722);
struct RetinaVertexOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}
@vertex fn waterVertex(@builtin(vertex_index) id: u32) -> RetinaVertexOut {
    let p=vec2f(select(-1.0,3.0,id == 1u),select(-1.0,3.0,id == 2u));
    return RetinaVertexOut(vec4f(p,0.0,1.0),vec2f(p.x*0.5+0.5,0.5-p.y*0.5));
}
fn toneMap(hdr: vec3f, exposure: f32, u: Uniforms) -> vec3f {
    let gain=u.gain.rgb/max(dot(u.gain.rgb,LUMA),1e-6);
    var rgb=max(hdr,vec3f(0.0))*gain*exposure*u.optics.z;
    let y=dot(rgb,LUMA);
    let saturation=select(clamp(u.optics.w,0.0,2.0),1.15,u.illumination.w > 0.5);
    rgb=vec3f(y)+(rgb-vec3f(y))*saturation;
    let minimum=min(rgb.r,min(rgb.g,rgb.b));
    if (minimum < 0.0) { rgb=max(vec3f(0.0),vec3f(y)+(rgb-vec3f(y))*y/max(y-minimum,1e-7)); }
    let peak=max(rgb.r,max(rgb.g,rgb.b));
    if (peak < 1e-10) { return vec3f(0.0); }
    var tone=min(0.9995,peak*(2.51*peak+0.03)/(peak*(2.43*peak+0.59)+0.14));
    tone=clamp(tone+(u.gain.w-1.0)*tone*(1.0-tone)*(tone-0.18),0.0,0.9995);
    return rgb*(tone/peak);
}
// Exact area-weighted restriction of piecewise-constant radiance. The complete
// footprint is covered at every scale, including fractional pixel boundaries.
// Total work is O(source pixels + output pixels); tiny glints cannot fall into
// gaps between a fixed set of downsampling taps.
fn retinaAreaAverage(current: texture_2d<f32>, p: vec2u, outputSize: vec2u) -> vec3f {
    let size=textureDimensions(current);
    let scale=vec2f(size)/vec2f(outputSize);
    let lo=vec2f(p)*scale;
    let hi=min(vec2f(p+vec2u(1))*scale,vec2f(size));
    let first=min(vec2u(floor(lo)),size-vec2u(1));
    let end=min(vec2u(ceil(hi)),size);
    var sum=vec3f(0.0);
    for (var y=first.y; y < end.y; y += 1u) {
        let wy=max(0.0,min(hi.y,f32(y+1u))-max(lo.y,f32(y)));
        for (var x=first.x; x < end.x; x += 1u) {
            let wx=max(0.0,min(hi.x,f32(x+1u))-max(lo.x,f32(x)));
            sum += max(textureLoad(current,vec2i(i32(x),i32(y)),0).rgb,vec3f(0.0))*(wx*wy);
        }
    }
    return sum/max((hi.x-lo.x)*(hi.y-lo.y),1e-12);
}
fn retinaIntegrate(now: vec3f, old: vec3f, attack: f32, release: f32) -> vec3f {
    if (!finite3(old)) { return now; }
    let alpha=select(release,attack,dot(now,LUMA) > dot(old,LUMA));
    return mix(old,now,clamp(alpha,0.0,1.0));
}
// All layers share the source filter. attack.w is prior-valid; the RGB alpha
// is chosen by luminance and shared across channels to preserve captured hue.
@compute @workgroup_size(8,8,1) fn retinaHistoryUpdate(@builtin(global_invocation_id) id: vec3u) {
    let p=id.xy;
    let size=textureDimensions(retinaNext);
    if (any(p >= size)) { return; }
    let now=retinaAreaAverage(retinaCurrent,p,size);
    if (retinaUpdate.attack.w <= 0.0) {
        textureStore(retinaNext,vec2i(p),vec4f(now,1.0));
        textureStore(retinaNextMiddle,vec2i(p),vec4f(now,1.0));
        textureStore(retinaNextSlow,vec2i(p),vec4f(now,1.0));
        return;
    }
    let a=retinaIntegrate(now,textureLoad(retinaFast,vec2i(p),0).rgb,retinaUpdate.attack.x,retinaUpdate.release.x);
    let b=retinaIntegrate(now,textureLoad(retinaMiddle,vec2i(p),0).rgb,retinaUpdate.attack.y,retinaUpdate.release.y);
    let c=retinaIntegrate(now,textureLoad(retinaSlow,vec2i(p),0).rgb,retinaUpdate.attack.z,retinaUpdate.release.z);
    textureStore(retinaNext,vec2i(p),vec4f(a,1.0));
    textureStore(retinaNextMiddle,vec2i(p),vec4f(b,1.0));
    textureStore(retinaNextSlow,vec2i(p),vec4f(c,1.0));
}
fn retinaHistoryChromaRetention(now: vec3f, old: vec3f) -> f32 {
    let peak=max(now.x,max(now.y,now.z));
    let oldPeak=max(old.x,max(old.y,old.z));
    let saturation=(peak-min(now.x,min(now.y,now.z)))/max(peak,1e-6);
    let oldSaturation=(oldPeak-min(old.x,min(old.y,old.z)))/max(oldPeak,1e-6);
    let protection=smoothstep(0.04,0.55,saturation)*smoothstep(0.005,0.045,peak)
        *smoothstep(0.015,0.06,oldPeak/max(peak,0.001));
    let required=max(saturation,0.001);
    return mix(1.0,smoothstep(0.25*required,0.8*required,oldSaturation),protection);
}
fn historySample(current: texture_2d<f32>, previous: texture_2d<f32>,
                 uv: vec2f, u: Uniforms) -> vec3f {
    let now=sample_mirror2(current,uv).rgb;
    if (u.post.z <= 0.0) { return now; }
    let old=sample_mirror2(previous,uv).rgb;
    if (!finite3(old)) { return now; }
    let chroma=dot(normalize(now+vec3f(0.001)),normalize(old+vec3f(0.001)));
    return mix(now,old,clamp(u.post.z,0.0,1.0)*smoothstep(0.6,0.95,chroma)
        *retinaHistoryChromaRetention(now,old));
}
// Post state=(active wall effect clock,subjective intensity0..2,historiesValid,0).
fn retinaDose(u: Uniforms, state: vec4f) -> f32 {
    if (u.post.x < 0.5 || u.time.z <= 0.0 || state.y <= 0.0) { return 0.0; }
    let t=state.x;
    let envelope=0.72+0.17*sin(t*0.19+0.7)+0.11*sin(t*0.071-1.2);
    return clamp(state.y,0.0,2.0)*clamp(u.time.z*0.25,0.0,2.0)*envelope;
}
fn retinaChannelOffset(uv: vec2f, channel: u32, u: Uniforms, state: vec4f, dose: f32) -> vec2f {
    let phase=f32(channel)*(2.0*RETINA_PI/3.0);
    let t=state.x;
    let field=0.19*sin(uv.y*6.0+t*0.09)+0.11*sin(uv.x*5.0-t*0.043);
    let angle=t*(0.13+0.009*f32(channel))+phase+field;
    let breathing=0.675+0.325*sin(t*0.12+phase+0.23*sin(uv.x*4.0+uv.y*3.0));
    let radius=max(u.time.w,0.0)*breathing*smoothstep(0.0,1.0,dose);
    return vec2f(cos(angle),sin(angle))*radius;
}
fn retinaTemporal(current: texture_2d<f32>, previous: texture_2d<f32>,
    fast: texture_2d<f32>, middle: texture_2d<f32>, slow: texture_2d<f32>,
    uv: vec2f, u: Uniforms, state: vec4f, dose: f32) -> vec3f {
    let now=historySample(current,previous,uv,u);
    if (state.z <= 0.0 || dose <= 0.0) { return now; }
    let pixel=u.viewport.zw*u.post.w;
    let t=state.x;
    let bend=vec2f(sin(uv.y*7.0+t*0.17),cos(uv.x*6.0-t*0.13));
    let amplitude=(0.5+0.45*max(u.time.w,0.0))*min(dose,2.0);
    var a=sample_mirror2(fast,uv+pixel*amplitude*(0.35*bend+vec2f(sin(t*0.31),cos(t*0.23))*0.35)).rgb;
    var b=sample_mirror2(middle,uv+pixel*amplitude*(0.60*bend+vec2f(cos(t*0.19+0.9),sin(t*0.27))*0.6)).rgb;
    var c=sample_mirror2(slow,uv+pixel*amplitude*(-0.8*bend+vec2f(sin(t*0.11+2.0),cos(t*0.17))*0.75)).rgb;
    if (!finite3(a)) { a=now; }
    if (!finite3(b)) { b=now; }
    if (!finite3(c)) { c=now; }
    let direction=normalize(now+vec3f(0.001));
    var affinity=smoothstep(vec3f(0.35),vec3f(0.92),vec3f(
        dot(direction,normalize(a+vec3f(0.001))),dot(direction,normalize(b+vec3f(0.001))),
        dot(direction,normalize(c+vec3f(0.001)))));
    affinity *= vec3f(retinaHistoryChromaRetention(now,a),retinaHistoryChromaRetention(now,b),
                      retinaHistoryChromaRetention(now,c));
    let weights=vec3f(0.12,0.11,0.09)*min(dose,1.7)*affinity;
    return now*(1.0-dot(weights,vec3f(1.0)))+a*weights.x+b*weights.y+c*weights.z;
}
fn retinaChroma(color: vec3f, contrast: f32, uv: vec2f, state: vec4f, dose: f32) -> vec3f {
    let y=dot(color,LUMA);
    if (y <= 1e-8) { return color; }
    let chroma=color-vec3f(y);
    let t=state.x;
    let p=(uv-vec2f(0.5))*8.0;
    var field=sin(p.x+sin(p.y*1.37+t*0.11))*cos(p.y*0.91-t*0.087);
    field += 0.35*sin(p.x*2.3-p.y*1.7+t*0.061);
    let amount=min(dose,2.0)*clamp(contrast*4.0,0.0,1.0);
    let angle=0.25*amount*field;
    let axis=normalize(LUMA);
    let rotated=(chroma*cos(angle)+cross(axis,chroma)*sin(angle))
        *(1.0+0.08*amount*sin(field*1.8+t*0.07));
    let minimum=min(rotated.x,min(rotated.y,rotated.z));
    var scale=1.0;
    if (minimum < -y) { scale=y/max(-minimum,1e-8); }
    return max(vec3f(y)+rotated*scale,vec3f(0.0));
}
struct DiffractionResult { color: vec3f, contrast: f32 }
fn retinaDiffraction(current: texture_2d<f32>, center: vec3f, uv: vec2f,
                     u: Uniforms, state: vec4f, dose: f32) -> DiffractionResult {
    let pixel=u.viewport.zw*u.post.w;
    var inner=vec3f(0.0);
    var outer=vec3f(0.0);
    let angle=0.24+0.07*sin(state.x*0.083);
    for (var i=0u; i < 4u; i += 1u) {
        let a=angle+f32(i)*RETINA_PI*0.5;
        let direction=vec2f(cos(a),sin(a))*pixel;
        let near=sample_mirror2(current,uv+direction*3.5).rgb;
        let far=sample_mirror2(current,uv+direction*10.5).rgb;
        inner += near*0.25;
        outer += far*0.25;
    }
    let wavelengths=vec3f(650.0,550.0,460.0)/550.0;
    let outerWeight=(36.0*wavelengths*wavelengths-vec3f(12.25))/98.0;
    let scattered=mix(inner,outer,outerWeight);
    let strength=0.13*min(dose,1.7);
    let contrast=length(center-inner)/max(0.05,length(center)+length(inner));
    return DiffractionResult(mix(center,scattered,strength),contrast);
}
@fragment fn postFragment(input: RetinaVertexOut) -> @location(0) vec4f {
    let u=retinaUniforms;
    let state=retinaState;
    let pupil=textureLoad(retinaExposure,vec2i(0),0).r;
    let dose=retinaDose(u,state);
    var rgb=vec3f(0.0);
    if (dose <= 0.0) { rgb=historySample(retinaCurrent,retinaPrevious,input.uv,u); }
    else {
        let pixel=u.viewport.zw*u.post.w;
        for (var c=0u; c < 3u; c += 1u) {
            let uv=input.uv+retinaChannelOffset(input.uv,c,u,state,dose)*pixel;
            rgb[c]=retinaTemporal(retinaCurrent,retinaPrevious,retinaFast,retinaMiddle,retinaSlow,uv,u,state,dose)[c];
        }
        let diffraction=retinaDiffraction(retinaCurrent,rgb,input.uv,u,state,dose);
        rgb=retinaChroma(diffraction.color,diffraction.contrast,input.uv,state,dose);
    }
    return vec4f(clamp(toneMap(rgb,pupil,u),vec3f(0.0),vec3f(1.0)),1.0);
}
