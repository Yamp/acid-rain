@group(0) @binding(0) var<uniform> exposureUniforms: Uniforms;
@group(0) @binding(1) var exposureCurrent: texture_2d<f32>;
@group(0) @binding(2) var exposurePrevious: texture_2d<f32>;
@group(0) @binding(3) var exposureResult: texture_storage_2d<r32float,write>;
const EXPOSURE_LUMA: vec3f=vec3f(0.2126,0.7152,0.0722);
@compute @workgroup_size(1) fn adaptExposure(@builtin(global_invocation_id) p: vec3u) {
    if (any(p != vec3u(0u))) { return; }
    let u=exposureUniforms;
    var sum=0.0;
    for (var y=0u; y < 18u; y += 1u) {
        for (var x=0u; x < 32u; x += 1u) {
            let rgb=sample_clamp2(exposureCurrent,(vec2f(f32(x),f32(y))+vec2f(0.5))/vec2f(32.0,18.0)).rgb;
            sum += log(max(0.001,dot(rgb,EXPOSURE_LUMA)));
        }
    }
    var targetExposure=clamp(0.25/exp(sum/576.0),0.4,6.4);
    if (u.illumination.w > 0.5) { targetExposure=0.85; }
    var old=textureLoad(exposurePrevious,vec2i(0),0).r;
    if (!finite1(old) || old < 0.4) { old=1.0; }
    let rate=select(1.1,4.5,targetExposure < old);
    let value=mix(old,targetExposure,1.0-exp(-rate*clamp(u.time.y,0.0,0.1)));
    textureStore(exposureResult,vec2i(0),vec4f(value,0.0,0.0,1.0));
}
