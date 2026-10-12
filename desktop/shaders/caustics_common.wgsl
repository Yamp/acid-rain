// Shared physical bed intersections for forward caustics and view refraction.
// Requires sample_clamp2 from the renderer's sampling module.
// Interpolating height geometry and analytic normals share this cached surfacePatch.
struct WaterSurfacePatch { rows: array<vec4f,4>, cell: vec2i, scale: vec2f }
fn waterCubicCoefficients(h: vec4f) -> vec4f {
    return vec4f(0.5*(-h.x+3.0*h.y-3.0*h.z+h.w),
        h.x-2.5*h.y+2.0*h.z-0.5*h.w,0.5*(h.z-h.x),h.y);
}
fn waterSurfacePatch(water: texture_2d<f32>, metres: vec2f, extent: f32) -> WaterSurfacePatch {
    let size=vec2i(textureDimensions(water)); let scale=vec2f(size)/extent;
    let cell=vec2i(floor(metres*scale)); var rows: array<vec4f,4>;
    for (var y=0i;y<4i;y+=1i) {
        var row=vec4f(0.0);
        for (var x=0i;x<4i;x+=1i) {
            row[u32(x)]=textureLoad(water,clamp(cell+vec2i(x-1i,y-1i),vec2i(0),size-vec2i(1)),0).x;
        }
        rows[u32(y)]=waterCubicCoefficients(row);
    }
    return WaterSurfacePatch(rows,cell,scale);
}
fn waterSurfaceValue(surfacePatch: WaterSurfacePatch, metres: vec2f) -> vec3f {
    let t=metres*surfacePatch.scale-vec2f(surfacePatch.cell); var value=vec4f(0.0); var dx=vec4f(0.0);
    for (var y=0u;y<4u;y+=1u) {
        let a=surfacePatch.rows[y];
        value[y]=((a.x*t.x+a.y)*t.x+a.z)*t.x+a.w;
        dx[y]=(3.0*a.x*t.x+2.0*a.y)*t.x+a.z;
    }
    let a=waterCubicCoefficients(value); let b=waterCubicCoefficients(dx);
    return vec3f(((a.x*t.y+a.y)*t.y+a.z)*t.y+a.w,
        (((b.x*t.y+b.y)*t.y+b.z)*t.y+b.w)*surfacePatch.scale.x,
        ((3.0*a.x*t.y+2.0*a.y)*t.y+a.z)*surfacePatch.scale.y);
}
fn waterSurface(water: texture_2d<f32>, metres: vec2f, extent: f32) -> vec3f {
    return waterSurfaceValue(waterSurfacePatch(water,metres,extent),metres);
}
fn causticsField(field: texture_2d<f32>, position: vec2f, extent: f32) -> vec4f {
    let dimensions = vec2f(textureDimensions(field));
    return sample_clamp2(field, position / extent + vec2f(0.5) / dimensions);
}

fn causticsFresnel(cosine: f32, ior: f32) -> f32 {
    let c = clamp(cosine, 0.0, 1.0);
    let t = sqrt(max(0.0, 1.0 - (1.0 - c * c) / (ior * ior)));
    let rs = (c - ior * t) / max(c + ior * t, 1e-7);
    let rp = (ior * c - t) / max(ior * c + t, 1e-7);
    return 0.5 * (rs * rs + rp * rp);
}

fn causticsBed(water: texture_2d<f32>, depth: texture_2d<f32>, point: vec2f, extent: f32) -> f32 {
    return causticsField(water, point, extent).x - max(0.001, causticsField(depth, point, extent).a);
}

fn causticsBedResidual(water: texture_2d<f32>, depth: texture_2d<f32>,
                      origin: vec3f, ray: vec3f, distance: f32, extent: f32) -> f32 {
    let p = origin + distance * ray;
    return p.z - causticsBed(water, depth, p.xy, extent);
}

// The edge bed height continues outside the resident surface as a local plane.
// Receiver coverage handles boundary rays independently of irradiance.
fn causticsBedDistance(water: texture_2d<f32>, depth: texture_2d<f32>,
                      origin: vec3f, ray: vec3f, localDepth: f32, extent: f32) -> f32 {
    var near = 0.0;
    var a = localDepth;
    var far = max(0.001, localDepth / max(-ray.z, 0.001));
    var b = causticsBedResidual(water, depth, origin, ray, far, extent);
    for (var i = 0u; i < 4u && b > 0.0; i += 1u) {
        far *= 2.0;
        b = causticsBedResidual(water, depth, origin, ray, far, extent);
    }
    if b > 0.001 { return -1.0; }
    for (var i = 0u; i < 10u; i += 1u) {
        if abs(b) < 0.00001 { return far; }
        let t = mix(near, far, clamp(a / max(a - b, 1e-8), 0.03, 0.97));
        let residual = causticsBedResidual(water, depth, origin, ray, t, extent);
        if abs(residual) < 0.00001 { return t; }
        if residual > 0.0 {
            near = t;
            a = residual;
        } else {
            far = t;
            b = residual;
        }
    }
    return 0.5 * (near + far);
}

// xyz is the bed hit and w its physical path length, <= 0 for a missing hit.
// Directions are normalized, and local world patches span two metres.
fn causticsBedHit(water: texture_2d<f32>, depth: texture_2d<f32>,
                  source: vec3f, direction: vec3f) -> vec4f {
    if direction.z >= -0.001 { return vec4f(source, -1.0); }
    let localDepth = source.z - causticsBed(water, depth, source.xy, 2.0);
    if localDepth <= 0.0 { return vec4f(source, 0.0); }
    let distance = causticsBedDistance(water, depth, source, direction, localDepth, 2.0);
    return vec4f(source + direction * max(distance, 0.0), distance);
}

// Coverage is geometric; using brightness here would fill real dark troughs.
fn causticsCoverage(receiver: vec2f, depth: f32, light: vec3f) -> f32 {
    let l = normalize(light);
    let cosT = sqrt(max(0.001, 1.0 - dot(l.xy, l.xy) / (1.333 * 1.333)));
    let source = receiver + l.xy * max(depth, 0.0) / (1.333 * cosT);
    let edge = min(min(source.x, source.y), min(2.0 - source.x, 2.0 - source.y));
    return smoothstep(0.04, 0.20, edge);
}
