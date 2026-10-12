import Foundation
import Metal
import simd
import CryptoKit

struct OpticalSamplingFailure: Error, CustomStringConvertible { let description: String }

/// Production Metal functions, checked against smoothness invariants and an
/// independent pixel-area integral of dielectric GGX (not another AA method).
@main struct OpticalSamplingTests {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw OpticalSamplingFailure(description: message) }
    }
    static func texture(_ device: MTLDevice, _ width: Int, _ height: Int,
                        _ values: [SIMD4<Float>]) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
            width: width, height: height, mipmapped: false)
        d.storageMode = .shared; d.usage = [.shaderRead, .renderTarget]
        guard let result = device.makeTexture(descriptor: d) else { throw OpticalSamplingFailure(description: "texture") }
        values.withUnsafeBytes { result.replace(region: MTLRegionMake2D(0,0,width,height), mipmapLevel: 0,
            withBytes: $0.baseAddress!, bytesPerRow: width*16) }
        return result
    }
    static func constant(_ device: MTLDevice, _ color: SIMD4<Float>) throws -> MTLTexture {
        try texture(device, 8, 8, Array(repeating: color, count: 64))
    }
    static func skySamples(_ device: MTLDevice, _ queue: MTLCommandQueue, _ library: MTLLibrary,
                           _ sky: MTLTexture, _ points: [SIMD4<Float>],
                           function: String = "sampleEnvironmentContract", count: Int? = nil,
                           resultStride: Int = 1) throws -> [SIMD4<Float>] {
        let invocationCount=count ?? points.count
        let resultCount=invocationCount*resultStride
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: function)!)
        let input = points.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let output = device.makeBuffer(length: resultCount*16, options: .storageModeShared)!
        let cmd = queue.makeCommandBuffer()!; let encoder = cmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline); encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1); encoder.setTexture(sky, index: 0)
        var u=ARUniforms();u.world=SIMD4(0,-0.3,0.3,0)
        encoder.setBytes(&u,length:MemoryLayout<ARUniforms>.stride,index:2)
        encoder.dispatchThreads(MTLSize(width: invocationCount,height:1,depth:1), threadsPerThreadgroup: MTLSize(width:1,height:1,depth:1))
        encoder.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        try require(cmd.status == .completed, "environment GPU command")
        return Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: SIMD4<Float>.self, capacity: resultCount), count: resultCount))
    }
    static func fresnel(_ cosine: Double, _ ior: Double) -> Double {
        let transmitted = sqrt(1 - (1-cosine*cosine)/(ior*ior))
        let s = (cosine-ior*transmitted)/(cosine+ior*transmitted)
        let p = (ior*cosine-transmitted)/(ior*cosine+transmitted)
        return 0.5*(s*s+p*p)
    }
    static func pointRadiance(_ x: Double, _ y: Double, _ ior: Double) -> Double {
        let view = simd_normalize(SIMD3<Double>(-x,-y,1))
        return pointRadiance(SIMD3<Double>(0,0,1),view,SIMD3<Double>(0,0,1),ior)
    }
    static func pointRadiance(_ n: SIMD3<Double>, _ view: SIMD3<Double>, _ light: SIMD3<Double>, _ ior: Double) -> Double {
        let ndv=simd_dot(n,view),ndl=simd_dot(n,light)
        if ndv <= 0 || ndl <= 0 { return 0 }
        let half=simd_normalize(view+light)
        let a2=0.0025,ndh=max(0,simd_dot(n,half))
        let d = a2 / (Double.pi * pow(ndh*ndh*(a2-1)+1,2))
        let smith=2*ndv*ndl/(ndl*sqrt(a2+(1-a2)*ndv*ndv)+ndv*sqrt(a2+(1-a2)*ndl*ndl))
        return fresnel(simd_dot(view,half),ior)*d*smith*0.18/(4*ndv)
    }
    static func renderWater(_ device: MTLDevice, _ queue: MTLCommandQueue, _ pipeline: MTLRenderPipelineState,
                            _ textures: [MTLTexture], _ volume: MTLTexture, _ target: MTLTexture,
                            _ shift: Float) throws -> [SIMD4<Float>] {
        var u = ARUniforms()
        let side = Float(target.width)
        u.viewport = SIMD4(side,side,1/side,1/side)
        u.eye = SIMD4(1,1,1,1); u.forward = SIMD4(shift,0,-1,0)
        u.right = SIMD4(1,0,0,0.9); u.up = SIMD4(0,1,0,0.9)
        u.light = SIMD4(0,0,1,0); u.world = SIMD4(0,-0.001,0.001,0)
        u.optics = SIMD4(1,0.2,1,1); u.post = .zero; u.illumination = .zero
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let cmd = queue.makeCommandBuffer()!; let encoder = cmd.makeRenderCommandEncoder(descriptor: pass)!
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBytes(&u, length: MemoryLayout<ARUniforms>.stride, index:0)
        var bedWorld = SIMD4<Float>.zero
        encoder.setFragmentBytes(&bedWorld, length:16,index:2)
        for (index,value) in textures.enumerated() { encoder.setFragmentTexture(value,index:index) }
        encoder.setFragmentTexture(volume,index:5)
        encoder.setFragmentTexture(textures[0],index:6)
        encoder.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:3)
        encoder.endEncoding();cmd.commit();cmd.waitUntilCompleted()
        try require(cmd.status == .completed, "production water GPU command: \(String(describing:cmd.error))")
        var values = Array(repeating:SIMD4<Float>.zero,count:target.width*target.height)
        values.withUnsafeMutableBytes { target.getBytes($0.baseAddress!,bytesPerRow:target.width*16,
            from:MTLRegionMake2D(0,0,target.width,target.height),mipmapLevel:0) }
        return values
    }
    static func run() throws {
        let destination = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "artifacts/smooth-optics/optical-tests/current")
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        var source = ""
        for name in ["Caustics", "BedLife", "Water"] {
            let path=name == "Water" && CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "native/Sources/\(name).metal"
            source += try String(contentsOfFile:path,encoding:.utf8)+"\n"
        }
        let shaderHash=SHA256.hash(data:Data(source.utf8)).map {String(format:"%02x",$0)}.joined()
        let refinementCall=source.contains("refineWaterIntersectionData") ? """
            WaterIntersection result=refineWaterIntersectionData(water,p[k].xyz,ray,p[k+2].xyz);
            float3 point=result.point; float2 slope=result.slope;
        """ : """
            float3 point=refineWaterIntersection(water,p[k].xyz,ray,p[k+2].xyz);
            float2 slope=waterSurface(water,point.xy).yz;
        """
        source += """
        kernel void sampleRefinementContract(constant float4 *p [[buffer(0)]],
            device float4 *out [[buffer(1)]],texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            uint k=i*3; float3 ray=normalize(p[k+1].xyz);
            \(refinementCall)
            float2 reference=waterSurface(water,point.xy).yz;
            out[i*2]=float4(point,length(slope-reference));
            out[i*2+1]=float4(slope,reference);
        }
        """
        source += """
        kernel void sampleEnvironmentContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],texture2d<float> sky [[texture(0)]],uint i [[thread_position_in_grid]]) {
            float theta=points[i].x*2*PI, phi=points[i].y*PI;
            float3 direction=float3(cos(theta)*sin(phi),sin(theta)*sin(phi),cos(phi));
            out[i]=float4(environment(sky,direction),1);
        }
        kernel void sampleNormalContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            out[i]=float4(surfaceNormal(water,points[i].xy),1);
        }
        kernel void sampleSurfaceContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            out[i]=float4(waterSurface(water,points[i].xy),1);
        }
        kernel void sampleIntersectionContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],constant Uniforms &u [[buffer(2)]],
            texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            float3 eye=float3(points[i].xy,.4),ray=normalize(float3(points[i].zw,-1));
            float3 hit=intersectWater(water,eye,ray,u);
            out[i]=float4(hit,hit.z-waterSurface(water,hit.xy).x);
        }
        kernel void sampleSecondaryContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],constant Uniforms &u [[buffer(2)]],
            texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            float3 hit=0;
            bool found=secondaryHit(water,points[i].xyz,normalize(float3(1,0,points[i].w)),hit,u);
            out[i]=float4(hit,found ? 1.f : 0.f);
        }
        kernel void sampleDegenerateContract(constant float4 *points [[buffer(0)]],
            device float4 *out [[buffer(1)]],constant Uniforms &u [[buffer(2)]],
            texture2d<float> water [[texture(0)]],uint i [[thread_position_in_grid]]) {
            float3 eye=float3(1.25,.5,.21630859375),ray=normalize(float3(-1,0,-.625));
            float3 hit=intersectWater(water,eye,ray,u);
            out[0]=float4(hit,hit.z-waterSurface(water,hit.xy).x);
            float3 secondary=0;
            bool found=secondaryHit(water,eye,ray,secondary,u);
            out[1]=float4(secondary,found ? secondary.z-waterSurface(water,secondary.xy).x : 1.f);
        }
        kernel void sampleLobeContract(constant float4 *p [[buffer(0)]],
            device float4 *out [[buffer(1)]],uint i [[thread_position_in_grid]]) {
            uint k=i*7;
            out[i]=float4(directHighlightPixel(p[k].xyz,p[k+1].xyz,p[k+2].xyz,
                p[k+3].xyz,p[k+4].xyz,p[k+5].xyz,p[k+6].xyz),1);
        }
        """
        guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {
            throw OpticalSamplingFailure(description:"Metal unavailable")
        }
        let options=MTLCompileOptions(); options.mathMode = .fast
        let library=try device.makeLibrary(source:source,options:options)
        // Stay above hardware bilinear-weight quantization; measure the change
        // in adjacent finite-difference slopes across a reconstruction knot.
        let epsilon:Float=0.04, width=32, knot:Float=8
        let points = [-epsilon,Float(0),epsilon].map { SIMD4<Float>((knot+0.5+$0)/Float(width),0.5,0,0) }
        let pattern=(0..<width*16).map { i -> SIMD4<Float> in
            let value=1+0.3*cos(Float(i%width)*2*Float.pi/8)
            return SIMD4(value,value*0.4,value*0.7,1)
        }
        let samples=try skySamples(device,queue,library,texture(device,width,16,pattern),points)
        let derivativeJump=abs(samples[2].x-2*samples[1].x+samples[0].x)/epsilon
        let colors=[SIMD4<Float>(0,0,0,1),SIMD4<Float>(7,0.2,2,1)]
        var constantError:Float=0
        for color in colors {
            let values=try skySamples(device,queue,library,constant(device,color),points)
            for value in values { for c in 0..<3 {constantError=max(constantError,abs(value[c]-color[c]))} }
        }
        print("Environment derivative jump=\(derivativeJump), constant error=\(constantError)")
        let normalPoints=(0..<61).map { i -> SIMD4<Float> in
            SIMD4(0.3+1.2*Float(i)/60,0.4+0.8*Float((i*17)%61)/60,0,0)
        }
        let affine=(0..<32*32).map { i -> SIMD4<Float> in
            let x=2*Float(i%32)/32,y=2*Float(i/32)/32
            return SIMD4(0.2*x+0.15*x*x-0.1*x*y-0.1*y+0.1*y*y,
                         0.2+0.3*x-0.1*y,-0.1-0.1*x+0.2*y,0)
        }
        let normals=try skySamples(device,queue,library,texture(device,32,32,affine),normalPoints,function:"sampleNormalContract")
        var affineError:Float=0
        for (p,value) in zip(normalPoints,normals) {
            let expected=simd_normalize(SIMD3<Float>(-(0.2+0.3*p.x-0.1*p.y),-(-0.1-0.1*p.x+0.2*p.y),1))
            affineError=max(affineError,simd_length(SIMD3(value.x,value.y,value.z)-expected))
        }
        let polynomial=try skySamples(device,queue,library,texture(device,32,32,affine),normalPoints,function:"sampleSurfaceContract")
        var polynomialError:Float=0
        for (p,value) in zip(normalPoints,polynomial) {
            let expected=SIMD3<Float>(0.2*p.x+0.15*p.x*p.x-0.1*p.x*p.y-0.1*p.y+0.1*p.y*p.y,
                0.2+0.3*p.x-0.1*p.y,-0.1-0.1*p.x+0.2*p.y)
            polynomialError=max(polynomialError,simd_length(SIMD3(value.x,value.y,value.z)-expected))
        }
        let waves=(0..<32*32).map { i -> SIMD4<Float> in
            let x=Float(i%32),y=Float(i/32)
            return SIMD4(0.025*sin(x*2*Float.pi/3)+0.018*cos(y*2*Float.pi/7),0,0,0)
        }
        let waveTexture=try texture(device,32,32,waves)
        var refinementInputs:[SIMD4<Float>]=[]
        for i in 0..<96 {
            let eye=SIMD3<Float>(0.25+1.5*Float(i%12)/11,0.3+1.2*Float(i/12)/7,0.4)
            let ray=simd_normalize(SIMD3<Float>(Float((i*7)%13-6)*0.3,Float((i*11)%9-4)*0.2,-1))
            // Deliberately displaced seeds exercise cell crossings and the
            // bounded iteration limit, in addition to ordinary convergence.
            let seed=eye+ray*(0.15+0.1*Float(i%8))
            refinementInputs += [SIMD4(eye,0),SIMD4(ray,0),SIMD4(seed,0)]
        }
        let refinementValues=try skySamples(device,queue,library,waveTexture,refinementInputs,
            function:"sampleRefinementContract",count:96,resultStride:2)
        var finalSlopeError:Float=0
        for i in 0..<96 {
            finalSlopeError=max(finalSlopeError,refinementValues[i*2].w)
        }
        let refinementJSON=refinementValues.map {[$0.x,$0.y,$0.z,$0.w]}
        try JSONSerialization.data(withJSONObject:refinementJSON).write(to:destination.appendingPathComponent("refinement.json"))
        let knotPoints=(3..<28).map {i -> SIMD4<Float> in
            let x=Float(i)/16
            let y=Float((i*3)%25+3)/16
            return SIMD4(x,y,0,0)
        }
        let knotValues=try skySamples(device,queue,library,waveTexture,knotPoints,function:"sampleSurfaceContract")
        var nodeError:Float=0
        for (p,value) in zip(knotPoints,knotValues) {
            let x=Int((p.x*16).rounded()),y=Int((p.y*16).rounded())
            nodeError=max(nodeError,abs(value.x-waves[y*32+x].x))
        }
        let rays=(0..<80).map {i -> SIMD4<Float> in
            let x:Float=0.3+1.2*Float(i%10)/9
            let y:Float=0.3+Float(i/10)/7
            let dx:Float=Float((i*7)%9-4)*0.12
            let dy:Float=Float((i*3)%7-3)*0.1
            return SIMD4(x,y,dx,dy)
        }
        let hits=try skySamples(device,queue,library,waveTexture,rays,function:"sampleIntersectionContract")
        let maxResidual=hits.map {abs($0.w)}.max()!
        try require(hits.allSatisfy {$0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite},"finite curved-surface intersections")
        let crest=(0..<64).map {i -> SIMD4<Float> in SIMD4(i%8 == 2 || i%8 == 3 ? 0.1 : 0,0,0,0)}
        let crestHit=try skySamples(device,queue,library,texture(device,8,8,crest),[SIMD4(0.5,0.8,0.11,0)],function:"sampleSecondaryContract")[0]
        let steepCrest=crest.map {SIMD4<Float>($0.x*2,0,0,0)}
        let uphillHit=try skySamples(device,queue,library,texture(device,8,8,steepCrest),[SIMD4(0.25,0.8,0.01,0.6)],function:"sampleSecondaryContract")[0]
        let degenerateField=(0..<128*128).map {i -> SIMD4<Float> in
            SIMD4(i%128 <= 65 ? 1/16 : 0,0,0,0)
        }
        let degenerateHits=try skySamples(device,queue,library,texture(device,128,128,degenerateField),[.zero],
            function:"sampleDegenerateContract",count:1,resultStride:2)
        // Unique real root of .5t³-.5t²+.15625t-.0390625=0.
        let expectedCrestX=1.0111863536242938
        let degenerateResidual=degenerateHits.map {abs($0.w)}.max()!
        let degenerateRootError=degenerateHits.map {abs(Double($0.x)-expectedCrestX)}.max()!
        // Independent pixel integrals cover a thin ribbon, oblique reflection,
        // grazing reflection, and physical hemisphere exclusions.
        let up=SIMD3<Float>(0,0,1),zero=SIMD3<Float>.zero
        let oblique=simd_normalize(SIMD3<Float>(0.8,0,0.6))
        let grazing=simd_normalize(SIMD3<Float>(0.98,0,0.2))
        var lobeCases:[[SIMD3<Float>]]=[
            [up,up,up,SIMD3(0.2,0,0),zero,zero,zero],
            [up,up,up,SIMD3(0.14,0,0),SIMD3(0,0.08,0),zero,zero],
            [up,oblique,SIMD3(-oblique.x,0,oblique.z),SIMD3(0.1,0,0),zero,SIMD3(0.01,0,0),zero],
            [up,grazing,SIMD3(-grazing.x,0,grazing.z),SIMD3(0.08,0,0),zero,SIMD3(0.005,0,0),zero],
            [up,up,-up,zero,zero,zero,zero],
            [up,-up,up,zero,zero,zero,zero],
        ]
        for phase in 0..<9 {
            let center=simd_normalize(SIMD3<Float>(0.4*(Float(phase)/8-0.5),0,1))
            lobeCases.append([center,up,up,SIMD3(0.4,0,0),zero,zero,zero])
        }
        lobeCases.append([up,up,up,SIMD3(0.3,0.3,0),SIMD3(-0.3,0.3,0),zero,zero])
        let lobeValues=try skySamples(device,queue,library,constant(device,SIMD4(1,1,1,1)),lobeCases.flatMap {$0.map {SIMD4($0,0)}},
            function:"sampleLobeContract",count:lobeCases.count)
        var lobeErrors:[Double]=[]
        for (index,c) in lobeCases.enumerated() {
            let v=c.map {SIMD3<Double>(Double($0.x),Double($0.y),Double($0.z))}
            var reference=0.0
            for y in 0..<128 {for x in 0..<128 {
                let a=(Double(x)+0.5)/128-0.5,b=(Double(y)+0.5)/128-0.5
                reference += pointRadiance(simd_normalize(v[0]+v[3]*a+v[4]*b),
                    simd_normalize(v[1]+v[5]*a+v[6]*b),v[2],1.333)
            }}
            reference /= 128*128
            let actual=Double(lobeValues[index].y)
            try require(actual.isFinite && actual >= 0,"finite passive local GGX integral")
            lobeErrors.append(reference > 0 ? abs(actual/reference-1) : abs(actual))
        }
        let desc=MTLRenderPipelineDescriptor();desc.vertexFunction=library.makeFunction(name:"waterVertex")
        desc.fragmentFunction=library.makeFunction(name:"waterFragment");desc.colorAttachments[0].pixelFormat = .rgba32Float
        let pipeline=try device.makeRenderPipelineState(descriptor:desc)
        let surface=try constant(device,SIMD4(0,0,0,0)), dye=try constant(device,SIMD4(0,0,0,0.5))
        let black=try constant(device,SIMD4(0,0,0,1)),sky=try constant(device,SIMD4(1,1,1,1))
        let volumeDesc=MTLTextureDescriptor();volumeDesc.textureType = .type3D; volumeDesc.pixelFormat = .rgba32Float
        volumeDesc.width=2;volumeDesc.height=2;volumeDesc.depth=2;volumeDesc.storageMode = .shared;volumeDesc.usage = .shaderRead
        let volume=device.makeTexture(descriptor:volumeDesc)!
        Array(repeating:SIMD4<Float>.zero,count:8).withUnsafeBytes {
            volume.replace(region:MTLRegionMake3D(0,0,0,2,2,2),mipmapLevel:0,slice:0,
                withBytes:$0.baseAddress!,bytesPerRow:32,bytesPerImage:64)
        }
        let side=16, subdivisions=32, phases=8
        let output=try texture(device,side,side,Array(repeating:.zero,count:side*side))
        var error=0.0,referenceEnergy=0.0,actualEnergy=0.0,referenceSquare=0.0
        var phaseEnergy:[Double]=[],phaseEnergyRatio:[Double]=[]
        for phase in 0..<phases {
            let shift=Float(phase)/Float(phases)*1.8/Float(side)
            let pixels=try renderWater(device,queue,pipeline,[surface,dye,black,sky,black],volume,output,shift)
            var total=0.0,referenceTotal=0.0
            for y in 0..<side {for x in 0..<side {
                var expected=0.0
                for sy in 0..<subdivisions {for sx in 0..<subdivisions {
                    let px=(Double(x)+(Double(sx)+0.5)/Double(subdivisions))/Double(side)*1.8-0.9+Double(shift)
                    let py=0.9-(Double(y)+(Double(sy)+0.5)/Double(subdivisions))/Double(side)*1.8
                    expected += pointRadiance(px,py,1.333)
                }}
                expected /= Double(subdivisions*subdivisions)
                let centerX=(Double(x)+0.5)/Double(side)*1.8-0.9+Double(shift)
                let centerY=0.9-(Double(y)+0.5)/Double(side)*1.8
                let background=fresnel(1/sqrt(1+centerX*centerX+centerY*centerY),1.333)
                let actual=Double(pixels[y*side+x].y)-background
                try require(actual.isFinite && actual >= -0.000001,"finite nonnegative water highlight: phase=\(phase) x=\(x) y=\(y) actual=\(actual) pixel=\(pixels[y*side+x]) background=\(background)")
                error += pow(actual-expected,2);referenceSquare += expected*expected
                actualEnergy += actual;referenceEnergy += expected;total += actual;referenceTotal += expected
            }}
            phaseEnergy.append(total)
            phaseEnergyRatio.append(total/referenceTotal)
        }
        let relativeRMSE=sqrt(error/referenceSquare),energyRatio=actualEnergy/referenceEnergy
        let metrics:[String:Any]=["device":device.name,"shader_sha256":shaderHash,"environment_knot_derivative_jump":derivativeJump,
            "affine_normal_max_error":affineError,"varying_normal_relative_errors":lobeErrors,
            "quadratic_saddle_max_error":polynomialError,"node_height_max_error":nodeError,
            "cubic_ray_max_residual_metres":maxResidual,"phase_energy_ratio":phaseEnergyRatio,
            "cubic_crest_secondary_hit":crestHit.w,
            "uphill_crest_secondary_hit":uphillHit.w,
            "final_hit_slope_max_error":finalSlopeError,
            "degenerate_crest_max_residual_metres":degenerateResidual,
            "degenerate_crest_root_max_error_metres":degenerateRootError,
            "constant_color_max_error":constantError,"highlight_pixel_integral_relative_rmse":relativeRMSE,
            "highlight_energy_ratio":energyRatio,"phase_energy":phaseEnergy,"reference_samples_per_pixel":subdivisions*subdivisions]
        let data=try JSONSerialization.data(withJSONObject:metrics,options:[.prettyPrinted,.sortedKeys])
        try data.write(to:destination.appendingPathComponent("metrics.json"))
        print(String(data:data,encoding:.utf8)!)
        try require(derivativeJump < 0.025,"C1 environment reconstruction has a texel-boundary kink: \(derivativeJump)")
        try require(constantError < 0.00002,"constant spectral radiance changed")
        try require(affineError < 0.0002,"affine hydraulic shading slope changed: \(affineError)")
        try require(polynomialError < 0.0002,"quadratic/mixed saddle reconstruction: \(polynomialError)")
        try require(nodeError < 1e-7,"resolved wave-node amplitude changed: \(nodeError)")
        try require(maxResidual < 0.00001,"ray does not lie on shading geometry: \(maxResidual)m")
        try require(crestHit.w == 1,"secondary reflection escaped through cubic crest above lattice maximum")
        try require(uphillHit.w == 1,"uphill reflection escaped through a steeper cubic crest")
        // Fast-math expression scheduling may change the final float rounding;
        // this tolerance is far below the independent surface oracle tolerance.
        try require(finalSlopeError < 5e-7,"cached gradient does not correspond to final returned ray point: \(finalSlopeError)")
        try require(degenerateResidual < 1e-6,"degenerate Newton ray returned off-surface: \(degenerateResidual)m")
        try require(degenerateRootError < 3e-6,"resolved cubic crest root mismatch: \(degenerateRootError)m")
        try require(lobeErrors.max()! < 0.025,"varying-normal/grazing/hemisphere local GGX integral: \(lobeErrors)")
        try require(relativeRMSE < 0.025,"GGX pixel-area integration error: \(relativeRMSE)")
        try require(abs(energyRatio-1) < 0.015,"GGX sweep energy changed: \(energyRatio)")
        try require(phaseEnergyRatio.allSatisfy {abs($0-1)<0.015},"per-phase GGX sweep energy changed: \(phaseEnergyRatio)")
        print("PASS: production environment C1/constant color, production water subpixel GGX integration and energy")
    }
    static func main() {do {try run()} catch {fputs("Optical sampling FAILED: \(error)\n",stderr);exit(1)}}
}
