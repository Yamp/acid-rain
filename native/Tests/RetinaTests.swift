import Foundation
import Metal
import simd
import CryptoKit

private struct RetinaFailure: Error, CustomStringConvertible { let description: String }

/// Real GPU contracts for the subjective post layer, independent of water optics.
@main struct RetinaTests {
    static let width = 128, height = 96
    static var checks = [String]()
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw RetinaFailure(description: message) }
    }
    static func defaults() -> ARUniforms {
        var u = ARUniforms()
        u.viewport = SIMD4(Float(width),Float(height),1/Float(width),1/Float(height))
        u.time = SIMD4(17,1/60,4,2)
        u.optics = SIMD4(1,0.2,1,1); u.gain = SIMD4(1,1,1,1)
        u.post = SIMD4(1,2,0,1)
        return u
    }
    static func texture(_ device: MTLDevice, width: Int = width, height: Int = height,
                        values: [SIMD4<Float>]? = nil, color: SIMD4<Float> = .zero) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:width,height:height,mipmapped:false)
        d.storageMode = .shared; d.usage = [.shaderRead,.shaderWrite,.renderTarget]
        guard let t = device.makeTexture(descriptor:d) else { throw RetinaFailure(description:"texture allocation") }
        let pixels = values ?? Array(repeating:color,count:width*height)
        pixels.withUnsafeBytes { t.replace(region:MTLRegionMake2D(0,0,width,height),mipmapLevel:0,withBytes:$0.baseAddress!,bytesPerRow:width*16) }
        return t
    }
    static func read(_ t: MTLTexture) -> [SIMD4<Float>] {
        var values = [SIMD4<Float>](repeating:.zero,count:t.width*t.height)
        values.withUnsafeMutableBytes { t.getBytes($0.baseAddress!,bytesPerRow:t.width*16,from:MTLRegionMake2D(0,0,t.width,t.height),mipmapLevel:0) }
        return values
    }
    static func difference(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
        zip(a,b).reduce(Float(0)) { $0 + abs($1.0.x-$1.1.x)+abs($1.0.y-$1.1.y)+abs($1.0.z-$1.1.z) } / Float(a.count*3)
    }
    static func main() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw RetinaFailure(description:"Metal unavailable") }
        let sourcePaths = ["Cloud","Caustics","BedLife","Water"].map { "native/Sources/\($0).metal" }
        let sources = try sourcePaths.map { try String(contentsOfFile:$0,encoding:.utf8) }
        let production = sources.joined(separator:"\n")
        let probes = """
        kernel void retinaContracts(constant Uniforms &u [[buffer(0)]], device float4 *out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
            if (tid >= 1024) return;
            float2 uv = float2(float(tid%32)/31,float((tid/32)%8)/7);
            float4 state = float4(float(tid)*.173f,2,1,0);
            float dose = retinaDose(u,state);
            float2 r = retinaChannelOffset(uv,0,u,state,dose);
            float2 g = retinaChannelOffset(uv,1,u,state,dose);
            float2 b = retinaChannelOffset(uv,2,u,state,dose);
            out[tid*5] = float4(r,g);
            out[tid*5+1] = float4(b,dose,retinaDose(u,state+float4(.001f,0,0,0)));
            out[tid*5+2] = float4(retinaChroma(float3(.6f),1,uv,state,dose),1);
            out[tid*5+3] = float4(retinaChroma(float3(.9f,.08f,.7f),1,uv,state,dose),1);
            out[tid*5+4] = float4(retinaChannelOffset(uv,0,u,state+float4(.001f,0,0,0),dose),0,0);
        }
        kernel void retinaPSFContracts(texture2d<float> current [[texture(0)]], texture2d<float,access::write> result [[texture(1)]],
            constant Uniforms &u [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
            if (p.x >= result.get_width() || p.y >= result.get_height()) return;
            float2 uv = (float2(p)+.5f)/float2(result.get_width(),result.get_height());
            float contrast;
            float3 rgb = retinaDiffraction(current,current.sample(linearClamp,uv,level(0)).rgb,uv,u,float4(17,1,0,0),1,contrast);
            result.write(float4(rgb,1),p);
        }
        """
        let source = production+"\n"+probes
        let options = MTLCompileOptions(); options.mathMode = .fast
        let library = try device.makeLibrary(source:source,options:options)
        func pipeline(_ fragment: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name:"waterVertex"); d.fragmentFunction = library.makeFunction(name:fragment)
            d.colorAttachments[0].pixelFormat = .rgba32Float
            return try device.makeRenderPipelineState(descriptor:d)
        }
        let post = try pipeline("postFragment")
        let exposure = try texture(device,width:1,height:1,color:SIMD4(1,0,0,0))
        let output = try texture(device)
        func render(_ current: MTLTexture, uniforms: ARUniforms = defaults(), state: SIMD4<Float> = SIMD4(17,1,0,0),
                    previous: MTLTexture? = nil, histories: [MTLTexture]? = nil,
                    renderPipeline: MTLRenderPipelineState? = nil) throws -> [SIMD4<Float>] {
            let command = queue.makeCommandBuffer()!
            let pass = MTLRenderPassDescriptor(); pass.colorAttachments[0].texture = output
            pass.colorAttachments[0].loadAction = .clear; pass.colorAttachments[0].storeAction = .store
            let e = command.makeRenderCommandEncoder(descriptor:pass)!
            e.setRenderPipelineState(renderPipeline ?? post)
            var u = uniforms, s = state
            e.setFragmentBytes(&u,length:MemoryLayout<ARUniforms>.stride,index:0)
            e.setFragmentBytes(&s,length:16,index:1)
            e.setFragmentTexture(current,index:0); e.setFragmentTexture(previous ?? current,index:1)
            e.setFragmentTexture(exposure,index:2)
            for i in 0..<3 { e.setFragmentTexture(histories?[i] ?? current,index:i+3) }
            e.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:3); e.endEncoding()
            command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed,"post GPU execution: \(String(describing:command.error))")
            let values = read(output)
            try require(values.allSatisfy { p in (0..<4).allSatisfy { p[$0].isFinite && p[$0] >= 0 } },"nonfinite/negative post result")
            return values
        }
        let pattern = try texture(device,values:(0..<width*height).map { i in
            let x = Float(i%width), y = Float(i/width)
            return SIMD4(0.2+0.8*pow(sin(x*0.13),2),0.05+0.9*pow(cos(y*0.19),2),0.1+pow(sin((x+y)*0.17),2),1)
        })
        var clean = defaults(); clean.post.x = 0
        let reference = try render(pattern,uniforms:clean)
        let zeroDose = try render(pattern,state:SIMD4(17,0,1,0))
        try require(difference(reference,zeroDose) < 0.000001,"LSD intensity zero must exactly bypass subjective effects")
        checks.append("LSD zero is an exact clean-post bypass")
        for disabled in ["toggle","trip"] {
            var u = defaults()
            if disabled == "toggle" { u.post.x = 0 } else { u.time.z = 0 }
            try require(difference(reference,try render(pattern,uniforms:u,state:SIMD4(17,2,1,0))) < 0.000001,"\(disabled) bypass")
        }
        checks.append("Effects toggle and Trip zero bypass the subjective layer")
        for color in [SIMD4<Float>(0,0,0,1),SIMD4(0.1,0.1,0.1,1),SIMD4(1,1,1,1),SIMD4(8,8,8,1),SIMD4(1,0,0.8,1),SIMD4(0,0.7,1,1)] {
            let current = try texture(device,color:color)
            let expected = try render(current,uniforms:clean)
            for intensity: Float in [0,1,2] {
                let actual = try render(current,state:SIMD4(173,intensity,1,0))
                try require(difference(expected,actual) < 0.000003,"constant image changed for color \(color), LSD \(intensity)")
            }
        }
        checks.append("Constant black, neutral HDR, magenta and cyan remain unchanged at all intensities")
        let magenta = try texture(device,color:SIMD4(1,0,1,1))
        let oldWhite = try texture(device,color:SIMD4(4,4,4,1))
        for intensity: Float in [1,2] {
            for previousWeight: Float in [0,0.13] {
                var u = defaults(); u.post.z = previousWeight
                let protected = try render(magenta,uniforms:u,state:SIMD4(173,intensity,1,0),
                                           previous:oldWhite,histories:[oldWhite,oldWhite,oldWhite])
                try require(protected.allSatisfy { $0.y < 0.02*max($0.x,$0.z) },
                            "bright neutral history washes saturated current into milk at LSD \(intensity), interpolation \(previousWeight)")
            }
        }
        checks.append("Bright neutral history cannot wash saturated magenta into milk, including frame interpolation")
        let invalid = try texture(device,color:SIMD4(repeating:Float.nan))
        let valid = try render(pattern,state:SIMD4(17,1,0,0))
        let invalidHistory = try render(pattern,state:SIMD4(17,1,0,0),previous:invalid,histories:[invalid,invalid,invalid])
        try require(difference(valid,invalidHistory) == 0,"invalid history must not be read before reset")
        checks.append("Reset post does not read poisoned NaN previous/history textures")
        var stale = defaults(); stale.post.z = 0.2
        let staleNaNs = try render(pattern,uniforms:stale,state:SIMD4(17,1,1,0),previous:invalid,histories:[invalid,invalid,invalid])
        try require(difference(valid,staleNaNs) < 0.000001,"nonfinite stale reservoirs must fall back to current HDR")
        checks.append("Nonfinite stale reservoirs and previous-frame samples fall back to current HDR")
        let black = try texture(device,color:SIMD4(0,0,0,1))
        let darkTrail = try render(black,state:SIMD4(173,1,1,0),histories:[magenta,magenta,magenta])
        try require(darkTrail.allSatisfy { $0.x > 0.03 && $0.z > 0.03 && $0.y < 0.25*max($0.x,$0.z) },
                    "chroma protection must preserve bright colored afterimages over dark current")
        checks.append("Bright colored afterimages remain visible over dark current")
        let vivid = try texture(device,color:SIMD4(0.8,0.08,1,1))
        let aged = try render(vivid,state:SIMD4(17,1,1,0),histories:[black,black,black])
        let noHistory = try render(vivid,state:SIMD4(17,1,0,0),histories:[black,black,black])
        try require(difference(aged,noHistory) > 0.0001,"older reservoirs do not affect the post image")
        let moving = try render(pattern,state:SIMD4(18,1,1,0))
        let nearby = try render(pattern,state:SIMD4(18.001,1,1,0))
        let repeated = try render(pattern,state:SIMD4(18,1,1,0))
        try require(difference(moving,repeated) == 0,"frozen clock/history must produce identical output")
        try require(difference(moving,nearby) > 0 && difference(moving,nearby) < 0.0002,"effect motion is discontinuous")
        try require(difference(valid,moving) > 0.0001,"independent RGB/adaptation fields are static")
        checks.append("Older layers contribute; frozen state is exact; changing clock is continuous and visible")
        var scaled = defaults()
        scaled.viewport = SIMD4(Float(width*2),Float(height*2),1/Float(width*2),1/Float(height*2))
        scaled.post.w = 2
        try require(difference(valid,try render(pattern,uniforms:scaled,state:SIMD4(17,1,0,0))) == 0,
                    "logical post pixel units change with render-quality scale")
        checks.append("Compensated render-quality scale preserves logical-pixel effects exactly")

        let contracts = try device.makeComputePipelineState(function:library.makeFunction(name:"retinaContracts")!)
        let buffer = device.makeBuffer(length:1024*5*16,options:.storageModeShared)!
        let command = queue.makeCommandBuffer()!, e = command.makeComputeCommandEncoder()!
        var u = defaults()
        e.setComputePipelineState(contracts); e.setBytes(&u,length:MemoryLayout<ARUniforms>.stride,index:0)
        e.setBuffer(buffer,offset:0,index:1)
        e.dispatchThreads(MTLSize(width:1024,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:64,height:1,depth:1))
        e.endEncoding(); command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed,"retina helper GPU execution")
        let values = buffer.contents().bindMemory(to:SIMD4<Float>.self,capacity:5120)
        var minimumDose = Float.infinity, maximumDose:Float = 0, maximumOffset:Float = 0
        let luma = SIMD3<Float>(0.2126,0.7152,0.0722), expectedLuma = simd_dot(SIMD3<Float>(0.9,0.08,0.7),luma)
        for i in 0..<1024 {
            let a = values[i*5], b = values[i*5+1], grey = values[i*5+2], chroma = values[i*5+3], next = values[i*5+4]
            for offset in [SIMD2(a.x,a.y),SIMD2(a.z,a.w),SIMD2(b.x,b.y)] {
                maximumOffset = max(maximumOffset,simd_length(offset))
                try require(simd_length(offset) <= 2.000001,"actual RGB displacement exceeds configured two-pixel radius")
            }
            try require(abs(b.z-b.w) < 0.001,"dose envelope jumps under 1 ms clock advance")
            try require(simd_length(SIMD2(a.x-next.x,a.y-next.y)) < 0.001,"RGB drift jumps under 1 ms clock advance")
            minimumDose = min(minimumDose,b.z); maximumDose = max(maximumDose,b.z)
            try require(abs(grey.x-0.6)+abs(grey.y-0.6)+abs(grey.z-0.6) < 0.000002,"chroma adaptation tints gray")
            let rgb = SIMD3(chroma.x,chroma.y,chroma.z)
            try require(abs(simd_dot(rgb,luma)-expectedLuma) < 0.000003 && min(rgb.x,min(rgb.y,rgb.z)) >= 0,"chroma adaptation changed luminance or left positive gamut")
        }
        try require(maximumDose-minimumDose > 0.4,"coherent dose does not vary sufficiently")
        checks.append("1024 GPU probes enforce RGB <=2px, smooth dose/drift, exact neutral and zero-luminance chroma rotation")

        let psfPipeline = try device.makeComputePipelineState(function:library.makeFunction(name:"retinaPSFContracts")!)
        var pulseValues = [SIMD4<Float>](repeating:SIMD4(0,0,0,1),count:width*height)
        pulseValues[(height/2)*width+width/2] = SIMD4(8,8,8,1)
        let pulse = try texture(device,values:pulseValues), psf = try texture(device)
        let psfCommand = queue.makeCommandBuffer()!, pe = psfCommand.makeComputeCommandEncoder()!
        pe.setComputePipelineState(psfPipeline); pe.setBytes(&u,length:MemoryLayout<ARUniforms>.stride,index:0)
        pe.setTexture(pulse,index:0); pe.setTexture(psf,index:1)
        pe.dispatchThreads(MTLSize(width:width,height:height,depth:1),threadsPerThreadgroup:MTLSize(width:8,height:8,depth:1))
        pe.endEncoding(); psfCommand.commit(); psfCommand.waitUntilCompleted()
        try require(psfCommand.status == .completed,"PSF GPU execution")
        var energy = SIMD3<Double>.zero, moments = SIMD3<Double>.zero
        for (i,p) in read(psf).enumerated() {
            let rgb = SIMD3(Double(p.x),Double(p.y),Double(p.z))
            let dx = Double(i%width-width/2), dy = Double(i/width-height/2)
            energy += rgb; moments += rgb*(dx*dx+dy*dy)
        }
        for c in 0..<3 { try require(abs(energy[c]-8) < 0.0005,"spectral PSF creates/loses flux: \(energy)") }
        try require(moments.x > moments.y && moments.y > moments.z && moments.z > 0,"PSF spatial spread must increase with wavelength")
        checks.append("Actual linear spectral PSF preserves each channel integral and spreads red > green > blue")

        let reservoirPipeline = try device.makeComputePipelineState(function:library.makeFunction(name:"retinaHistoryUpdate")!)
        struct UpdateParameters { var attack: SIMD4<Float>; var release: SIMD4<Float> }
        func layers(_ w: Int = 4, _ h: Int = 4, color: SIMD4<Float> = .zero) throws -> [MTLTexture] {
            try (0..<3).map { _ in try texture(device,width:w,height:h,color:color) }
        }
        func update(_ current: MTLTexture, _ prior: [MTLTexture], _ result: [MTLTexture],
                    attack: SIMD3<Float>, release: SIMD3<Float>, valid: Bool) throws {
            let c = queue.makeCommandBuffer()!, e = c.makeComputeCommandEncoder()!
            var state = UpdateParameters(attack:SIMD4(attack,valid ? 1 : 0),release:SIMD4(release,0))
            e.setComputePipelineState(reservoirPipeline)
            e.setBytes(&state,length:MemoryLayout<UpdateParameters>.stride,index:0)
            e.setTexture(current,index:0)
            for layer in 0..<3 { e.setTexture(prior[layer],index:layer+1); e.setTexture(result[layer],index:layer+4) }
            e.dispatchThreads(MTLSize(width:result[0].width,height:result[0].height,depth:1),threadsPerThreadgroup:MTLSize(width:8,height:8,depth:1))
            e.endEncoding(); c.commit(); c.waitUntilCompleted()
            try require(c.status == .completed,"fused history GPU update: \(String(describing:c.error))")
        }
        let ones = SIMD3<Float>(repeating:1), zeros = SIMD3<Float>.zero
        let incoming = try texture(device,width:1,height:1,color:SIMD4(0.9,0.1,0.7,1))
        let poisoned = try layers(color:SIMD4(repeating:Float.nan)), reset = try layers()
        try update(incoming,poisoned,reset,attack:zeros,release:zeros,valid:false)
        for layer in reset { for p in read(layer) {
            try require(simd_length(p-SIMD4(0.9,0.1,0.7,1)) < 0.000001,"reset reads poisoned old reservoir")
        } }
        checks.append("Fused three-layer reset never reads poisoned history")

        // Independent double-precision source-pixel overlap reference. Test all
        // layer outputs, noninteger ratios, large reductions, borders and upsampling.
        for (sw,sh,dw,dh) in [(16,16,3,3),(17,11,4,3),(32,17,3,2),(8,6,4,3),(3,2,7,5)] {
            let output = try layers(dw,dh)
            for sourceIndex in Set([0,1,sw-1,sw,sw*sh/2,sw*sh-1]) {
                var values = [SIMD4<Float>](repeating:SIMD4(0,0,0,1),count:sw*sh)
                values[sourceIndex] = SIMD4(1,0.25,0.75,1)
                let source = try texture(device,width:sw,height:sh,values:values)
                try update(source,poisoned,output,attack:ones,release:ones,valid:false)
                let sx = Double(sourceIndex%sw), sy = Double(sourceIndex/sw)
                let scaleX = Double(sw)/Double(dw), scaleY = Double(sh)/Double(dh)
                for layer in output {
                    var flux = SIMD3<Double>.zero
                    for (i,pixel) in read(layer).enumerated() {
                        let x0 = Double(i%dw)*scaleX, y0 = Double(i/dw)*scaleY
                        let weightX = max(0,min(x0+scaleX,sx+1)-max(x0,sx))
                        let weightY = max(0,min(y0+scaleY,sy+1)-max(y0,sy))
                        let coverage = weightX*weightY/(scaleX*scaleY)
                        let expected = SIMD3(coverage,coverage*0.25,coverage*0.75)
                        let actual = SIMD3(Double(pixel.x),Double(pixel.y),Double(pixel.z))
                        try require(simd_length(actual-expected) < 0.00001,"retinal footprint differs from exact pixel coverage")
                        flux += actual*(scaleX*scaleY)
                    }
                    try require(simd_length(flux-SIMD3(1,0.25,0.75)) < 0.00003,
                                "retinal downsample loses phase-dependent highlight flux: \(flux)")
                }
            }
        }
        checks.append("Area restriction preserves impulse flux and exact phase/edge coverage at five ratios, including >10x")

        let attacks: [Double] = [0.025,0.050,0.090], releases: [Double] = [0.24,0.72,1.65]
        func coefficients(_ dt: Double, _ halfLives: [Double]) -> SIMD3<Float> {
            SIMD3(Float(-expm1(-log(2)*dt/halfLives[0])),Float(-expm1(-log(2)*dt/halfLives[1])),Float(-expm1(-log(2)*dt/halfLives[2])))
        }
        let darkness = try texture(device,width:1,height:1,color:SIMD4(0,0,0,1))
        let flash = try texture(device,width:1,height:1,color:SIMD4(1,0.25,0.75,1))
        var cadenceResults = [[SIMD4<Float>]]()
        for fps in [30,60,120] {
            var prior = try layers(1,1,color:SIMD4(0,0,0,1)), next = try layers(1,1)
            let a = coefficients(1/Double(fps),attacks), r = coefficients(1/Double(fps),releases)
            for _ in 0..<(fps/10) { try update(flash,prior,next,attack:a,release:r,valid:true); swap(&prior,&next) }
            for (layer,t) in prior.enumerated() {
                let captured = read(t)[0].x
                try require(abs(Double(captured)-(1-pow(2,-0.1/attacks[layer]))) < 0.00001,"brief glint attack differs from analytic response")
                try require(captured > 0.5,"brief glint fails to enter a retinal layer")
            }
            for _ in 0..<(fps*9/10) { try update(darkness,prior,next,attack:a,release:r,valid:true); swap(&prior,&next) }
            let samples = prior.map { read($0)[0] }
            for (layer,pixel) in samples.enumerated() {
                let expected = Float((1-pow(2,-0.1/attacks[layer]))*pow(2,-0.9/releases[layer]))
                try require(simd_length(pixel-SIMD4(expected,expected*0.25,expected*0.75,1)) < 0.00002,
                            "pulse attack/release changes with display cadence")
            }
            cadenceResults.append(samples)
        }
        try require(difference(cadenceResults[0],cadenceResults[2]) < 0.00001,"30/120Hz pulse trails diverge")
        checks.append("Actual fused GPU captures a 100ms colored pulse and preserves its analytic tail at 30/60/120Hz")

        for color in [SIMD4<Float>(0,0,0,1),SIMD4(0.2,0.2,0.2,1),SIMD4(8,8,8,1),SIMD4(1,0,0.8,1)] {
            let source = try texture(device,width:17,height:11,color:color)
            let prior = try layers(4,3,color:color), next = try layers(4,3)
            try update(source,prior,next,attack:coefficients(0.02,attacks),release:coefficients(0.02,releases),valid:true)
            for layer in next { for pixel in read(layer) {
                try require(simd_length(pixel-color) < 0.00001,"constant black/neutral/HDR spectrum changes in fused update")
            } }
        }
        let brightNeutral = try texture(device,width:1,height:1,color:SIMD4(1,1,1,1))
        var neutralPrior = try layers(1,1,color:SIMD4(0,0,0,1)), neutralNext = try layers(1,1)
        try update(brightNeutral,neutralPrior,neutralNext,attack:coefficients(0.02,attacks),release:coefficients(0.02,releases),valid:true)
        swap(&neutralPrior,&neutralNext)
        try update(darkness,neutralPrior,neutralNext,attack:coefficients(0.3,attacks),release:coefficients(0.3,releases),valid:true)
        for t in neutralNext {
            let p = read(t)[0]
            try require(abs(p.x-p.y)+abs(p.y-p.z) < 0.000001,"achromatic flash acquires a temporal tint")
        }
        checks.append("Fused histories preserve constant black/neutral HDR/color and use one RGB clock for neutral flashes")

        let directory = URL(fileURLWithPath:"native/artifacts/retina-checks")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let report:[String:Any] = ["status":"PASS","device":device.name,"water_sha256":SHA256.hash(data:Data(sources.last!.utf8)).map{String(format:"%02x",$0)}.joined(),"checks":checks,"maximum_rgb_offset_pixels":maximumOffset,"dose_range":[minimumDose,maximumDose],"psf_channel_energy":[energy.x,energy.y,energy.z],"psf_channel_second_moment":[moments.x,moments.y,moments.z],"note":"GPU correctness checks only; no performance or displayed-FPS claim."]
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("report.json"))
        let markdown = "# Retina GPU contracts\n\nDevice: \(device.name)\n\n"+checks.map { "- PASS: \($0)" }.joined(separator:"\n")+"\n\nMaximum tested RGB displacement: \(maximumOffset) px. PSF channel energy: \(energy).\n\nNo performance or displayed-FPS claim.\n"
        try markdown.write(to:directory.appendingPathComponent("report.md"),atomically:true,encoding:.utf8)
        print("PASS \(checks.count) retina contracts on \(device.name)\n\(markdown)")
    }
}
