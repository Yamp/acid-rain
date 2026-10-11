import Foundation
import Metal
import simd
import CryptoKit

struct CausticsTestUniforms {
    var light: SIMD4<Float>
    var domain = SIMD4<Float>(2, -0.5, 3, 1)
    var grid: SIMD4<UInt32>
}
struct CausticsTestFailure: Error, CustomStringConvertible { let description: String }

@main struct CausticsTests {
    static func require(_ yes: Bool, _ why: String) throws {
        if !yes { throw CausticsTestFailure(description: why) }
    }
    static func field(_ device: MTLDevice, size: Int, _ values: [SIMD4<Float>]) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: size, height: size, mipmapped: false)
        d.storageMode = .shared; d.usage = .shaderRead
        guard let t = device.makeTexture(descriptor: d) else { throw CausticsTestFailure(description: "texture allocation failed") }
        values.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0,0,size,size), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: size * 16) }
        return t
    }
    static func fresnel(_ cosine: Double, _ eta: Double) -> Double {
        let t = sqrt(1 - (1 - cosine * cosine) / (eta * eta))
        let s = (cosine - eta * t) / (cosine + eta * t)
        let p = (eta * cosine - t) / (eta * cosine + t)
        return (s*s+p*p)*0.5
    }
    static func main() {
        do { try run() }
        catch { FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8)); exit(1) }
    }
    static func run() throws {
        let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "native/Sources/Caustics.metal"
        let outputDirectory = URL(fileURLWithPath: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "native/artifacts/impact-fix/caustics-checks")
        let source = try String(contentsOfFile: path, encoding: .utf8)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw CausticsTestFailure(description: "Metal unavailable") }
        let options = MTLCompileOptions(); options.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: options)
        let compute = try device.makeComputePipelineState(function: library.makeFunction(name: "causticsMap")!)
        let pipeline = MTLRenderPipelineDescriptor()
        pipeline.vertexFunction = library.makeFunction(name: "causticsVertex")
        pipeline.fragmentFunction = library.makeFunction(name: "causticsFragment")
        pipeline.colorAttachments[0].pixelFormat = .rgba16Float
        pipeline.colorAttachments[0].isBlendingEnabled = true
        pipeline.colorAttachments[0].sourceRGBBlendFactor = .one
        pipeline.colorAttachments[0].destinationRGBBlendFactor = .one
        pipeline.colorAttachments[0].sourceAlphaBlendFactor = .one
        pipeline.colorAttachments[0].destinationAlphaBlendFactor = .one
        let raster = try device.makeRenderPipelineState(descriptor: pipeline)
        let outputSize = 256, fieldSize = 128
        struct Result { let pixels: [SIMD4<Float>]; let mapping: MTLTexture; let flux: SIMD3<Double>; let mean: Double; let variance: Double; let maximum: Double }
        func render(cells: Int = 128, light: SIMD3<Float> = SIMD3(0,0,1), aperture: Float = 0.02,
                    surface: (Float,Float) -> SIMD4<Float>, depth: (Float,Float,Float) -> Float,
                    mappingOverride: ((Int,Int)->SIMD4<Float>)? = nil) throws -> Result {
            var water = [SIMD4<Float>](), dye = [SIMD4<Float>]()
            for y in 0..<fieldSize { for x in 0..<fieldSize {
                let px = Float(x) * 2 / Float(fieldSize), py = Float(y) * 2 / Float(fieldSize)
                let sample = surface(px,py); water.append(sample)
                dye.append(SIMD4(0,0,0,depth(px,py,sample.x)))
            } }
            let waterTexture = try field(device, size: fieldSize, water), depthTexture = try field(device, size: fieldSize, dye)
            let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: cells+1, height: cells+1, mipmapped: false)
            md.textureType = .type2DArray; md.arrayLength = 3; md.storageMode = .shared; md.usage = [.shaderRead,.shaderWrite]
            let mapping = device.makeTexture(descriptor: md)!
            let rd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: outputSize, height: outputSize, mipmapped: false)
            rd.storageMode = .shared; rd.usage = [.renderTarget,.shaderRead]
            let receiver = device.makeTexture(descriptor: rd)!
            var u = CausticsTestUniforms(light: SIMD4(simd_normalize(light), aperture), grid: SIMD4(UInt32(cells),UInt32(outputSize),UInt32(outputSize),0))
            try require(MemoryLayout<CausticsTestUniforms>.stride == 48, "caustics uniform layout changed")
            let mapCommand = queue.makeCommandBuffer()!
            let c = mapCommand.makeComputeCommandEncoder()!
            c.setComputePipelineState(compute); c.setBytes(&u,length:48,index:0)
            c.setTexture(waterTexture,index:0); c.setTexture(depthTexture,index:1); c.setTexture(mapping,index:2)
            c.dispatchThreads(MTLSize(width:cells+1,height:cells+1,depth:3),threadsPerThreadgroup:MTLSize(width:8,height:8,depth:1)); c.endEncoding()
            mapCommand.commit(); mapCommand.waitUntilCompleted()
            try require(mapCommand.status == .completed,"GPU ray mapping failed")
            if let mappingOverride {
                let n = cells+1
                let nodes = (0..<n*n).map { mappingOverride($0 % n,$0 / n) }
                for slice in 0..<3 { nodes.withUnsafeBytes {
                    mapping.replace(region:MTLRegionMake2D(0,0,n,n),mipmapLevel:0,slice:slice,withBytes:$0.baseAddress!,bytesPerRow:n*16,bytesPerImage:n*n*16)
                } }
            }
            let command = queue.makeCommandBuffer()!
            let pass = MTLRenderPassDescriptor(); pass.colorAttachments[0].texture = receiver
            pass.colorAttachments[0].loadAction = .clear; pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0)
            let r = command.makeRenderCommandEncoder(descriptor:pass)!
            r.setRenderPipelineState(raster); r.setCullMode(.none)
            r.setVertexBytes(&u,length:48,index:0); r.setFragmentBytes(&u,length:48,index:0); r.setVertexTexture(mapping,index:0)
            r.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:cells*cells*6,instanceCount:3); r.endEncoding()
            command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed,"GPU caustics execution failed: \(String(describing:command.error))")
            var half = [Float16](repeating:0,count:outputSize*outputSize*4)
            half.withUnsafeMutableBytes { receiver.getBytes($0.baseAddress!,bytesPerRow:outputSize*8,from:MTLRegionMake2D(0,0,outputSize,outputSize),mipmapLevel:0) }
            var pixels = [SIMD4<Float>](), flux = SIMD3<Double>(repeating:0), interior = [Double]()
            for y in 0..<outputSize { for x in 0..<outputSize {
                let i = (y*outputSize+x)*4, p = SIMD4(Float(half[i]),Float(half[i+1]),Float(half[i+2]),Float(half[i+3]))
                try require((0..<4).allSatisfy { p[$0].isFinite && p[$0] >= 0 },"nonfinite or negative caustic pixel")
                pixels.append(p); flux += SIMD3(Double(p.x),Double(p.y),Double(p.z)) * pow(3/Double(outputSize),2)
                let position = SIMD2((Float(x)+0.5)*3/Float(outputSize)-0.5,(Float(y)+0.5)*3/Float(outputSize)-0.5)
                if min(position.x,position.y) > 0.3 && max(position.x,position.y) < 1.7 { interior.append(Double(p.y)) }
            } }
            let mean = interior.reduce(0,+)/Double(interior.count)
            let variance = interior.reduce(0) { $0 + pow($1-mean,2) }/Double(interior.count)
            return Result(pixels:pixels,mapping:mapping,flux:flux,mean:mean,variance:variance,maximum:Double(pixels.map(\.y).max()!))
        }
        let flat: (Float,Float)->SIMD4<Float> = { _,_ in SIMD4(0,0,0,0) }
        let constantDepth: (Float,Float,Float)->Float = { _,_,_ in 0.7 }
        var records = [[String:Any]]()
        for cells in [64,128] {
            let result = try render(cells:cells,surface:flat,depth:constantDepth)
            for (c,eta) in [1.331,1.333,1.337].enumerated() {
                let expected = (1-fresnel(1,eta))*4
                try require(abs(result.flux[c]/expected-1) < 0.015,"flat total transmitted flux incorrect: \(result.flux[c]) vs \(expected)")
            }
            let expected = 1-fresnel(1,1.333)
            try require(abs(result.mean-expected) < 0.02,"flat bed irradiance differs from Fresnel transmission")
            try require(sqrt(result.variance)/result.mean < 0.03,"flat plane has visible grid variance")
            records.append(["name":"flat \(cells)² rays per wavelength","flux_rgb":[result.flux.x,result.flux.y,result.flux.z],"mean":result.mean,"variation":sqrt(result.variance)/result.mean,"maximum":result.maximum])
        }
        let light = simd_normalize(SIMD3<Float>(0.3,-0.2,1))
        let tilted = try render(light:light,surface:{ x,y in SIMD4(0.08*(x-1)-0.06*(y-1),0.08,-0.06,0) },depth:{ _,_,h in 0.7+h })
        let normal = simd_normalize(SIMD3<Float>(-0.08,0.06,1))
        let cosI = Double(simd_dot(normal,light)), projected = Double(simd_dot(normal,light)/normal.z)
        for (c,eta) in [1.331,1.333,1.337].enumerated() {
            let expected = 4*projected*(1-fresnel(cosI,eta))
            try require(abs(tilted.flux[c]/expected-1) < 0.02,"tilted surface projected-flux mismatch")
        }
        records.append(["name":"tilted surface and beam","flux_rgb":[tilted.flux.x,tilted.flux.y,tilted.flux.z],"mean":tilted.mean,"variation":sqrt(tilted.variance)/tilted.mean,"maximum":tilted.maximum])
        func wave(_ phase: Float) -> (Float,Float)->SIMD4<Float> {
            { x,y in let k = Float.pi*4, a:Float = 0.02
                return SIMD4(a*cos(k*x+phase)*cos(k*y),-a*k*sin(k*x+phase)*cos(k*y),-a*k*cos(k*x+phase)*sin(k*y),0) }
        }
        let focused = try render(surface:wave(0),depth:{ _,_,h in 0.7+h })
        let moved = try render(surface:wave(0.002),depth:{ _,_,h in 0.7+h })
        try require(focused.maximum > 1.3 && sqrt(focused.variance)/focused.mean > 0.15,"curved physical surface did not focus light")
        try require(abs(focused.flux.y/(4*(1-fresnel(1,1.333)))-1) < 0.025,"focusing creates or loses global flux")
        let difference = zip(focused.pixels,moved.pixels).reduce(0.0) { $0+Double(abs($1.0.y-$1.1.y)) }/Double(focused.pixels.count)
        try require(difference > 0 && difference < 0.03,"tiny surface motion creates a temporal jump")
        records.append(["name":"curved water focusing","flux_rgb":[focused.flux.x,focused.flux.y,focused.flux.z],"mean":focused.mean,"variation":sqrt(focused.variance)/focused.mean,"maximum":focused.maximum,"small_motion_mean_delta":difference])
        let bathymetry = try render(light:light,surface:flat,depth:{ x,_,_ in 0.7-0.2*x })
        var nodes = [SIMD4<Float>](repeating:SIMD4(repeating:0),count:129*129)
        nodes.withUnsafeMutableBytes { bathymetry.mapping.getBytes($0.baseAddress!,bytesPerRow:129*16,bytesPerImage:129*129*16,from:MTLRegionMake2D(0,0,129,129),mipmapLevel:0,slice:1) }
        let ray = SIMD3<Float>(-light.x/1.333,-light.y/1.333,-sqrt(1-(light.x*light.x+light.y*light.y)/(1.333*1.333)))
        for cell in [SIMD2(32,32),SIMD2(64,48),SIMD2(96,96)] {
            let sourcePoint = SIMD2<Float>(Float(cell.x)/64,Float(cell.y)/64)
            let t = (0.7-0.2*sourcePoint.x)/(0.2*ray.x-ray.z)
            let expected = sourcePoint+SIMD2(ray.x,ray.y)*t
            let actual = nodes[cell.y*129+cell.x]
            try require(simd_length(SIMD2(actual.x,actual.y)-expected) < 0.001,"actual sloping bed intersection incorrect")
        }
        let partial = try render(cells:1,surface:flat,depth:constantDepth,mappingOverride:{ x,y in
            x == 0 ? SIMD4(0,Float(y)*2,0,0) : SIMD4(1.4,1.4,1,0)
        })
        var centroid = SIMD2<Double>(repeating:0), weight = 0.0
        for (i,p) in partial.pixels.enumerated() {
            let point = SIMD2<Double>((Double(i%outputSize)+0.5)*3/Double(outputSize)-0.5,
                                      (Double(i/outputSize)+0.5)*3/Double(outputSize)-0.5)
            centroid += point*Double(p.y); weight += Double(p.y)
        }
        centroid /= weight
        try require(simd_length(centroid-SIMD2<Double>(1.4,1.4)) < 0.02,
                    "missing ray neighbors relocate valid incident flux: centroid \(centroid)")
        let report:[String:Any] = ["status":"PASS","device":device.name,"source":path,"shader_sha256":SHA256.hash(data:Data(source.utf8)).map{String(format:"%02x",$0)}.joined(),"cases":records,"contracts":["flat Fresnel flux and variance","tilted projected flux","forward focusing without energy creation","finite all-pixel output","small-motion continuity","actual sloped-bed Snell intersection","missing ray neighbors preserve valid flux position"],"note":"GPU correctness checks only; no timing or FPS claim."]
        try FileManager.default.createDirectory(at:outputDirectory,withIntermediateDirectories:true)
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:outputDirectory.appendingPathComponent("report.json"))
        var lines = ["# Caustics GPU contracts","","Device: \(device.name)","","| Case | Integrated green flux | Interior mean | Relative variation | Peak |","| --- | ---: | ---: | ---: | ---: |"]
        for r in records { let flux = r["flux_rgb"] as! [Double]; lines.append(String(format:"| %@ | %.5f | %.5f | %.5f | %.5f |",r["name"] as! String,flux[1],r["mean"] as! Double,r["variation"] as! Double,r["maximum"] as! Double)) }
        lines += ["","PASS all optical contracts. Finite-source reconstruction conserves continuous flux; raster totals are checked within 1.5–2.5% discretization tolerance. No performance claim."]
        let markdown = lines.joined(separator:"\n")+"\n"; print(markdown)
        try markdown.write(to:outputDirectory.appendingPathComponent("report.md"),atomically:true,encoding:.utf8)
    }
}
