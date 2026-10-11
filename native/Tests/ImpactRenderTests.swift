import Foundation
import Metal
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import simd

/// Paired production rendering isolates surface response from falling spheres,
/// pigment, pupil adaptation and unrelated background/camera motion. This is a
/// visual regression diagnostic, not a performance benchmark.
@main struct ImpactRenderTests {
    static let width = 800, height = 500
    static let size: Float = 3
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw NativeFailure(message) }
    }

    final class Renderer {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let waterPipeline: MTLRenderPipelineState
        let postPipeline: MTLRenderPipelineState
        let emptyVolume: MTLTexture
        let exposure: MTLTexture
        init(source: String) throws {
            guard let gpu = MTLCreateSystemDefaultDevice(), let commands = gpu.makeCommandQueue() else {
                throw NativeFailure("No Metal device")
            }
            device = gpu; queue = commands
            let options = MTLCompileOptions()
            options.mathMode = .fast
            let library = try gpu.makeLibrary(source: source, options: options)
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "waterVertex")
            descriptor.fragmentFunction = library.makeFunction(name: "waterFragment")
            descriptor.colorAttachments[0].pixelFormat = .rgba32Float
            waterPipeline = try gpu.makeRenderPipelineState(descriptor: descriptor)
            descriptor.fragmentFunction = library.makeFunction(name: "postFragment")
            descriptor.colorAttachments[0].pixelFormat = .rgba8Unorm_srgb
            postPipeline = try gpu.makeRenderPipelineState(descriptor: descriptor)
            let volume = MTLTextureDescriptor()
            volume.textureType = .type3D; volume.pixelFormat = .rgba32Float
            volume.width = 4; volume.height = 4; volume.depth = 4
            volume.storageMode = .shared; volume.usage = .shaderRead
            guard let blank = gpu.makeTexture(descriptor: volume) else { throw NativeFailure("No empty cloud texture") }
            let zero = [Float](repeating: 0, count: 4 * 4 * 4 * 4)
            zero.withUnsafeBytes {
                blank.replace(region: MTLRegionMake3D(0, 0, 0, 4, 4, 4), mipmapLevel: 0, slice: 0,
                              withBytes: $0.baseAddress!, bytesPerRow: 4 * 16, bytesPerImage: 4 * 4 * 16)
            }
            emptyVolume = blank
            let e = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 1, height: 1, mipmapped: false)
            e.storageMode = .shared; e.usage = .shaderRead
            guard let fixedExposure = gpu.makeTexture(descriptor: e) else { throw NativeFailure("No fixed exposure texture") }
            var one: Float = 1
            withUnsafeBytes(of: &one) {
                fixedExposure.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                                      withBytes: $0.baseAddress!, bytesPerRow: 4)
            }
            exposure = fixedExposure
        }

        func texture(width: Int, height: Int, values: [Float]? = nil,
                     format: MTLPixelFormat = .rgba32Float) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = [.shaderRead, .renderTarget]
            guard let result = device.makeTexture(descriptor: descriptor) else { throw NativeFailure("Texture allocation failed") }
            if let values {
                try require(values.count == width * height * 4, "Incorrect texture payload")
                values.withUnsafeBytes {
                    result.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                                   withBytes: $0.baseAddress!, bytesPerRow: width * 16)
                }
            }
            return result
        }

        func pass(pipeline: MTLRenderPipelineState, uniforms: ARUniforms, textures: [MTLTexture], output: MTLTexture) throws {
            guard let command = queue.makeCommandBuffer() else { throw NativeFailure("Command allocation failed") }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = output
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { throw NativeFailure("Render encoder allocation failed") }
            encoder.setRenderPipelineState(pipeline)
            var u = uniforms
            encoder.setFragmentBytes(&u, length: MemoryLayout<ARUniforms>.stride, index: 0)
            var retinaFrame = SIMD4<Float>(repeating: 0)
            encoder.setFragmentBytes(&retinaFrame, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
            var bedFrame = SIMD4<Float>(repeating: 0)
            encoder.setFragmentBytes(&bedFrame, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
            for (index, texture) in textures.enumerated() { encoder.setFragmentTexture(texture, index: index) }
            if textures.count <= 3, let current = textures.first {
                for index in 3...5 { encoder.setFragmentTexture(current, index: index) }
            }
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed, "Metal command failed: \(String(describing: command.error))")
        }

        func render(water: SimulationFrame, fixture: SimulationFrame, uniforms: ARUniforms, neutral: Bool) throws -> MTLTexture {
            let s = fixture.state
            let surface = try texture(width: Int(s.water_width), height: Int(s.water_height), values: water.water)
            // Depth is kept identical too: this isolates geometric height/normal
            // response, not a difference in the lower boundary's optical mass.
            let depth = try texture(width: Int(s.water_width), height: Int(s.water_height), values: fixture.dye)
            var bottom = fixture.bed, sky = fixture.sky, light = fixture.light
            if neutral {
                for y in 0..<Int(s.bed_height) { for x in 0..<Int(s.bed_width) {
                    let worldX = -1.5 + 5 * (Double(x) + 0.5) / Double(s.bed_width)
                    let worldY = -1.5 + 5 * (Double(y) + 0.5) / Double(s.bed_height)
                    let check = (Int(floor(worldX / 0.09)) + Int(floor(worldY / 0.09))) & 1
                    let index = (y * Int(s.bed_width) + x) * 4
                    bottom[index] = check == 0 ? 0.08 : 0.75
                    bottom[index + 1] = 0; bottom[index + 2] = 0; bottom[index + 3] = 0
                }}
                for y in 0..<Int(s.sky_height) { for x in 0..<Int(s.sky_width) {
                    let index = (y * Int(s.sky_width) + x) * 4
                    let stripe: Float = (x / 8) & 1 == 0 ? 0.25 : 1.3
                    for c in 0..<3 { sky[index + c] = stripe }
                    sky[index + 3] = 1
                }}
                light = Array(repeating: 1, count: light.count)
            }
            let bed = try texture(width: Int(s.bed_width), height: Int(s.bed_height), values: bottom)
            let environment = try texture(width: Int(s.sky_width), height: Int(s.sky_height), values: sky)
            let illumination = try texture(width: Int(s.light_width), height: Int(s.light_height), values: light)
            let caustics = try texture(width: 1, height: 1, values: [1, 1, 1, 1])
            let output = try texture(width: width, height: height)
            try pass(pipeline: waterPipeline, uniforms: uniforms,
                     textures: [surface, depth, bed, environment, illumination, emptyVolume, caustics], output: output)
            return output
        }

        func read(_ texture: MTLTexture) -> [Float] {
            var result = [Float](repeating: 0, count: texture.width * texture.height * 4)
            result.withUnsafeMutableBytes {
                texture.getBytes($0.baseAddress!, bytesPerRow: texture.width * 16,
                                 from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
            }
            return result
        }

        func png(_ texture: MTLTexture, uniforms: ARUniforms, destination: URL) throws {
            let display = try self.texture(width: width, height: height, format: .rgba8Unorm_srgb)
            try pass(pipeline: postPipeline, uniforms: uniforms, textures: [texture, texture, exposure], output: display)
            var bytes = [UInt8](repeating: 0, count: width * height * 4)
            bytes.withUnsafeMutableBytes {
                display.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                                 from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            }
            let provider = CGDataProvider(data: Data(bytes) as CFData)!
            let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            guard let file = CGImageDestinationCreateWithURL(destination as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw NativeFailure("Could not save \(destination.path)")
            }
            CGImageDestinationAddImage(file, image, nil)
            try require(CGImageDestinationFinalize(file), "Could not finish PNG")
        }
    }

    struct Statistics {
        var count = 0
        var squares: Double = 0
        var maximum: Float = 0
        mutating func add(_ value: Float) {
            count += 1; squares += Double(value) * Double(value); maximum = max(maximum, abs(value))
        }
        var rms: Double { sqrt(squares / Double(max(1, count))) }
        var json: [String: Any] { ["sample_count": count, "rms": rms, "maximum": maximum] }
    }

    static func main() throws {
        let args = CommandLine.arguments
        let shader = args.count > 1 ? args[1] : "native/Sources/Water.metal"
        let destination = URL(fileURLWithPath: args.count > 2 ? args[2] : "native/artifacts/impact-fix/render", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        var source = try String(contentsOfFile: shader, encoding: .utf8)
        let directory = URL(fileURLWithPath: shader).deletingLastPathComponent()
        for name in ["BedLife.metal", "Caustics.metal"] {
            let dependency = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: dependency.path) {
                source = try String(contentsOf: dependency, encoding: .utf8) + "\n" + source
            }
        }
        let renderer = try Renderer(source: source)
        let control = try CoreBridge(seed: 0xAC1D2026)
        let impacted = try CoreBridge(seed: 0xAC1D2026)
        for core in [control, impacted] {
            try core.commandOffline(3, value: 0)
            if core.current().state.travel != 0 { try core.commandOffline(11) }
            if core.current().state.current != 0 { try core.commandOffline(12) }
        }
        let initial = control.current()
        let center = SIMD2<Double>(initial.state.world_origin_x + Double(initial.state.uniforms.camera.x),
                                   initial.state.world_origin_y + Double(initial.state.uniforms.camera.y))
        try impacted.dropOffline(world: center, size: size)
        var contactTime: Double?
        var rows: [[String: Any]] = []
        var table = "| Age s | Fixture | Peak height mm | Peak slope | HDR max delta | Center RMS | Annulus RMS | Far RMS | Center relative RMS |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
        let targetAges: [Double] = [0, 0.05, 0.15, 0.4, 0.8]
        var nextAge = 0, steps = 0
        while nextAge < targetAges.count {
            steps += 1
            try require(steps <= 300, "Drop did not contact or observation time did not advance")
            try control.stepOffline(delta: 1.0 / 60)
            try impacted.stepOffline(delta: 1.0 / 60)
            let a = control.current(), b = impacted.current()
            if contactTime == nil, let event = b.drops.first(where: { $0.identity.z == 1 }) {
                contactTime = Double(event.dynamics.z)
            }
            guard let contact = contactTime else {
                try require(a.water == b.water, "Airborne drop disturbed water before contact")
                continue
            }
            let age = b.state.physical_time - contact
            if age + 1e-6 < targetAges[nextAge] { continue }
            var heightDelta = Statistics(), slopeDelta = Statistics()
            for i in stride(from: 0, to: a.water.count, by: 4) {
                heightDelta.add(b.water[i] - a.water[i])
                slopeDelta.add(b.water[i + 1] - a.water[i + 1])
                slopeDelta.add(b.water[i + 2] - a.water[i + 2])
            }
            for neutral in [false, true] {
                let label = neutral ? "neutral-grid" : "journey"
                var u = initial.state.uniforms
                u.time.x = 0; u.time.y = 0
                u.viewport = SIMD4(Float(width), Float(height), 1 / Float(width), 1 / Float(height))
                u.post = SIMD4(0, 2, 0, 1)
                u.world.y = -0.25; u.world.z = 0.25
                // Fit once against a fixed envelope: differences cannot come
                // from one case's extra extrema moving or resizing the camera.
                WaterCamera.fill(&u, aspect: Float(width) / Float(height))
                u.world.y = min(a.state.uniforms.world.y, b.state.uniforms.world.y)
                u.world.z = max(a.state.uniforms.world.z, b.state.uniforms.world.z)
                if neutral {
                    u.optics.y = 0.12; u.optics.z = 1; u.optics.w = 1
                    u.gain = SIMD4(repeating: 1)
                    u.light.w = 0
                    u.illumination = SIMD4(1, 1, 1, 0)
                }
                let before = try renderer.render(water: a, fixture: initial, uniforms: u, neutral: neutral)
                let after = try renderer.render(water: b, fixture: initial, uniforms: u, neutral: neutral)
                let beforeValues = renderer.read(before), afterValues = renderer.read(after)
                try require(beforeValues.allSatisfy(\.isFinite) && afterValues.allSatisfy(\.isFinite), "Nonfinite water radiance")
                try require(beforeValues.enumerated().contains(where: { $0.offset % 4 != 3 && $0.element > 1e-6 }), "Empty diagnostic fixture")
                var all = Statistics(), central = Statistics(), annulus = Statistics(), far = Statistics()
                var baselineAll = Statistics(), baselineCenter = Statistics(), baselineAnnulus = Statistics(), baselineFar = Statistics()
                var delta = beforeValues
                let eye = SIMD3<Float>(u.eye.x, u.eye.y, u.eye.z)
                let forward = SIMD3<Float>(u.forward.x, u.forward.y, u.forward.z)
                let right = SIMD3<Float>(u.right.x, u.right.y, u.right.z)
                let up = SIMD3<Float>(u.up.x, u.up.y, u.up.z)
                for y in 0..<height { for x in 0..<width {
                    let ray = forward + right * (((Float(x) + 0.5) / Float(width) * 2 - 1) * u.right.w)
                        + up * ((1 - (Float(y) + 0.5) / Float(height) * 2) * u.up.w)
                    let p = eye - ray * (eye.z / ray.z)
                    let radius = simd_length(SIMD2(p.x - u.camera.x, p.y - u.camera.y))
                    let index = (y * width + x) * 4
                    for c in 0..<3 {
                        let difference = afterValues[index + c] - beforeValues[index + c]
                        all.add(difference)
                        baselineAll.add(beforeValues[index + c])
                        if radius < 0.075 { central.add(difference); baselineCenter.add(beforeValues[index + c]) }
                        else if radius < 0.3 { annulus.add(difference); baselineAnnulus.add(beforeValues[index + c]) }
                        if radius > 0.5 { far.add(difference); baselineFar.add(beforeValues[index + c]) }
                        delta[index + c] = abs(difference) * 8
                    }
                    delta[index + 3] = 1
                }}
                if targetAges[nextAge] > 0 {
                    try require(heightDelta.maximum > Float.ulpOfOne, "No post-contact surface displacement")
                    try require(all.maximum > Float.ulpOfOne, "Actual Metal water pass has no visible response to impact")
                }
                let name = String(format: "age-%03d-%@", Int((targetAges[nextAge] * 1000).rounded()), label)
                try renderer.png(before, uniforms: u, destination: destination.appendingPathComponent(name + "-control.png"))
                try renderer.png(after, uniforms: u, destination: destination.appendingPathComponent(name + "-impact.png"))
                let difference = try renderer.texture(width: width, height: height, values: delta)
                try renderer.png(difference, uniforms: u, destination: destination.appendingPathComponent(name + "-delta8x.png"))
                rows.append(["requested_age": targetAges[nextAge], "actual_age": age, "physical_time": b.state.physical_time,
                             "fixture": label, "height_m": heightDelta.json, "slope": slopeDelta.json,
                             "radiance_all": all.json, "radiance_center": central.json,
                             "radiance_annulus": annulus.json, "radiance_far": far.json,
                             "relative_rms_all": all.rms / max(1e-12, baselineAll.rms),
                             "relative_rms_center": central.rms / max(1e-12, baselineCenter.rms),
                             "relative_rms_annulus": annulus.rms / max(1e-12, baselineAnnulus.rms),
                             "relative_rms_far": far.rms / max(1e-12, baselineFar.rms)])
                table += String(format: "| %.4f | %@ | %.3f | %.4f | %.5f | %.5f | %.5f | %.5f | %.4f |\n",
                                age, label, heightDelta.maximum * 1000, slopeDelta.maximum, all.maximum,
                                central.rms, annulus.rms, far.rms, central.rms / max(1e-12, baselineCenter.rms))
            }
            nextAge += 1
        }
        let report: [String: Any] = ["status": "PASS", "device": renderer.device.name,
            "shader_sha256": SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined(),
            "executable_sha256": SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: args[0])))
                .map { String(format: "%02x", $0) }.joined(),
            "drop_size": size, "ballistic_radius_m": 0.0055 * size, "seed": 0xAC1D2026,
            "contact_physical_time": contactTime!, "output_size": [width, height],
            "isolation": "Same-seed paired FFI simulation. Airborne and pigment omitted. Fixed camera, original environment, fixed exposure, no post effects/history. Control and impact share optical depth and lighting. GPU mathMode fast.",
            "region_definition": "Radial masks on z=0 reference plane: center < 0.075 m, annulus 0.075...0.3 m, far > 0.5 m. Radiance differences in linear HDR; PNG delta uses fixed gain 8 followed by production tone/sRGB.",
            "rows": rows]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: destination.appendingPathComponent("report.json"))
        try table.write(to: destination.appendingPathComponent("report.md"), atomically: true, encoding: .utf8)
        print(table)
        print("PASS paired real-Metal surface impact diagnostic: \(destination.path)")
    }
}
