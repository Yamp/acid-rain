import Foundation
import Metal
import simd
import CryptoKit

struct ShaderCheckFailure: Error, CustomStringConvertible {
    let description: String
}

/// Executes the production Metal functions, including the actual render passes.
@main struct ShaderTests {
    static var checks: [String] = []
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw ShaderCheckFailure(description: message) }
    }
    static func close(_ actual: Float, _ expected: Float, _ label: String, tolerance: Float = 0.000_02) throws {
        try require(actual.isFinite && abs(actual - expected) <= tolerance, "\(label): \(actual) != \(expected)")
    }
    static func defaults() -> ARUniforms {
        var u = ARUniforms()
        u.viewport = SIMD4(64, 32, 1 / 64, 1 / 32)
        u.time = SIMD4(17, 1 / 60, 4, 2)
        u.optics = SIMD4(1, 0.2, 1, 1)
        u.gain = SIMD4(1, 1, 1, 1)
        u.light = SIMD4(0.1, 0.2, 0.97, 0)
        u.post = SIMD4(1, 2, 0.25, 2)
        u.world = SIMD4(0, -0.01, 0.01, 0)
        u.eye = SIMD4(1, 1, 1, 1)
        u.forward = SIMD4(0, 0, -1, 0)
        u.right = SIMD4(1, 0, 0, 0.7)
        u.up = SIMD4(0, 1, 0, 0.35)
        return u
    }
    static func texture(_ device: MTLDevice, width: Int = 64, height: Int = 32,
                        format: MTLPixelFormat = .rgba32Float, color: SIMD4<Float>? = nil) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw ShaderCheckFailure(description: "cannot allocate test texture") }
        if let color {
            let values = Array(repeating: color, count: width * height)
            values.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: width * 16) }
        }
        return texture
    }
    static func read(_ texture: MTLTexture) -> [Float] {
        var values = [Float](repeating: 0, count: texture.width * texture.height * 4)
        values.withUnsafeMutableBytes { texture.getBytes($0.baseAddress!, bytesPerRow: texture.width * 16, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0) }
        return values
    }
    static func volume(_ device: MTLDevice, concentration: SIMD3<Float>, size: Int = 12) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = size; descriptor.height = size; descriptor.depth = size
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw ShaderCheckFailure(description: "cannot allocate 3D optical test volume")
        }
        let values = Array(repeating: SIMD4<Float>(concentration, 0), count: size * size * size)
        values.withUnsafeBytes {
            texture.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size), mipmapLevel: 0, slice: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: size * 16, bytesPerImage: size * size * 16)
        }
        return texture
    }
    static func pipeline(_ device: MTLDevice, _ library: MTLLibrary, fragment: String, format: MTLPixelFormat = .rgba32Float) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "waterVertex")
        descriptor.fragmentFunction = library.makeFunction(name: fragment)
        descriptor.colorAttachments[0].pixelFormat = format
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }
    static func render(_ queue: MTLCommandQueue, pipeline: MTLRenderPipelineState, uniforms: ARUniforms,
                       textures: [MTLTexture], output: MTLTexture) throws {
        guard let command = queue.makeCommandBuffer() else { throw ShaderCheckFailure(description: "no command buffer") }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = output
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { throw ShaderCheckFailure(description: "no render encoder") }
        encoder.setRenderPipelineState(pipeline)
        var uniforms = uniforms
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ARUniforms>.stride, index: 0)
        var retinaState = SIMD4<Float>(17, 0, 0, 0) // Isolate physical color transfer from authored perception.
        encoder.setFragmentBytes(&retinaState, length: 16, index: 1)
        var bedWorld = SIMD4<Float>.zero // Analytic fixtures omit authored bed material and direct caustics.
        encoder.setFragmentBytes(&bedWorld, length: 16, index: 2)
        for (index, texture) in textures.enumerated() { encoder.setFragmentTexture(texture, index: index) }
        let neutralCaustics = try texture(queue.device, width: 1, height: 1, color: SIMD4(1,1,1,0))
        encoder.setFragmentTexture(neutralCaustics, index: 6)
        if textures.count <= 3 {
            let neutralHistory = try texture(queue.device, width: 1, height: 1, color: SIMD4(0,0,0,1))
            for index in 3...5 { encoder.setFragmentTexture(neutralHistory, index: index) }
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try require(command.status == .completed, "GPU render command failed: \(String(describing: command.error))")
    }
    static func tone(_ input: Float) -> Float {
        min(0.9995, input * (input * 2.51 + 0.03) / (input * (input * 2.43 + 0.59) + 0.14))
    }
    static func fresnelReference(_ cosine: Double, _ ior: Double) -> Float {
        let sinT = sqrt(max(0, 1 - cosine * cosine)) / ior
        let cosT = sqrt(1 - sinT * sinT)
        let rs = (cosine - ior * cosT) / (cosine + ior * cosT)
        let rp = (ior * cosine - cosT) / (ior * cosine + cosT)
        return Float((rs * rs + rp * rp) / 2)
    }

    static func cloudTransferChecks(_ device: MTLDevice, _ queue: MTLCommandQueue,
                                    _ library: MTLLibrary) throws -> Int {
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "cloudOpticsContracts")!)
        let buffer = device.makeBuffer(length: 4 * 16, options: .storageModeShared)!
        func evaluate(_ volume: MTLTexture, _ concentration: SIMD3<Float>, _ origin: SIMD3<Float>,
                      _ direction: SIMD3<Float>, _ distance: Float, _ steps: Int,
                      _ illumination: SIMD3<Float>, study: Bool = false) throws -> [SIMD3<Float>] {
            var u = defaults(); u.illumination = SIMD4(illumination, study ? 1 : 0)
            let parameters = [SIMD4(origin, distance), SIMD4(simd_normalize(direction), Float(steps)), SIMD4(concentration, 0)]
            let command = queue.makeCommandBuffer()!
            let encoder = command.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&u, length: MemoryLayout<ARUniforms>.stride, index: 1)
            parameters.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            encoder.setTexture(volume, index: 0)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed, "GPU cloud transfer command failed: \(String(describing: command.error))")
            let values = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: 4)
            return (0..<4).map { SIMD3(values[$0].x, values[$0].y, values[$0].z) }
        }
        func validate(_ result: [SIMD3<Float>], concentration: SIMD3<Float>, distance: Float,
                      illumination: SIMD3<Float>, label: String) throws {
            let transmission = result[0], radiance = result[1], column = result[2], scattering = result[3]
            let sigmaWater = SIMD3<Float>(1.2, 1.6, 1.2) * 0.2 + SIMD3<Float>(repeating: 0.015)
            let absorption = SIMD3<Float>(simd_dot(concentration, SIMD3(2,0.30,0.25)),
                                          simd_dot(concentration, SIMD3(0.65,1.8,0.65)),
                                          simd_dot(concentration, SIMD3(0.30,0.45,1.8))) * 0.85
            let total = sigmaWater + absorption + scattering
            for c in 0..<3 {
                try require(scattering[c].isFinite && scattering[c] >= 0 && scattering[c] <= total[c], "\(label): passive nonnegative pigment scattering")
                let expectedTransmission = exp(-total[c] * distance)
                // expm1 in Double avoids cancellation for tiny optical distances.
                let removed = Float(-expm1(-Double(total[c]) * Double(distance)))
                let expectedRadiance = illumination[c] * (0.015 + scattering[c]) / total[c] * removed
                try close(transmission[c], expectedTransmission, "\(label): analytic Beer", tolerance: 0.0001)
                try close(radiance[c], expectedRadiance, "\(label): analytic single scatter",
                          tolerance: max(0.000000002, abs(expectedRadiance) * 0.0005))
                try close(column[c], concentration[c] * distance, "\(label): integrated concentration", tolerance: 0.0001)
                try require(transmission[c] >= 0 && transmission[c] <= 1.00001 && radiance[c] >= 0, "\(label): finite passive transfer range")
                if illumination[c] > 0 {
                    try require(radiance[c] / illumination[c] + transmission[c] <= 1.0001, "\(label): outgoing energy exceeds incident light")
                } else {
                    try close(radiance[c], 0, "\(label): black illumination cannot emit")
                }
            }
        }
        let illumination = SIMD3<Float>(0.7, 0.5, 0.9)
        let origin = SIMD3<Float>(1, 1, -0.05)
        let direction = SIMD3<Float>(0, 0, -1)
        var count = 0
        for amount: Float in [0, 1, 20] {
            let concentration = SIMD3<Float>(0.2, 0.3, 0.4) * amount
            let field = try volume(device, concentration: concentration)
            for steps in [0, 1, 4, 8, 32, 64, 128] {
                for distance: Float in [0, 0.00001, 0.6] {
                    for source in [illumination, SIMD3<Float>(repeating: 0)] {
                        let actual = try evaluate(field, concentration, origin, direction, distance, steps, source)
                        try validate(actual, concentration: concentration, distance: distance, illumination: source,
                                     label: "cloud concentration \(amount), distance \(distance), steps \(steps)")
                        count += 1
                    }
                }
            }
        }
        // Outside the resident field means zero pigment, not clamped edge pigment;
        // clear-water extinction still applies over the caller-supplied path.
        let occupied = try volume(device, concentration: SIMD3(0.8, 0.2, 0.4))
        let outside: [(SIMD3<Float>, SIMD3<Float>)] = [
            (SIMD3(-0.1,1,-0.3), SIMD3(-1,0,0)), (SIMD3(2.1,1,-0.3), SIMD3(1,0,0)),
            (SIMD3(1,-0.1,-0.3), SIMD3(0,-1,0)), (SIMD3(1,2.1,-0.3), SIMD3(0,1,0)),
            (SIMD3(1,1,-0.9), SIMD3(0,0,-1)), (SIMD3(1,1,0.2), SIMD3(0,0,1))]
        for (start, direction) in outside {
            let actual = try evaluate(occupied, SIMD3(repeating: 0), start, direction, 0.2, 8, illumination)
            try validate(actual, concentration: SIMD3(repeating: 0), distance: 0.2, illumination: illumination, label: "outside pigment volume")
            count += 1
        }
        // Each pure species scatters colored light, but remains passive.
        for concentration: SIMD3<Float> in [SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(0,0,1)] {
            let field = try volume(device, concentration: concentration)
            let actual = try evaluate(field, concentration, origin, direction, 0.6, 8, illumination)
            try validate(actual, concentration: concentration, distance: 0.6, illumination: illumination, label: "pure pigment species")
            let scattering = actual[3]
            try require(min(scattering.x,scattering.y,scattering.z) >= 0 && max(scattering.x,scattering.y,scattering.z) > 0,
                        "pure pigment should scatter some light")
            try require(max(scattering.x,scattering.y,scattering.z) - min(scattering.x,scattering.y,scattering.z) > 0.000001,
                        "pure pigment scattering should have a colored spectrum")
            count += 1
        }
        let studyConcentration = SIMD3<Float>(0.8, 0.2, 0.4)
        let dark = try evaluate(occupied, studyConcentration, origin, direction, 0.6, 32, SIMD3(repeating: 0))
        let study = try evaluate(occupied, studyConcentration, origin, direction, 0.6, 32, SIMD3(repeating: 0), study: true)
        try validate(study, concentration: studyConcentration, distance: 0.6,
                     illumination: SIMD3(1,0.98,1.12), label: "study lighting source")
        for c in 0..<3 {
            try close(study[0][c], dark[0][c], "study lighting preserves physical transmittance")
            try close(study[2][c], dark[2][c], "study lighting preserves integrated pigment")
            try require(study[1][c] > 0 && dark[1][c] == 0, "study flag introduces the explicit studio source only")
        }
        count += 1
        checks += ["3D constant concentration: analytic Beer/scatter/column, thin/thick/zero distance and step-count invariance (0...128 requested steps)",
                   "3D cloud passivity, black-illumination limit and colored pure-pigment scattering",
                   "Outside all six resident volume faces uses clear-water transfer without edge pigment",
                   "Study lighting changes the incident source while preserving cloud transmittance and optical mass"]
        return count
    }

    static func run() throws {
        let shaderPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "native/Sources/Water.metal"
        let destination = URL(fileURLWithPath: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "native/artifacts/shader-checks", isDirectory: true)
        var source = try String(contentsOfFile: shaderPath, encoding: .utf8)
        let shaderDirectory = URL(fileURLWithPath: shaderPath).deletingLastPathComponent()
        for name in ["BedLife.metal", "Caustics.metal"] {
            let module = shaderDirectory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: module.path) {
                source = try String(contentsOf: module, encoding: .utf8) + "\n" + source
            }
        }
        let appended = """
        kernel void opticsContracts(device float4 *out [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
                                    texture2d<float> grid [[texture(0)]], texture2d<float> flat [[texture(1)]],
                                    uint id [[thread_position_in_grid]]) {
            if (id != 0) return;
            out[0] = float4(fresnel(1), 1);
            out[1] = float4(fresnel(.5f), 1);
            out[2] = float4(pigmentTau(float3(.2f, .3f, .4f)), 1);
            out[3] = float4(pigmentTau(float3(-2, 0, 0)), 1);
            out[4] = float4(toneMap(float3(16, 0, 0), 1, u), 1);
            out[5] = float4(toneMap(float3(.18f), 1, u), 1);
            out[6] = float4(toneMap(float3(20, 0, 10), 1, u), 1);
            out[7] = float4(toneMap(float3(0), 1, u), 1);
            out[8] = field(grid, float2(0));
            out[9] = field(grid, float2(2.f / 3.f, 1));
            out[10] = float4(intersectWater(flat, float3(1,1,2), float3(0,0,-1), u), 1);
        }
        fragment float4 transferContract(VertexOut in [[stage_in]], constant Uniforms &u [[buffer(0)]]) {
            return float4(u.optics.xyz, 1);
        }
        kernel void cloudOpticsContracts(device float4 *out [[buffer(0)]], constant Uniforms &u [[buffer(1)]],
                                         constant float4 *parameters [[buffer(2)]], texture3d<float> volume [[texture(0)]],
                                         uint id [[thread_position_in_grid]]) {
            if (id != 0) return;
            CloudOptics transfer = cloudTransfer(volume, parameters[0].xyz, parameters[1].xyz,
                                                parameters[0].w, u, uint(parameters[1].w));
            out[0] = float4(transfer.transmittance, 1);
            out[1] = float4(transfer.radiance, 1);
            out[2] = float4(transfer.column, 1);
            out[3] = float4(pigmentScattering(parameters[2].xyz), 1);
        }
        """
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw ShaderCheckFailure(description: "Metal device unavailable") }
        let options = MTLCompileOptions()
        options.mathMode = .fast
        let library = try device.makeLibrary(source: source + "\n" + appended, options: options)
        let compute = try device.makeComputePipelineState(function: library.makeFunction(name: "opticsContracts")!)
        let buffer = device.makeBuffer(length: 11 * 16, options: .storageModeShared)!
        var u = defaults()
        let grid = try texture(device, width: 3, height: 2, color: SIMD4(0, 0, 0, 1))
        let flatGrid = try texture(device, width: 4, height: 4, color: SIMD4(0, 0, 0, 1))
        let values: [SIMD4<Float>] = [0, 1, 2, 10, 11, 12].map { SIMD4($0, 0, 0, 1) }
        values.withUnsafeBytes { grid.replace(region: MTLRegionMake2D(0,0,3,2), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 3 * 16) }
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(compute)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&u, length: MemoryLayout<ARUniforms>.stride, index: 1)
        encoder.setTexture(grid, index: 0)
        encoder.setTexture(flatGrid, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed, "GPU compute command failed")
        let results = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: 11)
        for (channel, ior) in [1.331, 1.333, 1.337].enumerated() {
            try close(results[0][channel], Float(pow((ior - 1) / (ior + 1), 2)), "normal Fresnel")
            try close(results[1][channel], fresnelReference(0.5, ior), "oblique Fresnel")
        }
        let mass = SIMD3<Float>(0.2, 0.3, 0.4)
        for (channel, spectrum) in [SIMD3<Float>(2,0.3,0.25), SIMD3<Float>(0.65,1.8,0.65), SIMD3<Float>(0.3,0.45,1.8)].enumerated() {
            try close(results[2][channel], simd_dot(mass, spectrum) * 0.85, "pigment extinction")
            try close(results[3][channel], 0, "negative mass is transparent")
            try close(results[5][channel], tone(0.18), "neutral tone curve")
            try close(results[7][channel], 0, "black tone curve")
        }
        try close(results[4].x, tone(16), "red peak")
        try close(results[4].y, 0, "red stays saturated")
        try close(results[6].x, tone(20), "magenta peak")
        try close(results[6].y, 0, "magenta stays saturated")
        try close(results[6].z, tone(20) * 0.5, "HDR channel ratios")
        try close(results[8].x, 0, "lattice origin")
        try close(results[9].x, 11, "asymmetric lattice node and half-texel alignment")
        try close(results[10].x, 1, "flat surface intersection x")
        try close(results[10].y, 1, "flat surface intersection y")
        try close(results[10].z, 0, "flat surface intersection height")
        checks += ["GPU Fresnel against independent dielectric equations", "GPU pigment optical depth and empty/negative mass", "GPU tone curve, black and saturated HDR channel ratios", "GPU asymmetric field indexing, lattice sampling and flat-surface intersection"]

        let cloudCases = try cloudTransferChecks(device, queue, library)

        let post = try pipeline(device, library, fragment: "postFragment")
        let output = try texture(device)
        let exposure = try texture(device, width: 1, height: 1, color: SIMD4(1, 0, 0, 1))
        var renderedCases = 0
        for color: SIMD3<Float> in [SIMD3(0.18,0.18,0.18), SIMD3(20,0,10), SIMD3(0.1,0.5,1), SIMD3(0,0,0)] {
            let current = try texture(device, color: SIMD4(color, 1))
            for effects: Float in [0, 1] {
                for shift: Float in [0, 2] {
                    for dose: Float in [0, 4] {
                        u = defaults(); u.post.x = effects; u.time.w = shift; u.time.z = dose
                        try render(queue, pipeline: post, uniforms: u, textures: [current, current, exposure], output: output)
                        let pixels = read(output)
                        let peak = max(color.x, color.y, color.z)
                        let expected = peak == 0 ? SIMD3<Float>(repeating: 0) : color * (tone(peak) / peak)
                        for pixel in stride(from: 0, to: pixels.count, by: 4) {
                            for channel in 0..<3 { try close(pixels[pixel + channel], expected[channel], "constant postprocess output", tolerance: 0.0001) }
                            try close(pixels[pixel + 3], 1, "postprocess alpha")
                        }
                        renderedCases += 1
                    }
                }
            }
        }
        checks.append("Zero-subjective-dose postFragment preserves constant colors across effects/shift/trip (\(renderedCases) cases, all pixels)")

        // A newly allocated or reset history must be bypassed before reading it.
        let poison = try texture(device, color: SIMD4(Float.nan, Float.nan, Float.nan, 1))
        let held = try texture(device, color: SIMD4(0.18,0.18,0.18,1))
        u = defaults(); u.post.z = 0
        try render(queue, pipeline: post, uniforms: u, textures: [held,poison,exposure], output: output)
        let fresh = read(output)
        for pixel in stride(from: 0, to: fresh.count, by: 4) {
            for channel in 0..<3 { try close(fresh[pixel + channel], tone(0.18), "invalid previous history is bypassed", tolerance: 0.0001) }
        }
        checks.append("Reset history bypass remains finite even with deliberately nonfinite previous pixels")

        // A zero subjective dose must be an exact clean-image bypass even when
        // Trip, effects and the configured RGB radius are nonzero. Active layer
        // drift and diffraction are exercised separately by RetinaTests.
        let gradient = try texture(device, width: 128, height: 128)
        let gradientOutput = try texture(device, width: 128, height: 128)
        func sourceColor(_ x: Float, _ y: Float) -> SIMD3<Float> { SIMD3(0.1 + 0.7 * x / 127, 0.1 + 0.7 * y / 127, 0.2 + 0.4 * x / 127) }
        var gradientPixels = [SIMD4<Float>]()
        for y in 0..<128 { for x in 0..<128 { gradientPixels.append(SIMD4(sourceColor(Float(x), Float(y)), 1)) } }
        gradientPixels.withUnsafeBytes { gradient.replace(region: MTLRegionMake2D(0,0,128,128), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 128 * 16) }
        for effects: Float in [0, 1] {
            for shift: Float in [0, 2] {
                u = defaults(); u.viewport = SIMD4(128,128,1/128,1/128)
                u.post = SIMD4(effects,2,0.25,1); u.time.w = shift
                try render(queue, pipeline: post, uniforms: u, textures: [gradient,gradient,exposure], output: gradientOutput)
                let pixels = read(gradientOutput)
                for position: SIMD2<Int> in [SIMD2(64,64), SIMD2(52,70)] {
                    for channel in 0..<3 {
                        let color = sourceColor(Float(position.x), Float(position.y))
                        let peak = max(color.x,color.y,color.z)
                        let expected = color[channel] * tone(peak) / peak
                        try close(pixels[(position.y * 128 + position.x) * 4 + channel], expected,
                                  "zero subjective dose preserves affine image regardless of RGB radius", tolerance: 0.0001)
                    }
                }
            }
        }
        checks.append("Actual postFragment zero-dose bypass preserves affine RGB despite configured shift/effects")

        let transfer = try pipeline(device, library, fragment: "transferContract", format: .rgba8Unorm_srgb)
        let encoded = try texture(device, format: .rgba8Unorm_srgb)
        for value: Float in [0, 0.003_130_8, 0.18, 1] {
            u = defaults(); u.optics = SIMD4(value, value, value, 1)
            try render(queue, pipeline: transfer, uniforms: u, textures: [], output: encoded)
            var bytes = [UInt8](repeating: 0, count: encoded.width * encoded.height * 4)
            bytes.withUnsafeMutableBytes { encoded.getBytes($0.baseAddress!, bytesPerRow: encoded.width * 4, from: MTLRegionMake2D(0,0,encoded.width,encoded.height), mipmapLevel: 0) }
            let expected = value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1 / 2.4) - 0.055
            for pixel in stride(from: 0, to: bytes.count, by: 4) {
                for c in 0..<3 { try require(abs(Int(bytes[pixel + c]) - Int((expected * 255).rounded())) <= 1, "sRGB transfer toe/midgray/white") }
            }
        }
        checks.append("sRGB attachment transfer: zero, 0.0031308 toe, 0.18 midgray and white")

        let waterPipeline = try pipeline(device, library, fragment: "waterFragment")
        let flat = try texture(device, width: 8, height: 8, color: SIMD4(0,0,0,0))
        let pigment = try texture(device, width: 8, height: 8, color: SIMD4(0.2,0.3,0.4,0.7))
        let bed = try texture(device, width: 8, height: 8, color: SIMD4(0.8,0,0,0))
        let blackSky = try texture(device, width: 16, height: 8, color: SIMD4(0,0,0,1))
        let whiteLight = try texture(device, width: 8, height: 8, color: SIMD4(1,1,1,1))
        let emptyVolume = try volume(device, concentration: SIMD3(repeating: 0))
        u = defaults(); u.illumination = SIMD4(0,0,0,0)
        try render(queue, pipeline: waterPipeline, uniforms: u, textures: [flat,pigment,bed,blackSky,whiteLight,emptyVolume], output: output)
        let darkness = read(output)
        for pixel in stride(from: 0, to: darkness.count, by: 4) {
            for c in 0..<3 { try close(darkness[pixel + c], 0, "unlit dyed water cannot emit light") }
        }
        checks.append("Actual waterFragment: absorbing dyed water with no incident/emitted light stays black and finite")

        // A vertical view through a homogeneous 3D cloud has an exact reference.
        // The 2D pigment texture deliberately remains nonzero: it must not be
        // applied again on top of the new volume's optical mass.
        let concentration = SIMD3<Float>(0.2, 0.3, 0.4)
        let occupiedVolume = try volume(device, concentration: concentration)
        let emission = SIMD3<Float>(0.3, 0.4, 0.5)
        let emissiveBed = try texture(device, width: 8, height: 8, color: SIMD4(0.8, emission.x, emission.y, emission.z))
        let pixelOutput = try texture(device, width: 1, height: 1)
        u = defaults(); u.viewport = SIMD4(1,1,1,1); u.illumination = SIMD4(repeating: 0)
        try render(queue, pipeline: waterPipeline, uniforms: u,
                   textures: [flat,pigment,emissiveBed,blackSky,whiteLight,occupiedVolume], output: pixelOutput)
        let bedView = read(pixelOutput)
        let absorption = SIMD3<Float>(simd_dot(concentration, SIMD3(2,0.30,0.25)),
                                      simd_dot(concentration, SIMD3(0.65,1.8,0.65)),
                                      simd_dot(concentration, SIMD3(0.30,0.45,1.8))) * 0.85
        let scattering = SIMD3<Float>(simd_dot(concentration, SIMD3(0.03,0.90,1.10)),
                                      simd_dot(concentration, SIMD3(0.85,0.04,0.65)),
                                      simd_dot(concentration, SIMD3(1.10,0.50,0.04))) * 0.32
        let sigma = SIMD3<Float>(1.2,1.6,1.2) * 0.2 + SIMD3<Float>(repeating: 0.015) + absorption + scattering
        for (c, ior) in [1.331,1.333,1.337].enumerated() {
            let expected = (1 - fresnelReference(1, ior)) / Float(ior * ior) * emission[c] * 0.96 * exp(-sigma[c] * 0.7)
            try close(bedView[c], expected, "actual volume optical path to emitting bed (one view leg)", tolerance: 0.0001)
        }
        try render(queue, pipeline: waterPipeline, uniforms: u,
                   textures: [flat,pigment,bed,blackSky,whiteLight,occupiedVolume], output: pixelOutput)
        let unlitCloud = read(pixelOutput)
        for c in 0..<3 { try close(unlitCloud[c], 0, "actual colored cloud with no illumination cannot emit") }
        checks.append("Actual waterFragment uses 3D Beer attenuation once for emissive bed and keeps an unlit occupied cloud black")

        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let hash = SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
        let report: [String: Any] = ["status": "PASS", "device": device.name, "math_mode": "fast (production)",
                                   "created_utc": ISO8601DateFormatter().string(from: Date()), "shader": shaderPath, "shader_sha256": hash,
                                   "checks": checks, "post_render_cases": renderedCases, "cloud_transfer_cases": cloudCases,
                                   "note": "Offscreen GPU correctness checks; no displayed FPS or performance claim."]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: destination.appendingPathComponent("report.json"))
        var lines = ["# Actual Metal shader contracts", "", "Device: \(device.name)", "", "| Contract | Result |", "| --- | --- |"]
        lines += checks.map { "| \($0) | PASS |" }
        lines += ["", "Production shader SHA256: `\(hash)`", "", "Offscreen correctness checks; no displayed FPS or performance claim."]
        let markdown = lines.joined(separator: "\n") + "\n"
        try markdown.write(to: destination.appendingPathComponent("report.md"), atomically: true, encoding: .utf8)
        print(markdown)
    }
    static func main() {
        do { try run() }
        catch { FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8)); exit(1) }
    }
}
