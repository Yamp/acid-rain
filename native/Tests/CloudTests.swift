import Foundation
import Metal
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

private struct CloudTestUniforms {
    var step = SIMD4<Float>(0, 0, 0, 0)
    var rebase = SIMD4<Float>(repeating: 0)
    var domain = SIMD4<Float>(2, 2, -0.8, 0.9)
    var options = SIMD4<Float>(0, 1, 0.00001, 0)
}

private struct CloudTestDrop {
    var position = SIMD4<Float>(1, 1, 0, 0.016)
    var pigment = SIMD4<Float>(0.1, 0.8, 0.1, 0.035)
    var dynamics = SIMD4<Float>(0.024, 0.02, 0, 0.25)
    var identity = SIMD4<UInt32>(1, 0, 1, 1)
}

private struct CloudTestFailure: Error, CustomStringConvertible {
    let description: String
}

/// Executes the production kernels, checking tracer invariants, not shader source text.
@main struct CloudTests {
    static let width = 96, height = 64, depth = 32
    static let cellVolume = 2.0 / 96 * 2.0 / 64 * 0.9 / 32
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw CloudTestFailure(description: message) }
    }
    static func volume(_ device: MTLDevice) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = width; descriptor.height = height; descriptor.depth = depth
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = device.makeTexture(descriptor: descriptor)!
        upload(texture, Array(repeating: SIMD4<Float>(repeating: 0), count: width * height * depth))
        return texture
    }
    static func flat(_ device: MTLDevice, _ value: SIMD4<Float>) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 8, height: 8, mipmapped: false)
        descriptor.storageMode = .shared; descriptor.usage = .shaderRead
        let texture = device.makeTexture(descriptor: descriptor)!
        let pixels = Array(repeating: value, count: 64)
        pixels.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8 * 16) }
        return texture
    }
    static func upload(_ texture: MTLTexture, _ values: [SIMD4<Float>]) {
        values.withUnsafeBytes {
            texture.replace(region: MTLRegionMake3D(0, 0, 0, width, height, depth), mipmapLevel: 0, slice: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: width * 16, bytesPerImage: width * height * 16)
        }
    }
    static func read(_ texture: MTLTexture) -> [SIMD4<Float>] {
        var pixels = Array(repeating: SIMD4<Float>(repeating: 0), count: width * height * depth)
        pixels.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: width * 16, bytesPerImage: width * height * 16,
                             from: MTLRegionMake3D(0, 0, 0, width, height, depth), mipmapLevel: 0, slice: 0)
        }
        return pixels
    }
    static func mass(_ values: [SIMD4<Float>]) -> SIMD3<Double> {
        values.reduce(SIMD3<Double>(repeating: 0)) { $0 + SIMD3(Double($1.x), Double($1.y), Double($1.z)) * cellVolume }
    }
    static func centroid(_ values: [SIMD4<Float>]) -> SIMD3<Double> {
        var moment = SIMD3<Double>(repeating: 0), weight = 0.0
        for (index, value) in values.enumerated() {
            let x = index % width, y = index / width % height, z = index / (width * height)
            let amount = Double(value.x + value.y + value.z)
            let px = (Double(x) + 0.5) * 2.0 / Double(width)
            let py = (Double(y) + 0.5) * 2.0 / Double(height)
            let pz = -0.8 + (Double(z) + 0.5) * 0.9 / Double(depth)
            moment += SIMD3<Double>(px, py, pz) * amount
            weight += amount
        }
        return moment / max(weight, 1e-30)
    }
    static func diagnostic(_ values: [SIMD4<Float>], name: String, directory: String) throws {
        let destination = URL(fileURLWithPath: directory, isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for kind in ["xy-projection", "xz-slice", "xz-projection"] {
            let panelHeight = kind == "xy-projection" ? height : depth
            var panel = [Float](repeating: 0, count: width * panelHeight)
            for z in 0..<depth { for y in 0..<height { for x in 0..<width {
                let p = values[x + width * (y + height * z)]
                let concentration = p.x + p.y + p.z
                if kind == "xy-projection" { panel[x + width * (height - 1 - y)] += concentration }
                else if kind == "xz-projection" { panel[x + width * (depth - 1 - z)] += concentration }
                else if y == height / 2 { panel[x + width * (depth - 1 - z)] = concentration }
            } } }
            let peak = max(panel.max() ?? 1, 1e-20)
            let scale = 6, pixelsWide = width * scale, pixelsHigh = panelHeight * scale
            var rgba = [UInt8](repeating: 255, count: pixelsWide * pixelsHigh * 4)
            for y in 0..<pixelsHigh { for x in 0..<pixelsWide {
                let intensity = sqrt(max(0, panel[x / scale + width * (y / scale)] / peak))
                let offset = 4 * (x + pixelsWide * y)
                rgba[offset] = UInt8(min(255, 255 * max(0, intensity * 2 - 1)))
                rgba[offset + 1] = UInt8(min(255, 255 * intensity))
                rgba[offset + 2] = UInt8(min(255, 255 * sqrt(intensity)))
            } }
            let provider = CGDataProvider(data: Data(rgba) as CFData)!
            let image = CGImage(width: pixelsWide, height: pixelsHigh, bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: pixelsWide * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
                                decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            let url = destination.appendingPathComponent("\(name)-\(kind).png")
            let writer = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(writer, image, nil)
            try require(CGImageDestinationFinalize(writer), "diagnostic PNG failed")
        }
    }
    static func run() throws {
        let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "native/Sources/Cloud.metal"
        let source = try String(contentsOfFile: path, encoding: .utf8)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw CloudTestFailure(description: "Metal unavailable")
        }
        let options = MTLCompileOptions(); options.mathMode = .fast
        let library = try device.makeLibrary(source: source, options: options)
        let names = ["cloudNormalize", "cloudPrepare", "cloudVelocity", "cloudAdvect"]
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in names { pipelines[name] = try device.makeComputePipelineState(function: library.makeFunction(name: name)!) }
        let a = volume(device), b = volume(device), velocity = volume(device)
        let water = flat(device, SIMD4(0, 0, 0, 0)), flow = flat(device, SIMD4(0, 0, 0, 0.7))
        let normalizations = device.makeBuffer(length: 512 * 16, options: .storageModeShared)!
        func dispatch(_ name: String, _ input: MTLTexture, _ output: MTLTexture, _ uniforms: CloudTestUniforms,
                      _ drops: [CloudTestDrop] = [], _ flowTexture: MTLTexture? = nil) throws {
            let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipelines[name]!)
            var u = uniforms
            encoder.setBytes(&u, length: MemoryLayout<CloudTestUniforms>.stride, index: 0)
            let records = drops.isEmpty ? [CloudTestDrop()] : drops
            let dropBuffer = records.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
            encoder.setBuffer(dropBuffer, offset: 0, index: 1)
            encoder.setBuffer(normalizations, offset: 0, index: 2)
            encoder.setTexture(input, index: 0); encoder.setTexture(output, index: 1)
            encoder.setTexture(velocity, index: 2); encoder.setTexture(flowTexture ?? flow, index: 3)
            encoder.setTexture(water, index: 4)
            if name == "cloudNormalize" {
                encoder.dispatchThreads(MTLSize(width: max(1, drops.count), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            } else {
                encoder.dispatchThreads(MTLSize(width: width, height: height, depth: depth), threadsPerThreadgroup: MTLSize(width: 8, height: 4, depth: 2))
            }
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed, "\(name): \(String(describing: command.error))")
        }
        var u = CloudTestUniforms(), drop = CloudTestDrop()
        var checks: [String] = []
        u.step.w = 1; u.rebase = SIMD4(0, 0, 1, 1)
        for position: SIMD2<Float> in [SIMD2(1, 1), SIMD2(0.006, 0.004), SIMD2(1.994, 1.996)] {
            drop.position.x = position.x; drop.position.y = position.y
            try dispatch("cloudNormalize", a, b, u, [drop])
            try dispatch("cloudPrepare", a, b, u, [drop])
            let values = read(b), integral = mass(values)
            for channel in 0..<3 {
                let expected = Double(drop.pigment[channel] * drop.pigment.w)
                try require(abs(integral[channel] - expected) < 2e-7, "source mass near \(position): \(integral), expected species \(channel)=\(expected)")
            }
            try require(values.allSatisfy { $0.x >= 0 && $0.y >= 0 && $0.z >= 0 && $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "invalid source")
        }
        checks.append("discrete source mass preserved at center and clipped XY edges")
        drop.position = SIMD4(1, 1, 0, 0.016)
        try dispatch("cloudNormalize", a, b, u, [drop]); try dispatch("cloudPrepare", a, b, u, [drop])
        let initial = read(b), initialMass = mass(initial)
        u.rebase = SIMD4(repeating: 0)
        try dispatch("cloudPrepare", b, a, u, [drop])
        try require(read(a) == initial, "repeated render snapshot reinjects pigment")
        checks.append("source-free prepare is an exact copy (no repeated-snapshot reinjection)")
        u.step = SIMD4(0, 0.2, 0, 1)
        try dispatch("cloudVelocity", b, a, u, [drop])
        let speeds = read(velocity)
        let maximum = speeds.reduce(Float(0)) { max($0, abs($1.x), abs($1.y), abs($1.z)) }
        try require(maximum > 0.08 && maximum <= 0.350001, "wake absent or CFL velocity bound broken: \(maximum)")
        checks.append("descending wake is active and bounded for dt<=0.008")
        u.step.x = 0.008
        var input = b, output = a
        for index in 0..<160 {
            u.step.y = Float(index) * u.step.x
            if index % 4 == 0 { try dispatch("cloudVelocity", input, output, u, [drop]) }
            try dispatch("cloudAdvect", input, output, u)
            swap(&input, &output)
        }
        let transported = read(input), afterMass = mass(transported)
        try require(transported.allSatisfy { $0.x >= -1e-8 && $0.y >= -1e-8 && $0.z >= -1e-8 && $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "advection is negative/nonfinite")
        for channel in 0..<3 {
            try require(abs(afterMass[channel] - initialMass[channel]) < 1e-7, "transport changed species mass \(initialMass) -> \(afterMass)")
        }
        try require(centroid(transported).z < centroid(initial).z - 0.035, "drop did not descend into a cloud")
        checks.append("160 MUSCL steps preserve every species mass, positivity and finite values; cloud descends")
        let beforePause = read(input)
        u.step.x = 0; u.step.z = 4
        try dispatch("cloudAdvect", input, output, u)
        try require(read(output) == beforePause, "paused volume changes")
        checks.append("paused transport and bleaching are bitwise frozen")
        u.step.x = 0.008; u.step.z = 2
        try dispatch("cloudAdvect", input, output, u)
        let bleached = mass(read(output)), factor = exp(-log(2) * Double(u.step.x * u.step.z) / 8)
        for channel in 0..<3 { try require(abs(bleached[channel] - afterMass[channel] * factor) < 2e-8, "bleach half-life mismatch") }
        checks.append("bleaching uses physical half-life 8/rate seconds")
        var marker = Array(repeating: SIMD4<Float>(repeating: 0), count: width * height * depth)
        marker[10 + width * (11 + height * 12)] = SIMD4(2, 3, 4, 0)
        upload(a, marker); u.rebase = SIMD4(2, -3, 0, 0); u.step.w = 0
        try dispatch("cloudPrepare", a, b, u)
        let shifted = read(b)
        try require(shifted[8 + width * (14 + height * 12)] == SIMD4(2, 3, 4, 0), "world rebase has wrong sign or interpolation")
        try require(mass(shifted) == mass(marker), "interior rebase altered mass")
        u.rebase = SIMD4(1000, -1000, 0, 0)
        try dispatch("cloudPrepare", a, b, u)
        try require(read(b).allSatisfy { $0 == SIMD4<Float>(repeating: 0) }, "large rebase reads outside grid")
        u.rebase = SIMD4(0, 0, 1, 0)
        try dispatch("cloudPrepare", a, b, u)
        try require(read(b).allSatisfy { $0 == SIMD4<Float>(repeating: 0) }, "revision clear retains pigment")
        checks.append("integer world rebase, exposed-cell clearing, large jump and generation reset")
        // An independent uniform-flow moment law catches axis, sign, cell size,
        // or per-frame/physical-second confusion despite conserved total mass.
        upload(a, initial)
        let uniformVelocity = SIMD4<Float>(0.12, -0.03, -0.08, 1)
        upload(velocity, Array(repeating: uniformVelocity, count: width * height * depth))
        u = CloudTestUniforms(); u.step.x = 0.008; u.options.z = 0
        input = a; output = b
        for _ in 0..<80 {
            try dispatch("cloudAdvect", input, output, u); swap(&input, &output)
        }
        let shiftedCenter = centroid(read(input)), startCenter = centroid(initial)
        for axis in 0..<3 {
            let expected = startCenter[axis] + Double(uniformVelocity[axis]) * Double(u.step.x) * 80
            // TVD slopes on a grid-scale Gaussian carry a sub-cell moment error;
            // 2 mm is < 0.1 of the smallest voxel.
            try require(abs(shiftedCenter[axis] - expected) < 0.002, "uniform-flow centroid axis \(axis): \(shiftedCenter[axis]) != \(expected)")
        }
        checks.append("uniform-flow centroid follows velocity times physical seconds on all three axes")
        // Harsh discontinuities and opposing velocities exercise the strongest
        // outgoing Courant budget, not only a smooth benign Gaussian.
        var checker = marker, worstVelocity = marker
        for index in checker.indices {
            let x = index % width, y = index / width % height, z = index / (width * height)
            checker[index] = ((x + y + z) % 2 == 0) ? SIMD4(1, 0.7, 0.2, 0) : SIMD4(0, 0, 0, 0)
            worstVelocity[index] = SIMD4(x % 3 == 0 ? -0.35 : 0.35, y % 3 == 0 ? -0.35 : 0.35, z % 3 == 0 ? -0.35 : 0.35, 1)
        }
        upload(a, checker); upload(velocity, worstVelocity)
        u.options.z = 0.00001; input = a; output = b
        for _ in 0..<64 { try dispatch("cloudAdvect", input, output, u); swap(&input, &output) }
        let harsh = read(input), harshMass = mass(harsh), checkerMass = mass(checker)
        try require(harsh.allSatisfy { $0.x >= 0 && $0.y >= 0 && $0.z >= 0 && $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "worst-CFL case loses positivity")
        for axis in 0..<3 { try require(abs(harshMass[axis] - checkerMass[axis]) < 2e-6, "worst-CFL case loses mass") }
        checks.append("checkerboard and reversing maximal flow preserve positivity and mass at supported CFL limit")
        // Rebase changes local coordinates but must not shift a physical curl.
        // Exercise both an ordinary shift and positive/negative 2pi wrapping.
        let cellX = 2.0 / Double(width), cellY = 2.0 / Double(height)
        let period = 2 * Double.pi
        let flatMovingFlow = flat(device, SIMD4(0.03, -0.015, 0, 0.7))
        var rebaseVelocityError: Float = 0
        for origin in [SIMD2<Double>(0.7, -0.9), SIMD2<Double>(period - cellX / 2, -period + cellY / 2)] {
            u = CloudTestUniforms(); u.step.y = 1.234
            u.options.x = Float(origin.x.truncatingRemainder(dividingBy: period))
            u.options.w = Float(origin.y.truncatingRemainder(dividingBy: period))
            try dispatch("cloudVelocity", a, b, u, [], flatMovingFlow)
            let before = read(velocity)
            u.options.x = Float((origin.x + cellX).truncatingRemainder(dividingBy: period))
            u.options.w = Float((origin.y - cellY).truncatingRemainder(dividingBy: period))
            try dispatch("cloudVelocity", a, b, u, [], flatMovingFlow)
            let after = read(velocity)
            var sensitivity: Float = 0
            for z in 4..<27 { for y in 1..<height { for x in 0..<(width - 1) {
                let index = x + width * (y + height * z)
                let oldIndex = (x + 1) + width * ((y - 1) + height * z)
                for axis in 0..<3 {
                    rebaseVelocityError = max(rebaseVelocityError, abs(after[index][axis] - before[oldIndex][axis]))
                    sensitivity = max(sensitivity, abs(after[index][axis] - before[index][axis]))
                }
                try require(after[index].w == before[oldIndex].w, "rebase changed fluid occupancy in a flat column")
            } } }
            try require(sensitivity > 0.002, "world-origin velocity test lacks phase sensitivity")
        }
        try require(rebaseVelocityError < 0.000002, "curl changes at fixed world point after rebasing: \(rebaseVelocityError) m/s")
        checks.append("world curl velocity remains continuous across XY voxel rebase and positive/negative 2pi origin wrapping")
        if CommandLine.arguments.count > 3 {
            upload(a, initial); input = a; output = b
            u = CloudTestUniforms(); u.step = SIMD4(0.008, 0, 0, 1)
            try diagnostic(initial, name: "age-0.000", directory: CommandLine.arguments[3])
            for index in 0..<250 {
                u.step.y = Float(index) * u.step.x
                try dispatch("cloudVelocity", input, output, u, [drop])
                try dispatch("cloudAdvect", input, output, u); swap(&input, &output)
                if [39, 124, 249].contains(index) {
                    let name = String(format: "age-%.3f", Float(index + 1) * u.step.x)
                    try diagnostic(read(input), name: name, directory: CommandLine.arguments[3])
                }
            }
        }
        let report: [String: Any] = ["device": device.name, "grid": [width, height, depth], "checks": checks, "passed": true, "initial_mass": [initialMass.x, initialMass.y, initialMass.z], "transport_mass": [afterMass.x, afterMass.y, afterMass.z], "maximum_world_rebase_velocity_error_m_s": rebaseVelocityError]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        if CommandLine.arguments.count > 2 {
            let destination = URL(fileURLWithPath: CommandLine.arguments[2])
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: destination)
        }
        print(String(decoding: data, as: UTF8.self))
    }
    static func main() {
        do { try run() } catch { fputs("Cloud tests failed: \(error)\n", stderr); exit(1) }
    }
}
