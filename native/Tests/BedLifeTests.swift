import Foundation
import Metal
import ImageIO
import UniformTypeIdentifiers
import simd
import CryptoKit

struct BedLifeFailure: Error, CustomStringConvertible { let description: String }

/// Runs the actual production material on Metal. These are bounded correctness
/// probes, not performance benchmarks or a second CPU implementation.
@main struct BedLifeTests {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw BedLifeFailure(description: message) }
    }

    static func main() throws {
        let sourceURL = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "native/Sources/BedLife.metal")
        let output = URL(fileURLWithPath: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "native/artifacts/bed-life-checks")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let production = try String(contentsOf: sourceURL, encoding: .utf8)
        let probe = """
        kernel void bedLifeProbe(device const float4 *points [[buffer(0)]],
                                 device float4 *results [[buffer(1)]],
                                 constant float4 &origin [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
            float4 p = points[index];
            BedMaterial m = animatedBed(p.xy + origin.xy, p.z, float4(0.48f,0.03f,0.02f,0.04f), p.w);
            results[3*index] = float4(m.albedo, m.roughness);
            results[3*index+1] = float4(m.emission, 0);
            results[3*index+2] = float4(m.slope, 0, 0);
        }
        """
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw BedLifeFailure(description: "No Metal device")
        }
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) { options.mathMode = .fast } else { options.fastMathEnabled = true }
        let library = try device.makeLibrary(source: production + "\n" + probe, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "bedLifeProbe")!)
        func evaluate(_ points: [SIMD4<Float>], origin: SIMD2<Float> = .zero) throws -> [SIMD4<Float>] {
            let inputs = points.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
            let results = device.makeBuffer(length: points.count * 3 * 16, options: .storageModeShared)!
            let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputs, offset: 0, index: 0)
            encoder.setBuffer(results, offset: 0, index: 1)
            var origin = SIMD4<Float>(origin.x, origin.y, 0, 0)
            encoder.setBytes(&origin, length: 16, index: 2)
            encoder.dispatchThreads(MTLSize(width: points.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(128, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            try require(command.status == .completed, "Bed material GPU probe failed: \(String(describing: command.error))")
            return Array(UnsafeBufferPointer(start: results.contents().bindMemory(to: SIMD4<Float>.self, capacity: points.count * 3), count: points.count * 3))
        }
        let side = 192
        let points = (0..<side * side).map { index in
            SIMD4<Float>(Float(index % side) / 32 - 3, Float(index / side) / 32 - 3, 17, 0.002)
        }
        let initial = try evaluate(points)
        for index in 0..<points.count {
            let a = initial[3 * index], e = initial[3 * index + 1], n = initial[3 * index + 2]
            try require((0..<4).allSatisfy { a[$0].isFinite && e[$0].isFinite && n[$0].isFinite }, "Nonfinite material")
            try require((0..<3).allSatisfy { a[$0] >= 0.0199 && a[$0] <= 0.9201 && e[$0] >= 0 && e[$0] <= 0.5501 }, "Reflectance/emission bounds")
            try require(abs(n.x) <= 0.3201 && abs(n.y) <= 0.3201 && a.w >= 0.3 && a.w <= 0.95, "Relief/roughness bounds")
        }
        let later = try evaluate(points.map { SIMD4<Float>($0.x, $0.y, $0.z + 10, $0.w) })
        let nearby = try evaluate(points.map { SIMD4<Float>($0.x, $0.y, $0.z + 0.0001, $0.w) })
        func difference(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> (mean: Float, maximum: Float) {
            var sum: Float = 0, maximum: Float = 0
            for (x, y) in zip(a, b) { for c in 0..<4 { let d = abs(x[c] - y[c]); sum += d; maximum = max(maximum, d) } }
            return (sum / Float(a.count * 4), maximum)
        }
        let animation = difference(initial, later), continuity = difference(initial, nearby)
        try require(animation.mean > 0.001, "Living material must visibly evolve")
        try require(continuity.maximum < 0.003, "Material animation has a temporal jump")
        let origin = SIMD2<Float>(1.5, -2.25)
        let rebased = try evaluate(points.map { SIMD4<Float>($0.x - origin.x, $0.y - origin.y, $0.z, $0.w) }, origin: origin)
        try require(difference(initial, rebased).maximum == 0, "World rebase moved the bed artwork")
        let coarse = try evaluate(points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0.2) })
        try require(coarse.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "Coarse material is nonfinite")
        for index in 1..<points.count {
            try require(coarse[index * 3] == coarse[0] && coarse[index * 3 + 1] == coarse[1] && coarse[index * 3 + 2] == coarse[2],
                        "Unresolvable fine material must fade to a stable average")
        }
        var seams: [SIMD4<Float>] = []
        for cell in -4...4 { for row in 0..<128 {
            let x = Float(cell) * 0.72, y = Float(row) / 32 - 2
            seams.append(SIMD4(x - 0.00001, y, 17, 0.002)); seams.append(SIMD4(x + 0.00001, y, 17, 0.002))
        } }
        let edges = try evaluate(seams)
        var seamDifference: Float = 0
        for pair in stride(from: 0, to: seams.count, by: 2) {
            for layer in 0..<3 { for channel in 0..<4 {
                seamDifference = max(seamDifference, abs(edges[pair * 3 + layer][channel] - edges[(pair + 1) * 3 + layer][channel]))
            } }
        }
        try require(seamDifference < 0.005, "Organism cell boundary is visible")

        func savePreview(_ path: URL, time: Float) throws {
            let width = 768, height = 768
            let p = (0..<width * height).map { i in SIMD4<Float>(Float(i % width) * 3 / Float(width) - 1.5,
                                                               Float(i / width) * 3 / Float(height) - 1.5, time, 3 / Float(width)) }
            let sample = try evaluate(p)
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            let light = simd_normalize(SIMD3<Float>(-0.5, 0.3, 1))
            for i in 0..<p.count {
                let n = sample[3 * i + 2]
                let normal = simd_normalize(SIMD3<Float>(-n.x, -n.y, 1))
                let illumination = 0.25 as Float + 0.75 * max(0, simd_dot(normal, light))
                for c in 0..<3 {
                    let linear = sample[3 * i][c] * illumination + sample[3 * i + 1][c]
                    let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
                    rgba[4 * i + c] = UInt8(max(0, min(255, encoded * 255)))
                }
            }
            let provider = CGDataProvider(data: Data(rgba) as CFData)!
            let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            let destination = CGImageDestinationCreateWithURL(path as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, nil)
            try require(CGImageDestinationFinalize(destination), "Cannot write bed material preview")
        }
        try savePreview(output.appendingPathComponent("material-t17.png"), time: 17)
        try savePreview(output.appendingPathComponent("material-t27.png"), time: 27)
        let report: [String: Any] = ["device": device.name, "checks": 6, "sample_points": points.count,
            "maximum_nearby_time_difference": continuity.maximum, "mean_ten_second_change": animation.mean,
            "maximum_cell_seam_difference": seamDifference, "rebase_difference": 0,
            "source_sha256": SHA256.hash(data: Data(production.utf8)).map { String(format: "%02x", $0) }.joined()]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("report.json"))
        print("PASS: bounded material, animation, temporal continuity, exact world rebase, coarse filtering, cell continuity")
        print("Saved material previews and correctness report: \(output.path)")
    }
}
