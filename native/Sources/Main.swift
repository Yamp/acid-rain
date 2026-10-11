import AppKit
import Metal
import ImageIO
import UniformTypeIdentifiers
import QuartzCore

struct LaunchOptions {
    var width = 1280
    var height = 800
    var frames = 240
    var warmup = 30
    var seed: UInt64 = 0xAC1D2026
    var quality: RenderQuality = .high
    var dropDemo = false
    var cloudStudy = false
    var showFallingDrops = false
    var subjectiveIntensity: Float = 1
    var captureSeries: String?
    var captureEvery = 30
    var capture: String?
    var benchmark: String?
    var smoke: Double?
    var fullscreen = false
    var exerciseControls = false
    var controls: [(UInt32, Float)] = []

    init(arguments: [String]) throws {
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            func value() throws -> String {
                index += 1
                guard index < arguments.count else { throw NativeFailure("Missing value for \(flag).") }
                return arguments[index]
            }
            switch flag {
            case "--width": width = Int(try value()) ?? 0
            case "--height": height = Int(try value()) ?? 0
            case "--frames": frames = Int(try value()) ?? 0
            case "--warmup": warmup = Int(try value()) ?? -1
            case "--capture": capture = try value()
            case "--drop-demo": dropDemo = true
            case "--cloud-study": cloudStudy = true
            case "--show-drops": showFallingDrops = true
            case "--hide-drops": showFallingDrops = false
            case "--lsd":
                guard let amount = Float(try value()), amount.isFinite, (0...2).contains(amount) else {
                    throw NativeFailure("--lsd must be between 0 and 2.")
                }
                subjectiveIntensity = amount
            case "--capture-series": captureSeries = try value()
            case "--capture-every": captureEvery = Int(try value()) ?? 0
            case "--benchmark": benchmark = try value()
            case "--smoke": smoke = Double(try value()) ?? 0
            case "--seed": guard let parsed = UInt64(try value()) else { throw NativeFailure("Seed must be an unsigned integer.") }; seed = parsed
            case "--quality":
                let q = try value().lowercased()
                guard let parsed = RenderQuality.allCases.first(where: { $0.name.lowercased() == q }) else { throw NativeFailure("Quality: auto, high, balanced, low or native.") }
                quality = parsed
            case "--mode":
                let mode = try value()
                guard let number = ["classic", "drain", "rapids"].firstIndex(of: mode) else { throw NativeFailure("Mode: classic, drain or rapids.") }
                controls.append((9, Float(number)))
            case "--scene":
                let scene = try value()
                guard let number = ["journey", "acid", "aurora", "storm", "abyss", "lava", "prism"].firstIndex(of: scene) else { throw NativeFailure("Unknown scene \(scene).") }
                controls.append((10, Float(number)))
            case "--speed", "--trip", "--rain", "--drop-size", "--bleach", "--rgb-shift":
                guard let amount = Float(try value()), amount.isFinite else { throw NativeFailure("Invalid numeric value for \(flag).") }
                let codes: [String: UInt32] = ["--speed": 2, "--trip": 6, "--rain": 3, "--drop-size": 4, "--bleach": 5, "--rgb-shift": 7]
                controls.append((codes[flag]!, amount))
            case "--no-travel": controls.append((11, 0))
            case "--no-current": controls.append((12, 0))
            case "--no-effects": controls.append((13, 0))
            case "--fullscreen": fullscreen = true
            case "--exercise-controls": exerciseControls = true
            case "--help", "-h":
                print("""
                Acid Rain · Metal
                  --fullscreen                 Open in macOS fullscreen
                  --quality high|auto|balanced|low|native
                  --width N --height N          Window points / capture pixels
                  --capture path.png            Render real Metal frames offscreen
                  --drop-demo                   Isolated size-3 drop with observation lighting
                  --cloud-study                 Observation lighting; p toggles at runtime
                  --show-drops                  Show falling drops (hidden by default); a toggles
                  --hide-drops                  Hide only airborne drops; impacts remain active
                  --lsd N                       Subjective effects 0...2 (default 1); n/m adjust
                  --capture-series directory    Save a deterministic drop progression
                  --capture-every N             Series interval in frames (default 30)
                  --frames N --warmup N         Capture frames; benchmark warm-up frames
                  --benchmark path.json         Save measured CPU / GPU timings
                  --smoke seconds               Open a real window, then exit
                  --exercise-controls           Exercise controls and resize during smoke
                  --seed N --mode classic|drain|rapids --scene journey|acid|aurora|storm|abyss|lava|prism
                  --speed N --trip N --rain N --drop-size N --bleach N --rgb-shift N
                  --no-travel --no-current --no-effects
                Press h in the window for every keyboard control.
                """)
                exit(0)
            default: if !flag.hasPrefix("-psn_") { throw NativeFailure("Unknown option: \(flag). Use --help.") }
            }
            index += 1
        }
        guard (64...8192).contains(width), (64...8192).contains(height), width * height <= 33_554_432,
              (1...100_000).contains(frames), (1...100_000).contains(captureEvery), warmup >= 0, warmup < frames,
              smoke == nil || ((smoke ?? 0).isFinite && (smoke ?? 0) > 0) else {
            throw NativeFailure("Invalid render dimensions, frame count, warm-up or smoke duration.")
        }
        controls.append((8, quality.shaderLevel))
    }

    func apply(to core: CoreBridge) throws {
        for (command, value) in controls { try core.commandOffline(command, value: value) }
        if dropDemo {
            try core.commandOffline(3, value: 0)
            try core.commandOffline(4, value: 3)
            if core.current().state.travel != 0 { try core.commandOffline(11) }
            if core.current().state.current != 0 { try core.commandOffline(12) }
            try core.commandOffline(16)
            let s = core.current().state
            try core.dropOffline(world: SIMD2<Double>(s.world_origin_x + Double(s.uniforms.camera.x),
                s.world_origin_y + Double(s.uniforms.camera.y)), size: 3)
        }
    }

    static func libraryURL() throws -> URL {
        let resources = Bundle.main.resourceURL ?? URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        for name in ["Water.metallib", "Water.metal"] {
            let url = resources.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw NativeFailure("Water Metal shaders are missing from the application bundle. Run native/build.sh.")
    }
}

func writeBenchmark(path: String, measurements: [RenderMeasurement], device: String,
                    extra: [String: Any] = [:]) throws {
    func summary(_ samples: [Double]) -> [String: Double] {
        guard !samples.isEmpty else { return [:] }
        let sorted = samples.sorted()
        return ["mean": samples.reduce(0, +) / Double(samples.count),
                "p50": sorted[sorted.count / 2], "p95": sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]]
    }
    var report: [String: Any] = ["device": device, "frames": measurements.count,
        "cpu_submit_ms": summary(measurements.map(\.cpuMilliseconds)),
        "gpu_ms": summary(measurements.map(\.gpuMilliseconds)),
        "simulation_ms": summary(measurements.map(\.simulationMilliseconds)),
        "render_width": measurements.last?.width ?? 0, "render_height": measurements.last?.height ?? 0,
        "measurements": measurements.map { ["cpu_submit_ms": $0.cpuMilliseconds, "gpu_ms": $0.gpuMilliseconds,
                                             "simulation_ms": $0.simulationMilliseconds] }]
    for (key, value) in extra { report[key] = value }
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    print("| Device | Frames | Simulation mean ms | CPU submit mean ms | GPU mean ms | GPU p95 ms |")
    print("|---|---:|---:|---:|---:|---:|")
    print(String(format: "| %@ | %d | %.3f | %.3f | %.3f | %.3f |", device, measurements.count,
        summary(measurements.map(\.simulationMilliseconds))["mean"] ?? 0,
        summary(measurements.map(\.cpuMilliseconds))["mean"] ?? 0,
        summary(measurements.map(\.gpuMilliseconds))["mean"] ?? 0,
        summary(measurements.map(\.gpuMilliseconds))["p95"] ?? 0))
    print("Saved benchmark: \(url.path)")
}

func savePNG(texture: MTLTexture, path: String) throws {
    let width = texture.width, height = texture.height
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    bytes.withUnsafeMutableBytes { storage in
        texture.getBytes(storage.baseAddress!, bytesPerRow: width * 4,
            from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
    }
    let data = Data(bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
        throw NativeFailure("Cannot construct a color-managed capture image.")
    }
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let output = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NativeFailure("Cannot write capture: \(path)")
    }
    CGImageDestinationAddImage(output, image, nil)
    guard CGImageDestinationFinalize(output) else { throw NativeFailure("Failed to finish capture: \(path)") }
    print("Saved Metal capture: \(url.path)")
}

@main struct NativeMain {
    static func main() {
        do {
            let options = try LaunchOptions(arguments: Array(CommandLine.arguments.dropFirst()))
            if options.capture != nil || options.captureSeries != nil || (options.benchmark != nil && options.smoke == nil) {
                let core = try CoreBridge(seed: options.seed)
                try options.apply(to: core)
                let renderer = try MetalRenderer(core: core, libraryURL: try LaunchOptions.libraryURL())
                renderer.quality = options.quality; renderer.recordMeasurements = true
                renderer.cloudStudy = options.dropDemo || options.cloudStudy
                renderer.showFallingDrops = options.showFallingDrops
                renderer.subjectiveIntensity = options.subjectiveIntensity
                let output = try renderer.outputTexture(width: options.width, height: options.height)
                let began = CACurrentMediaTime()
                var seriesFrames: [[String: Any]] = []
                for frameIndex in 0..<options.frames {
                    try autoreleasepool {
                        try core.stepOffline(delta: 1.0 / 60.0)
                        try renderer.drawOffline(frame: core.current(), destination: output, delta: 1.0 / 60.0)
                        if let directory = options.captureSeries,
                           frameIndex % options.captureEvery == 0 || frameIndex == options.frames - 1 {
                            let file = URL(fileURLWithPath: directory).appendingPathComponent(String(format: "frame-%05d.png", frameIndex)).path
                            try savePNG(texture: output, path: file)
                            let state = core.current().state
                            seriesFrames.append(["file": URL(fileURLWithPath: file).lastPathComponent,
                                "frame_index": frameIndex, "physical_seconds": state.physical_time,
                                "scene_seconds": state.uniforms.time.x, "world_origin": [state.world_origin_x, state.world_origin_y],
                                "airborne_drops": core.current().drops.filter { $0.identity.z == 0 }.count,
                                "retained_impacts": core.current().drops.filter { $0.identity.z == 1 }.count])
                        }
                    }
                }
                if let directory = options.captureSeries {
                    let metadata: [String: Any] = ["seed": options.seed, "drop_demo": options.dropDemo, "cloud_study": renderer.cloudStudy,
                        "show_falling_drops": renderer.showFallingDrops,
                        "subjective_intensity": renderer.subjectiveIntensity,
                        "wall_step_seconds": 1.0 / 60.0, "width": options.width, "height": options.height,
                        "frames": seriesFrames]
                    let path = URL(fileURLWithPath: directory).appendingPathComponent("frames.json")
                    try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]).write(to: path)
                }
                if let path = options.capture { try savePNG(texture: output, path: path) }
                if let path = options.benchmark {
                    try writeBenchmark(path: path, measurements: Array(renderer.measurements().dropFirst(options.warmup)), device: renderer.device.name,
                        extra: ["mode": "offline", "wall_seconds": CACurrentMediaTime() - began,
                            "quality": options.quality.name, "output_width": options.width, "output_height": options.height,
                            "warmup_frames": options.warmup, "seed": options.seed])
                }
            } else {
                let app = NSApplication.shared
                app.setActivationPolicy(.regular)
                let delegate = AppController(options: options)
                app.delegate = delegate
                withExtendedLifetime(delegate) { app.run() }
            }
        } catch {
            FileHandle.standardError.write(Data(("Acid Rain Metal: \(error)\n").utf8)); exit(1)
        }
    }
}
