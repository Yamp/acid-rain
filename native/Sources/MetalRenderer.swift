import AppKit
import MetalKit
import QuartzCore
import simd

struct RenderMeasurement {
    let cpuMilliseconds: Double
    let gpuMilliseconds: Double
    let simulationMilliseconds: Double
    let width: Int
    let height: Int
}

private struct CloudUniforms {
    var step: SIMD4<Float>
    var rebase = SIMD4<Float>.zero
    var domain = SIMD4<Float>(2, 2, -0.8, 0.9)
    var options = SIMD4<Float>(0, 1, 0.00001, 0)
}

private struct CausticsUniforms {
    var light: SIMD4<Float>
    var domain = SIMD4<Float>(2, -0.5, 3, 1)
    var grid = SIMD4<UInt32>(128, 256, 256, 0)
}

private final class FrameResources {
    let water: MTLTexture
    let dye: MTLTexture
    let flow: MTLTexture
    let airborne: MTLBuffer
    let sources: MTLBuffer
    let wakes: MTLBuffer
    let normalization: MTLBuffer
    let bed: MTLTexture
    let sky: MTLTexture
    let light: MTLTexture
    var bedRevision: UInt64 = .max

    init(device: MTLDevice, frame: SimulationFrame) throws {
        func texture(_ width: UInt32, _ height: UInt32, _ label: String) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                width: Int(width), height: Int(height), mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let result = device.makeTexture(descriptor: descriptor) else {
                throw NativeFailure("Cannot allocate \(label) simulation texture.")
            }
            result.label = label
            return result
        }
        let s = frame.state
        water = try texture(s.water_width, s.water_height, "Water surface")
        dye = try texture(s.water_width, s.water_height, "Pigment and depth")
        flow = try texture(s.water_width, s.water_height, "Volume driving velocity")
        func buffer(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let result = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw NativeFailure("Cannot allocate \(label).")
            }
            result.label = label
            return result
        }
        airborne = try buffer(512 * MemoryLayout<ARDrop>.stride, "Airborne drop instances")
        sources = try buffer(512 * MemoryLayout<ARDrop>.stride, "New pigment sources")
        wakes = try buffer(24 * MemoryLayout<ARDrop>.stride, "Recent impact vortex rings")
        normalization = try buffer(512 * 16, "Discrete source mass normalization")
        bed = try texture(s.bed_width, s.bed_height, "Streamed bottom")
        sky = try texture(s.sky_width, s.sky_height, "Spectral environment")
        light = try texture(s.light_width, s.light_height, "Bottom incident light")
    }

    func upload(_ frame: SimulationFrame) {
        func replace(_ texture: MTLTexture, _ values: [Float]) {
            values.withUnsafeBytes { bytes in
                texture.replace(region: MTLRegionMake2D(0, 0, texture.width, texture.height),
                    mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: texture.width * 16)
            }
        }
        replace(water, frame.water)
        replace(dye, frame.dye)
        replace(flow, frame.flow)
        replace(sky, frame.sky)
        replace(light, frame.light)
        if bedRevision != frame.state.bed_revision {
            replace(bed, frame.bed)
            bedRevision = frame.state.bed_revision
        }
    }
}

final class MetalRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let core: CoreBridge
    var quality: RenderQuality = .high
    var interpolation = true
    var cloudStudy = false
    var showFallingDrops = false
    var subjectiveIntensity: Float = 1
    var helpVisible = false
    var onFailure: ((String) -> Void)?
    var onFrame: ((SimulationFrame, RenderMeasurement?) -> Void)?
    private let queue: MTLCommandQueue
    private let waterPipeline: MTLRenderPipelineState
    private let postPipeline: MTLRenderPipelineState
    private let exposurePipeline: MTLComputePipelineState
    private let retinaHistoryPipeline: MTLComputePipelineState
    private let dropletPipeline: MTLRenderPipelineState
    private let causticsMapPipeline: MTLComputePipelineState
    private let causticsRenderPipeline: MTLRenderPipelineState
    private var causticsRays: MTLTexture!
    private var causticsLight: MTLTexture!
    private let cloudNormalizePipeline: MTLComputePipelineState
    private let cloudPreparePipeline: MTLComputePipelineState
    private let cloudVelocityPipeline: MTLComputePipelineState
    private let cloudAdvectPipeline: MTLComputePipelineState
    private var clouds: [MTLTexture] = []
    private var cloudVelocity: MTLTexture!
    private var cloudIndex = 0
    private var cloudClock = CloudClock()
    private var consumedImpacts = Set<UInt64>()
    private var pickingFrame: (SimulationFrame, ARUniforms)?
    private var slots: [FrameResources] = []
    private let inFlight = DispatchSemaphore(value: 3)
    private var nextSlot = 0
    private var hdr: [MTLTexture] = []
    private var exposures: [MTLTexture] = []
    private var extent = RenderExtent(width: 0, height: 0)
    private var historyIndex = 0
    private var historyValid = false
    private var retinaHistories: [[MTLTexture]] = []
    private var retinaIndex = 0
    private var retinaPolicy = RetinaHistoryPolicy()
    private var presentationClock = PresentationClock()
    private var cameraPresentation = CameraPresentation()
    private var offlinePresentationTime = 0.0
    private var previousSceneMode: SIMD2<UInt32>?
    private var lastFrameTime = CACurrentMediaTime()
    private var occluded = false
    private var policy = RenderPolicy()
    private let statisticsLock = NSLock()
    private var lastMeasurement: RenderMeasurement?
    private var records: [RenderMeasurement] = []
    private var presentedTimes: [Double] = []
    private var failed = false
    var recordMeasurements = false

    init(core: CoreBridge, libraryURL: URL) throws {
        guard let gpu = MTLCreateSystemDefaultDevice(), let commandQueue = gpu.makeCommandQueue() else {
            throw NativeFailure("This Mac does not provide a Metal device and command queue.")
        }
        device = gpu; queue = commandQueue; self.core = core
        let library: MTLLibrary
        if libraryURL.pathExtension == "metal" {
            let options = MTLCompileOptions()
            if #available(macOS 15.0, *) { options.mathMode = .fast } else { options.fastMathEnabled = true }
            var source = try String(contentsOf: libraryURL, encoding: .utf8)
            // Bundled Water.metal already contains all modules. A direct source URL
            // used by diagnostics gets the same ordered module composition here.
            var modules: [String] = []
            for name in ["Cloud.metal", "Caustics.metal", "BedLife.metal"] {
                let url = libraryURL.deletingLastPathComponent().appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: url.path) {
                    modules.append(try String(contentsOf: url, encoding: .utf8))
                }
            }
            modules.append(source)
            source = modules.joined(separator: "\n")
            library = try gpu.makeLibrary(source: source, options: options)
        } else { library = try gpu.makeLibrary(URL: libraryURL) }
        guard let vertex = library.makeFunction(name: "waterVertex"),
              let fragment = library.makeFunction(name: "waterFragment"),
              let post = library.makeFunction(name: "postFragment"),
              let exposure = library.makeFunction(name: "adaptExposure") else {
            throw NativeFailure("Water.metallib is missing a required shader entry point.")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        descriptor.label = "Linear HDR water optics"
        waterPipeline = try gpu.makeRenderPipelineState(descriptor: descriptor)
        descriptor.fragmentFunction = post
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        descriptor.label = "Retinal optics and sRGB output"
        postPipeline = try gpu.makeRenderPipelineState(descriptor: descriptor)
        exposurePipeline = try gpu.makeComputePipelineState(function: exposure)
        func compute(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw NativeFailure("Missing Metal kernel \(name).") }
            return try gpu.makeComputePipelineState(function: function)
        }
        retinaHistoryPipeline = try compute("retinaHistoryUpdate")
        cloudNormalizePipeline = try compute("cloudNormalize")
        cloudPreparePipeline = try compute("cloudPrepare")
        cloudVelocityPipeline = try compute("cloudVelocity")
        cloudAdvectPipeline = try compute("cloudAdvect")
        causticsMapPipeline = try compute("causticsMap")
        guard let causticsVertex = library.makeFunction(name: "causticsVertex"),
              let causticsFragment = library.makeFunction(name: "causticsFragment") else {
            throw NativeFailure("Missing bottom caustics Metal pipeline.")
        }
        let causticsDescriptor = MTLRenderPipelineDescriptor()
        causticsDescriptor.vertexFunction = causticsVertex
        causticsDescriptor.fragmentFunction = causticsFragment
        causticsDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        causticsDescriptor.colorAttachments[0].isBlendingEnabled = true
        causticsDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        causticsDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one
        causticsDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        causticsDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
        causticsDescriptor.label = "Spectral caustic irradiance"
        causticsRenderPipeline = try gpu.makeRenderPipelineState(descriptor: causticsDescriptor)
        guard let dropVertex = library.makeFunction(name: "dropletVertex"),
              let dropFragment = library.makeFunction(name: "dropletFragment") else {
            throw NativeFailure("Missing falling-drop Metal pipeline.")
        }
        descriptor.vertexFunction = dropVertex
        descriptor.fragmentFunction = dropFragment
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        descriptor.label = "Visible airborne pigment droplets"
        dropletPipeline = try gpu.makeRenderPipelineState(descriptor: descriptor)
        super.init()
        precondition(MemoryLayout<ARUniforms>.size == 208 && MemoryLayout<ARDrop>.stride == 64
            && MemoryLayout<CloudUniforms>.stride == 64 && MemoryLayout<CausticsUniforms>.stride == 48,
            "Metal / Rust uniform ABI mismatch")
        let rayDescriptor = MTLTextureDescriptor()
        rayDescriptor.textureType = .type2DArray; rayDescriptor.arrayLength = 3
        rayDescriptor.pixelFormat = .rgba32Float; rayDescriptor.width = 129; rayDescriptor.height = 129
        rayDescriptor.storageMode = .private; rayDescriptor.usage = [.shaderRead, .shaderWrite]
        guard let rays = gpu.makeTexture(descriptor: rayDescriptor) else { throw NativeFailure("Cannot allocate spectral caustic rays.") }
        rays.label = "Spectral refraction ray map"
        causticsRays = rays
        let receiverDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
            width: 256, height: 256, mipmapped: false)
        receiverDescriptor.storageMode = .shared; receiverDescriptor.usage = [.renderTarget, .shaderRead]
        guard let receiver = gpu.makeTexture(descriptor: receiverDescriptor) else { throw NativeFailure("Cannot allocate bottom caustic irradiance.") }
        receiver.label = "Bottom caustic irradiance"
        let neutral = [Float16](repeating: Float16(0.98), count: 256 * 256 * 4)
        neutral.withUnsafeBytes { values in
            receiver.replace(region: MTLRegionMake2D(0, 0, 256, 256), mipmapLevel: 0,
                withBytes: values.baseAddress!, bytesPerRow: 256 * 8)
        }
        causticsLight = receiver
        for label in ["Pigment volume A", "Pigment volume B", "Cloud velocity"] {
            let d = MTLTextureDescriptor()
            d.textureType = .type3D; d.pixelFormat = .rgba32Float
            d.width = 96; d.height = 64; d.depth = 32
            d.storageMode = .private; d.usage = [.shaderRead, .shaderWrite]
            guard let texture = gpu.makeTexture(descriptor: d) else { throw NativeFailure("Cannot allocate \(label).") }
            texture.label = label
            if clouds.count < 2 { clouds.append(texture) } else { cloudVelocity = texture }
        }
        for _ in 0..<3 { slots.append(try FrameResources(device: gpu, frame: core.current())) }
        for index in 0..<2 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 1, height: 1, mipmapped: false)
            d.storageMode = .shared; d.usage = [.shaderRead, .shaderWrite]
            guard let texture = gpu.makeTexture(descriptor: d) else { throw NativeFailure("Cannot allocate pupil exposure state.") }
            texture.label = "Pupil exposure \(index)"
            var value: Float = 1
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &value, bytesPerRow: 4)
            exposures.append(texture)
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { resetHistory() }

    func resetHistory(preserveRetina: Bool = false) {
        historyValid = false
        if !preserveRetina { retinaPolicy.invalidate() }
    }

    func setOccluded(_ hidden: Bool) {
        guard occluded != hidden else { return }
        occluded = hidden
        let now = CACurrentMediaTime()
        lastFrameTime = now
        presentationClock.reanchor(at: now)
    }

    func drop(at uv: SIMD2<Float>) {
        guard let (frame, uniforms) = pickingFrame,
              let local = WaterCamera.surfacePoint(at: uv, uniforms: uniforms, water: frame.water,
                  width: Int(frame.state.water_width), height: Int(frame.state.water_height)) else { return }
        core.drop(world: SIMD2<Double>(frame.state.world_origin_x + Double(local.x),
            frame.state.world_origin_y + Double(local.y)), size: frame.state.drop_size)
    }

    func draw(in view: MTKView) {
        guard !failed, !occluded, view.drawableSize.width > 0, view.drawableSize.height > 0,
              inFlight.wait(timeout: .now()) == .success else { return }
        guard let drawable = view.currentDrawable else { inFlight.signal(); return }
        let frame = core.current()
        let now = CACurrentMediaTime()
        let dt = max(0, now - lastFrameTime)
        lastFrameTime = now
        do {
            let buffer = try encode(frame: frame, destination: drawable.texture, delta: dt,
                backingScale: Float(view.window?.backingScaleFactor ?? 1), predictCamera: true, displayTime: now)
            if recordMeasurements {
                drawable.addPresentedHandler { [weak self] shown in
                    guard let self, shown.presentedTime > 0 else { return }
                    self.statisticsLock.lock()
                    self.presentedTimes.append(shown.presentedTime)
                    self.statisticsLock.unlock()
                }
            }
            buffer.present(drawable)
            buffer.addCompletedHandler { [weak self] _ in self?.inFlight.signal() }
            buffer.commit()
            statisticsLock.lock(); let measurement = lastMeasurement; statisticsLock.unlock()
            onFrame?(frame, measurement)
        } catch {
            inFlight.signal(); fail(error)
        }
    }

    /// Real Metal rendering with an explicit wait only for deterministic offline captures.
    func drawOffline(frame: SimulationFrame, destination: MTLTexture, delta: Double) throws {
        offlinePresentationTime += max(0, delta)
        let buffer = try encode(frame: frame, destination: destination, delta: delta,
            backingScale: 1, predictCamera: false, displayTime: offlinePresentationTime)
        let completed = DispatchSemaphore(value: 0)
        buffer.addCompletedHandler { _ in completed.signal() }
        buffer.commit()
        buffer.waitUntilCompleted()
        completed.wait()
        if buffer.status == .error { throw NativeFailure(buffer.error?.localizedDescription ?? "Metal command failed.") }
    }

    func measurements() -> [RenderMeasurement] {
        statisticsLock.lock(); defer { statisticsLock.unlock() }
        return records
    }

    func presentationSummary() -> [String: Any] {
        statisticsLock.lock(); let times = presentedTimes.sorted(); statisticsLock.unlock()
        let intervals = zip(times.dropFirst(), times).map { ($0.0 - $0.1) * 1000 }.filter { $0 > 0 }.sorted()
        var summary: [String: Any] = ["presented_frames": times.count]
        if let first = times.first, let last = times.last, last > first, !intervals.isEmpty {
            summary["presented_fps"] = Double(times.count - 1) / (last - first)
            summary["presentation_interval_p50_ms"] = intervals[intervals.count / 2]
            summary["presentation_interval_p95_ms"] = intervals[min(intervals.count - 1, Int(Double(intervals.count) * 0.95))]
            summary["presentation_interval_max_ms"] = intervals.last!
        }
        return summary
    }

    func outputTexture(width: Int, height: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: width, height: height, mipmapped: false)
        d.storageMode = .shared; d.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: d) else { throw NativeFailure("Cannot allocate capture texture.") }
        return texture
    }

    private func ensureHDR(width: Int, height: Int) throws {
        let wanted = RenderPolicy.extent(width: width, height: height, quality: quality,
                                        automaticPixelBudget: policy.automaticPixelBudget)
        guard wanted != extent else { return }
        var next: [MTLTexture] = []
        for index in 0..<2 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                width: wanted.width, height: wanted.height, mipmapped: false)
            d.storageMode = .private; d.usage = [.renderTarget, .shaderRead]
            guard let texture = device.makeTexture(descriptor: d) else { throw NativeFailure("Cannot allocate HDR target at \(wanted.width) × \(wanted.height).") }
            texture.label = "HDR history \(index)"
            next.append(texture)
        }
        let reservoirExtent = RetinaHistoryPolicy.extent(width: wanted.width, height: wanted.height)
        var reservoirs: [[MTLTexture]] = []
        for generation in 0..<2 {
            var layers: [MTLTexture] = []
            for layer in 0..<3 {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                    width: reservoirExtent.width, height: reservoirExtent.height, mipmapped: false)
                descriptor.storageMode = .private; descriptor.usage = [.shaderRead, .shaderWrite]
                guard let texture = device.makeTexture(descriptor: descriptor) else {
                    throw NativeFailure("Cannot allocate retinal history.")
                }
                texture.label = "Unwarped retinal history \(layer), generation \(generation)"
                layers.append(texture)
            }
            reservoirs.append(layers)
        }
        hdr = next; extent = wanted
        retinaHistories = reservoirs; retinaIndex = 0
        resetHistory()
    }

    private func encodeClouds(_ frame: SimulationFrame, resources: FrameResources,
                              commands: MTLCommandBuffer) throws -> Int {
        func id(_ drop: ARDrop) -> UInt64 { UInt64(drop.identity.x) | (UInt64(drop.identity.y) << 32) }
        let step = cloudClock.consume(time: frame.state.physical_time,
            origin: SIMD2<Double>(frame.state.world_origin_x, frame.state.world_origin_y),
            revision: frame.state.cloud_revision)
        if step.clear { consumedImpacts.removeAll(keepingCapacity: true) }
        let retained = Set(frame.drops.map(id))
        consumedImpacts.formIntersection(retained)
        let pendingSources = frame.drops.filter { $0.identity.z == 1 && !consumedImpacts.contains(id($0)) }
        let plan = CloudTransportPlan(delta: step.delta, endTime: frame.state.physical_time,
            impactTimes: pendingSources.map { Double($0.dynamics.z) })
        let newSources = plan.sourceOrder.map { pendingSources[$0] }
        for source in newSources { consumedImpacts.insert(id(source)) }
        let airborne = frame.drops.filter { $0.identity.z == 0 }
        let recentWakes = Array(frame.drops.filter {
            let age = frame.state.physical_time - Double($0.dynamics.z)
            return $0.identity.z == 1 && age >= 0 && age <= 2.5
        }.suffix(24))
        func upload(_ drops: [ARDrop], to buffer: MTLBuffer) {
            guard !drops.isEmpty else { return }
            drops.withUnsafeBytes { bytes in buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count) }
        }
        upload(airborne, to: resources.airborne)
        upload(newSources, to: resources.sources)
        upload(recentWakes, to: resources.wakes)
        var uniforms = CloudUniforms(step: SIMD4<Float>(0, Float(frame.state.physical_time), frame.state.bleach, 0))
        uniforms.options.x = Float(frame.state.world_origin_x.truncatingRemainder(dividingBy: 2 * Double.pi))
        uniforms.options.w = Float(frame.state.world_origin_y.truncatingRemainder(dividingBy: 2 * Double.pi))
        let grid = MTLSize(width: 96, height: 64, depth: 32)
        func dispatch(_ pipeline: MTLComputePipelineState, events: MTLBuffer,
                      count: Int, source: MTLTexture, target: MTLTexture, normalize: Bool = false,
                      eventOffset: Int = 0) throws {
            guard let encoder = commands.makeComputeCommandEncoder() else { throw NativeFailure("Cannot encode pigment transport.") }
            uniforms.step.w = Float(count)
            encoder.setComputePipelineState(pipeline)
            encoder.setBytes(&uniforms, length: MemoryLayout<CloudUniforms>.stride, index: 0)
            encoder.setBuffer(events, offset: eventOffset * MemoryLayout<ARDrop>.stride, index: 1)
            encoder.setBuffer(resources.normalization, offset: eventOffset * 16, index: 2)
            encoder.setTexture(source, index: 0)
            encoder.setTexture(target, index: 1)
            encoder.setTexture(cloudVelocity, index: 2)
            encoder.setTexture(resources.flow, index: 3)
            encoder.setTexture(resources.water, index: 4)
            if normalize {
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            } else {
                encoder.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: 8, height: 4, depth: 4))
            }
            encoder.endEncoding()
        }
        if step.clear || step.shift != .zero {
            uniforms.rebase = SIMD4<Float>(Float(step.shift.x), Float(step.shift.y), step.clear ? 1 : 0, 0)
            try dispatch(cloudPreparePipeline, events: resources.sources, count: 0,
                source: clouds[cloudIndex], target: clouds[1 - cloudIndex])
            cloudIndex = 1 - cloudIndex
            uniforms.rebase = .zero
        }
        if !newSources.isEmpty {
            try dispatch(cloudNormalizePipeline, events: resources.sources, count: newSources.count,
                source: clouds[cloudIndex], target: clouds[1 - cloudIndex], normalize: true)
        }
        func inject(_ range: Range<Int>) throws {
            guard !range.isEmpty else { return }
            uniforms.rebase.w = 1
            try dispatch(cloudPreparePipeline, events: resources.sources, count: range.count,
                source: clouds[cloudIndex], target: clouds[1 - cloudIndex], eventOffset: range.lowerBound)
            cloudIndex = 1 - cloudIndex
            uniforms.rebase.w = 0
        }
        try inject(plan.injections[0])
        if plan.substepCount > 0 {
            uniforms.step.y = Float(frame.state.physical_time - step.delta * 0.5)
            try dispatch(cloudVelocityPipeline, events: resources.wakes, count: recentWakes.count,
                source: clouds[cloudIndex], target: clouds[1 - cloudIndex])
            uniforms.step.x = Float(plan.substepDelta)
            for substep in 1...plan.substepCount {
                try dispatch(cloudAdvectPipeline, events: resources.wakes, count: 0,
                    source: clouds[cloudIndex], target: clouds[1 - cloudIndex])
                cloudIndex = 1 - cloudIndex
                try inject(plan.injections[substep])
            }
        }
        return airborne.count
    }

    private func encodeCaustics(resources: FrameResources, uniforms u: ARUniforms,
                                commands: MTLCommandBuffer) throws {
        var parameters = CausticsUniforms(light: SIMD4<Float>(u.light.x, u.light.y, u.light.z, 0.02))
        guard let map = commands.makeComputeCommandEncoder() else { throw NativeFailure("Cannot encode refraction ray map.") }
        map.label = "Map spectral refraction to the bed"
        map.setComputePipelineState(causticsMapPipeline)
        map.setBytes(&parameters, length: MemoryLayout<CausticsUniforms>.stride, index: 0)
        map.setTexture(resources.water, index: 0)
        map.setTexture(resources.dye, index: 1)
        map.setTexture(causticsRays, index: 2)
        map.dispatchThreads(MTLSize(width: 129, height: 129, depth: 3),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        map.endEncoding()
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = causticsLight
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        guard let receiver = commands.makeRenderCommandEncoder(descriptor: pass) else {
            throw NativeFailure("Cannot accumulate bottom caustic irradiance.")
        }
        receiver.label = "Conserve spectral flux on the bottom"
        receiver.setRenderPipelineState(causticsRenderPipeline)
        receiver.setCullMode(.none)
        receiver.setVertexBytes(&parameters, length: MemoryLayout<CausticsUniforms>.stride, index: 0)
        receiver.setFragmentBytes(&parameters, length: MemoryLayout<CausticsUniforms>.stride, index: 0)
        receiver.setVertexTexture(causticsRays, index: 0)
        receiver.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 128 * 128 * 6, instanceCount: 3)
        receiver.endEncoding()
    }

    private func encodeRetinalHistory(current: MTLTexture, delta: Double, held: Bool,
                                      commands: MTLCommandBuffer) throws -> SIMD4<Float> {
        let step = retinaPolicy.advance(delta: delta, held: held)
        if step.update {
            let next = 1 - retinaIndex
            for layer in 0..<3 {
                guard let encoder = commands.makeComputeCommandEncoder() else {
                    throw NativeFailure("Cannot encode retinal history.")
                }
                var parameters = SIMD4<Float>(step.alphas[layer], step.seed ? 0 : 1, 0, 0)
                let result = retinaHistories[next][layer]
                encoder.label = "Time-normalized retinal reservoir \(layer)"
                encoder.setComputePipelineState(retinaHistoryPipeline)
                encoder.setBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.setTexture(current, index: 0)
                encoder.setTexture(retinaHistories[retinaIndex][layer], index: 1)
                encoder.setTexture(result, index: 2)
                encoder.dispatchThreads(MTLSize(width: result.width, height: result.height, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                encoder.endEncoding()
            }
            retinaIndex = next
        }
        return SIMD4<Float>(step.time, max(0, min(2, subjectiveIntensity)), 1, 0)
    }

    private func encode(frame: SimulationFrame, destination: MTLTexture, delta: Double,
                        backingScale: Float, predictCamera: Bool, displayTime: Double) throws -> MTLCommandBuffer {
        let cpuStart = CACurrentMediaTime()
        try ensureHDR(width: destination.width, height: destination.height)
        guard let commands = queue.makeCommandBuffer() else { throw NativeFailure("Cannot create Metal command buffer.") }
        commands.label = "Water → pupil → retina"
        let resources = slots[nextSlot]
        nextSlot = (nextSlot + 1) % slots.count
        resources.upload(frame)
        let airborneCount = try encodeClouds(frame, resources: resources, commands: commands)
        let current = historyIndex, previous = 1 - historyIndex
        var uniforms = frame.state.uniforms
        let held = frame.state.paused != 0 || helpVisible
        let sceneMode = SIMD2<UInt32>(frame.state.scene, frame.state.mode)
        if previousSceneMode != sceneMode { resetHistory(); previousSceneMode = sceneMode }
        let presentedTime = presentationClock.sample(sceneTime: Double(uniforms.time.x),
            sampledAt: predictCamera ? frame.simulationSampleAt : displayTime,
            speed: Double(frame.state.speed), now: displayTime, held: held)
        uniforms.time.x = Float(presentedTime)
        uniforms.time.y = held ? 0 : Float(min(0.1, delta))
        uniforms.post.y = quality.shaderLevel
        uniforms.post.z = historyValid && interpolation && frame.state.paused == 0 && !helpVisible ? 0.13 : 0
        uniforms.post.w = backingScale
        uniforms.illumination.w = cloudStudy ? 1 : 0
        if cloudStudy { uniforms.optics.x *= 0.65 }
        cameraPresentation.fill(&uniforms,
            origin: SIMD2<Double>(frame.state.world_origin_x, frame.state.world_origin_y),
            snapshotSceneTime: frame.state.uniforms.time.x, presentationSceneTime: Float(presentedTime),
            delta: delta > 0.25 ? 0 : delta, held: held, aspect: Float(destination.width) / Float(destination.height))
        pickingFrame = (frame, uniforms)
        try encodeCaustics(resources: resources, uniforms: uniforms, commands: commands)
        uniforms.viewport = SIMD4<Float>(Float(extent.width), Float(extent.height), 1 / Float(extent.width), 1 / Float(extent.height))
        let waterPass = MTLRenderPassDescriptor()
        waterPass.colorAttachments[0].texture = hdr[current]
        waterPass.colorAttachments[0].loadAction = .clear
        waterPass.colorAttachments[0].storeAction = .store
        waterPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        guard let water = commands.makeRenderCommandEncoder(descriptor: waterPass) else { throw NativeFailure("Cannot encode the water render pass.") }
        water.setRenderPipelineState(waterPipeline)
        water.setFragmentBytes(&uniforms, length: MemoryLayout<ARUniforms>.size, index: 0)
        var bedWorld = SIMD4<Float>(Float(frame.state.world_origin_x), Float(frame.state.world_origin_y),
            Float(frame.state.physical_time), cloudStudy ? 0 : 1)
        water.setFragmentBytes(&bedWorld, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
        for (index, texture) in [resources.water, resources.dye, resources.bed, resources.sky, resources.light].enumerated() {
            water.setFragmentTexture(texture, index: index)
        }
        water.setFragmentTexture(clouds[cloudIndex], index: 5)
        water.setFragmentTexture(causticsLight, index: 6)
        water.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        if showFallingDrops && airborneCount > 0 {
            water.setRenderPipelineState(dropletPipeline)
            water.setVertexBytes(&uniforms, length: MemoryLayout<ARUniforms>.size, index: 0)
            water.setVertexBuffer(resources.airborne, offset: 0, index: 1)
            water.setFragmentBuffer(resources.airborne, offset: 0, index: 1)
            water.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: airborneCount)
        }
        water.endEncoding()
        var retinaState = try encodeRetinalHistory(current: hdr[current], delta: delta, held: held, commands: commands)
        guard let exposure = commands.makeComputeCommandEncoder() else { throw NativeFailure("Cannot encode pupil adaptation.") }
        exposure.setComputePipelineState(exposurePipeline)
        exposure.setBytes(&uniforms, length: MemoryLayout<ARUniforms>.size, index: 0)
        exposure.setTexture(hdr[current], index: 0)
        exposure.setTexture(exposures[previous], index: 1)
        exposure.setTexture(exposures[current], index: 2)
        exposure.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        exposure.endEncoding()
        uniforms.viewport = SIMD4<Float>(Float(destination.width), Float(destination.height), 1 / Float(destination.width), 1 / Float(destination.height))
        let postPass = MTLRenderPassDescriptor()
        postPass.colorAttachments[0].texture = destination
        postPass.colorAttachments[0].loadAction = .dontCare
        postPass.colorAttachments[0].storeAction = .store
        guard let post = commands.makeRenderCommandEncoder(descriptor: postPass) else { throw NativeFailure("Cannot encode the retinal render pass.") }
        post.setRenderPipelineState(postPipeline)
        post.setFragmentBytes(&uniforms, length: MemoryLayout<ARUniforms>.size, index: 0)
        post.setFragmentBytes(&retinaState, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
        post.setFragmentTexture(hdr[current], index: 0)
        post.setFragmentTexture(hdr[previous], index: 1)
        post.setFragmentTexture(exposures[current], index: 2)
        for layer in 0..<3 { post.setFragmentTexture(retinaHistories[retinaIndex][layer], index: layer + 3) }
        post.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        post.endEncoding()
        historyIndex = previous; historyValid = true
        let cpu = (CACurrentMediaTime() - cpuStart) * 1000
        let renderExtent = extent
        commands.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            if buffer.status == .error {
                DispatchQueue.main.async { self.fail(NativeFailure(buffer.error?.localizedDescription ?? "Metal GPU execution failed.")) }
                return
            }
            let measurement = RenderMeasurement(cpuMilliseconds: cpu,
                gpuMilliseconds: max(0, buffer.gpuEndTime - buffer.gpuStartTime) * 1000,
                simulationMilliseconds: frame.simulationMilliseconds,
                width: renderExtent.width, height: renderExtent.height)
            self.statisticsLock.lock()
            self.lastMeasurement = measurement
            if self.recordMeasurements { self.records.append(measurement) }
            self.statisticsLock.unlock()
            DispatchQueue.main.async { if self.quality == .auto { self.policy.observe(gpuMilliseconds: measurement.gpuMilliseconds) } }
        }
        return commands
    }

    private func fail(_ error: Error) {
        guard !failed else { return }
        failed = true
        onFailure?(String(describing: error))
    }
}
