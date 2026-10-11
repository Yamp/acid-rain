import Foundation
import QuartzCore
import simd

struct NativeFailure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

struct SimulationFrame {
    var state: ARSnapshot
    let water: [Float]
    let dye: [Float]
    let flow: [Float]
    let drops: [ARDrop]
    let bed: [Float]
    let sky: [Float]
    let light: [Float]
    let publishedAt: Double
    let simulationSampleAt: Double
    let simulationMilliseconds: Double
}

/// The Rust core lives on one serial queue. The display consumes immutable latest frames;
/// it never queues simulation work behind the display or waits for a slow physics step.
final class CoreBridge {
    private let core: OpaquePointer
    private let queue = DispatchQueue(label: "AcidRain.physics", qos: .userInitiated)
    private let lock = NSLock()
    private var latest: SimulationFrame?
    private var timer: DispatchSourceTimer?
    private var cachedBed: [Float] = []
    private var cachedBedRevision: UInt64 = .max
    private var lastTick: Double = CACurrentMediaTime()
    private var simulationSampleAt: Double = CACurrentMediaTime()
    private var held = false
    private var failure: String?
    var onFailure: ((String) -> Void)?

    init(seed: UInt64) throws {
        guard ar_core_abi_version() == 2, let created = ar_core_create(seed) else {
            throw NativeFailure("The water simulation could not initialize (ABI or core allocation).")
        }
        core = created
        simulationSampleAt = CACurrentMediaTime()
        try publish(milliseconds: 0)
    }

    deinit {
        timer?.cancel()
        // The timer and queued commands capture self weakly; no work remains when deinit runs.
        ar_core_destroy(core)
    }

    func current() -> SimulationFrame {
        lock.lock(); defer { lock.unlock() }
        return latest!
    }

    func currentFailure() -> String? {
        lock.lock(); defer { lock.unlock() }
        return failure
    }

    func setHeld(_ held: Bool) {
        queue.async { [weak self] in self?.held = held; self?.lastTick = CACurrentMediaTime() }
    }

    func command(_ code: UInt32, value: Float = 0) {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                guard ar_core_command(self.core, code, value) == 1 else {
                    throw NativeFailure("The water core rejected command \(code).")
                }
                try self.publish(milliseconds: 0)
            } catch { self.fail(error) }
        }
    }

    func drop(world: SIMD2<Double>, size: Float) {
        queue.async { [weak self] in
            guard let self else { return }
            // A click can leave the resident window before a queued command executes.
            // A full journal or stale click is harmless and should not terminate rendering.
            guard ar_core_drop(self.core, world.x, world.y, size) == 1 else { return }
            do { try self.publish(milliseconds: 0) } catch { self.fail(error) }
        }
    }

    func dropOffline(world: SIMD2<Double>, size: Float) throws {
        precondition(timer == nil)
        guard ar_core_drop(core, world.x, world.y, size) == 1 else { throw NativeFailure("Cannot create the demonstration drop.") }
        try publish(milliseconds: 0)
    }

    func adjust(_ code: UInt32, by delta: Float) {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                var state = ARSnapshot()
                guard ar_core_snapshot(self.core, &state) == 1 else { throw NativeFailure("Cannot read adjustment state.") }
                let base: Float
                switch code {
                case 2: base = state.speed
                case 3: base = state.rain
                case 4: base = state.drop_size
                case 5: base = state.bleach
                case 6: base = state.uniforms.time.z
                case 7: base = state.uniforms.time.w
                default: throw NativeFailure("Invalid relative control \(code).")
                }
                guard ar_core_command(self.core, code, base + delta) == 1 else { throw NativeFailure("Cannot apply control adjustment.") }
                try self.publish(milliseconds: 0)
            } catch { self.fail(error) }
        }
    }

    func start() {
        guard timer == nil else { return }
        lastTick = CACurrentMediaTime()
        let source = DispatchSource.makeTimerSource(queue: queue)
        source.schedule(deadline: .now(), repeating: 1.0 / 60.0, leeway: .milliseconds(1))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let now = CACurrentMediaTime()
            let dt = min(0.1, max(0, now - self.lastTick))
            self.lastTick = now
            guard !self.held else { return }
            do { try self.step(delta: dt) } catch { self.fail(error) }
        }
        timer = source
        source.resume()
    }

    /// Offline mode only. The timer must not have been started.
    func stepOffline(delta: Double) throws {
        precondition(timer == nil)
        try step(delta: delta)
    }

    func commandOffline(_ code: UInt32, value: Float = 0) throws {
        precondition(timer == nil)
        guard ar_core_command(core, code, value) == 1 else { throw NativeFailure("Invalid offline control.") }
        try publish(milliseconds: 0)
    }

    private func step(delta: Double) throws {
        let began = CACurrentMediaTime()
        guard ar_core_step(core, delta) == 1 else { throw NativeFailure("The water simulation failed. The Rust panic was contained; rendering has stopped.") }
        try publish(milliseconds: (CACurrentMediaTime() - began) * 1000, sampleAt: began)
    }

    private func publish(milliseconds: Double, sampleAt: Double? = nil) throws {
        var state = ARSnapshot()
        guard ar_core_snapshot(core, &state) == 1,
              let water = state.water, let dye = state.dye, let bed = state.bed,
              let sky = state.sky, let light = state.bed_light, let flow = state.flow,
              state.drop_count <= 512, state.drop_count == 0 || state.drops != nil else {
            throw NativeFailure("The water simulation returned an invalid snapshot.")
        }
        // Physics integrates the wall interval sampled at step start. Publication
        // comes later, after simulation and copies; commands preserve this anchor.
        if let sampleAt, latest?.state.frame != state.frame { simulationSampleAt = sampleAt }
        if cachedBedRevision != state.bed_revision {
            cachedBed = Array(UnsafeBufferPointer(start: bed, count: Int(state.bed_width * state.bed_height) * 4))
            cachedBedRevision = state.bed_revision
        }
        let frame = SimulationFrame(
            state: state,
            water: Array(UnsafeBufferPointer(start: water, count: Int(state.water_width * state.water_height) * 4)),
            dye: Array(UnsafeBufferPointer(start: dye, count: Int(state.water_width * state.water_height) * 4)),
            flow: Array(UnsafeBufferPointer(start: flow, count: Int(state.water_width * state.water_height) * 4)),
            drops: state.drop_count == 0 ? [] : Array(UnsafeBufferPointer(start: state.drops!, count: Int(state.drop_count))),
            bed: cachedBed,
            sky: Array(UnsafeBufferPointer(start: sky, count: Int(state.sky_width * state.sky_height) * 4)),
            light: Array(UnsafeBufferPointer(start: light, count: Int(state.light_width * state.light_height) * 4)),
            publishedAt: CACurrentMediaTime(), simulationSampleAt: simulationSampleAt,
            simulationMilliseconds: milliseconds)
        lock.lock(); latest = frame; lock.unlock()
    }

    private func fail(_ error: Error) {
        timer?.cancel(); timer = nil
        lock.lock(); failure = String(describing: error); lock.unlock()
        DispatchQueue.main.async { [weak self] in self?.onFailure?(String(describing: error)) }
    }
}
