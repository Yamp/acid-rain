import Foundation

enum RenderQuality: Int, CaseIterable {
    case auto, high, balanced, low, native
    var name: String { ["Auto", "High", "Balanced", "Low", "Native"][rawValue] }
    var next: RenderQuality { RenderQuality(rawValue: (rawValue + 1) % 5)! }
    var shaderLevel: Float { self == .low ? 0 : self == .balanced ? 1 : 2 }
}

struct RenderExtent: Equatable { let width: Int; let height: Int }

struct RenderPolicy {
    private(set) var automaticPixelBudget = 1_200_000
    private var slowFrames = 0
    private var fastFrames = 0

    static func extent(width: Int, height: Int, quality: RenderQuality,
                       automaticPixelBudget: Int = 1_200_000) -> RenderExtent {
        let width = max(1, width), height = max(1, height)
        let budget: Int
        switch quality {
        case .native: budget = width * height
        case .high: budget = 1_600_000
        case .balanced: budget = 900_000
        case .low: budget = 450_000
        case .auto: budget = automaticPixelBudget
        }
        let scale = min(1, sqrt(Double(budget) / Double(width * height)))
        return RenderExtent(width: max(1, Int(Double(width) * scale)),
                            height: max(1, Int(Double(height) * scale)))
    }

    mutating func observe(gpuMilliseconds: Double) {
        guard gpuMilliseconds.isFinite && gpuMilliseconds > 0 else { return }
        if gpuMilliseconds > 20 {
            slowFrames += 1; fastFrames = 0
        } else if gpuMilliseconds < 11 {
            fastFrames += 1; slowFrames = 0
        } else {
            slowFrames = 0; fastFrames = 0
        }
        if slowFrames >= 30 {
            automaticPixelBudget = max(350_000, Int(Double(automaticPixelBudget) * 0.80))
            slowFrames = 0
        } else if fastFrames >= 180 {
            automaticPixelBudget = min(1_600_000, Int(Double(automaticPixelBudget) * 1.10))
            fastFrames = 0
        }
    }
}

struct CloudStep {
    let delta: Double
    let shift: SIMD2<Int32>
    let clear: Bool
}

/// The volume follows the physics clock, never the number of display callbacks.
/// A severe display stall has a bounded 16-substep recovery rather than a GPU spiral.
struct CloudClock {
    private var time: Double?
    private var origin = SIMD2<Double>.zero
    private var revision: UInt64?

    mutating func consume(time nextTime: Double, origin nextOrigin: SIMD2<Double>, revision nextRevision: UInt64) -> CloudStep {
        let clear = revision != nextRevision
        let elapsed = clear ? 0 : max(0, nextTime - (time ?? nextTime))
        let displacement = nextOrigin - origin
        let shift = clear ? SIMD2<Int32>.zero : SIMD2<Int32>(
            Int32(max(-96, min(96, (displacement.x * 48).rounded()))),
            Int32(max(-64, min(64, (displacement.y * 32).rounded()))))
        time = nextTime; origin = nextOrigin; revision = nextRevision
        return CloudStep(delta: min(0.128, elapsed), shift: shift, clear: clear)
    }
}

/// Injection happens at the first CFL boundary at or after contact. A source can
/// be at most one substep late, but is never advected or bleached before impact.
struct CloudTransportPlan {
    let substepCount: Int
    let substepDelta: Double
    let sourceOrder: [Int]
    let injections: [Range<Int>]

    init(delta: Double, endTime: Double, impactTimes: [Double]) {
        precondition(delta >= 0 && delta <= 0.128)
        substepCount = delta > 0 ? max(1, Int(ceil(delta / 0.008))) : 0
        substepDelta = substepCount > 0 ? delta / Double(substepCount) : 0
        sourceOrder = impactTimes.indices.sorted {
            impactTimes[$0] == impactTimes[$1] ? $0 < $1 : impactTimes[$0] < impactTimes[$1]
        }
        var counts = [Int](repeating: 0, count: substepCount + 1)
        let start = endTime - delta
        for source in sourceOrder {
            let bucket = substepCount == 0 ? 0 : max(0, min(substepCount,
                Int(ceil((impactTimes[source] - start) / substepDelta))))
            counts[bucket] += 1
        }
        var cursor = 0
        injections = counts.map { count in
            defer { cursor += count }
            return cursor..<(cursor + count)
        }
    }
}
