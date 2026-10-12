import Foundation

/// Tracks the simulation's sample time, not the later publication timestamp.
/// Small phase errors converge smoothly; missing snapshots allow only a bounded lead.
struct PresentationClock {
    private(set) var value = 0.0
    private var lastWall: Double?
    private var wasHeld = false

    mutating func reanchor(at wallTime: Double) {
        guard lastWall != nil else { return }
        lastWall = wallTime
        wasHeld = true
    }

    mutating func sample(sceneTime: Double, sampledAt: Double, speed: Double,
                         now: Double, held: Bool) -> Double {
        let rawAge = max(0, now - sampledAt)
        let age = wasHeld ? 0 : min(0.1, rawAge)
        let target = sceneTime + (held ? 0 : age * speed)
        guard let previousWall = lastWall else {
            value = target; lastWall = now; wasHeld = held
            return value
        }
        let rawDelta = max(0, now - previousWall)
        lastWall = now; wasHeld = held
        // MTKView produces no callbacks while occluded. Reanchor the first
        // visible frame without integrating any of that unseen wall interval.
        guard !held && rawDelta > 0 && rawDelta <= 0.25 else { return value }
        let delta = min(0.1, rawDelta)
        let nominalAdvance = delta * speed
        let predicted = value + nominalAdvance
        let alpha = -expm1(-delta / 0.18)
        let correction = (target - predicted) * alpha
        let advance = max(nominalAdvance * 0.25, min(nominalAdvance * 1.5, nominalAdvance + correction))
        let leadLimit = sceneTime + 0.1 * speed
        if rawAge >= 0.1 || value <= leadLimit {
            value = max(value, min(leadLimit, value + advance))
        } else {
            // A speed reduction can shrink the allowable lead below our current
            // phase. Recover by slowing down, without rewinding or a long stop.
            value += advance
        }
        return value
    }
}

struct RetinaHistoryStep {
    let time: Float
    let attackAlphas: SIMD3<Float>
    let releaseAlphas: SIMD3<Float>
    let update: Bool
    let seed: Bool
}

/// Stores only radiometric history. Spatial and RGB distortions are output-only.
struct RetinaHistoryPolicy {
    static let attackHalfLives = [0.025, 0.050, 0.090]
    static let releaseHalfLives = [0.24, 0.72, 1.65]
    private var valid = false
    private var time = 0.0

    mutating func invalidate() { valid = false }

    static func alpha(delta: Double, halfLife: Double) -> Double {
        -expm1(-log(2) * (delta.isFinite ? max(0, delta) : 0) / halfLife)
    }

    static func extent(width: Int, height: Int) -> RenderExtent {
        let width = max(1, (width + 1) / 2), height = max(1, (height + 1) / 2)
        let scale = min(1, sqrt(350_000 / Double(width * height)))
        return RenderExtent(width: max(1, Int(Double(width) * scale)),
                            height: max(1, Int(Double(height) * scale)))
    }

    mutating func advance(delta: Double, held: Bool) -> RetinaHistoryStep {
        // The host explicitly excludes hidden/held time. A slow active frame
        // still ages the retina; discarding it would abruptly erase its trails.
        let activeDelta = held || !delta.isFinite ? 0 : max(0, delta)
        time += activeDelta
        let seed = !valid
        let update = seed || activeDelta > 0
        func coefficients(_ halfLives: [Double]) -> SIMD3<Float> {
            seed ? SIMD3<Float>(repeating: 1) : SIMD3<Float>(
                Float(Self.alpha(delta: activeDelta, halfLife: halfLives[0])),
                Float(Self.alpha(delta: activeDelta, halfLife: halfLives[1])),
                Float(Self.alpha(delta: activeDelta, halfLife: halfLives[2])))
        }
        if update { valid = true }
        return RetinaHistoryStep(time: Float(time), attackAlphas: coefficients(Self.attackHalfLives),
                                 releaseAlphas: coefficients(Self.releaseHalfLives), update: update, seed: seed)
    }
}
