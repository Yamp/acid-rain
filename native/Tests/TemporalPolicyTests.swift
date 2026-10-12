import Foundation

@main struct TemporalPolicyTests {
    static func main() {
        for cadence in [1.0 / 30, 1.0 / 60, 1.0 / 120] {
            var clock = PresentationClock()
            var previous = -1.0
            for i in 0...240 {
                let now = Double(i) * cadence
                let sampledAt = max(0, floor((now - 0.008) * 60) / 60)
                let scene = clock.sample(sceneTime: sampledAt * 2.4, sampledAt: sampledAt,
                    speed: 2.4, now: now, held: false)
                precondition(scene >= previous && abs(scene - now * 2.4) < 1e-8,
                    "A stable camera clock must be independent of CPU publication delay and display cadence")
                previous = scene
            }
        }
        var clock = PresentationClock()
        _ = clock.sample(sceneTime: 10, sampledAt: 0, speed: 2.4, now: 0, held: false)
        let beforeCommand = clock.sample(sceneTime: 10, sampledAt: 0, speed: 2.4, now: 0.025, held: false)
        let afterCommand = clock.sample(sceneTime: 10, sampledAt: 0, speed: 2.4, now: 0.026, held: false)
        precondition(afterCommand >= beforeCommand && abs(afterCommand - beforeCommand - 0.0024) < 1e-8,
            "Republishing a command must not reset the scene clock anchor")
        let held = clock.sample(sceneTime: 10.04, sampledAt: 0.026, speed: 2.4, now: 0.05, held: true)
        for now in [0.07, 0.2, 1.0, 100.0] {
            precondition(clock.sample(sceneTime: 10.04, sampledAt: 0.026, speed: 2.4, now: now, held: true) == held)
        }
        let resumed = clock.sample(sceneTime: 10.04, sampledAt: 100, speed: 0.4, now: 100.01, held: false)
        precondition(resumed >= held && resumed - held < 0.02, "Resume and speed changes cannot jump the camera")
        for i in 1...1000 {
            _ = clock.sample(sceneTime: 10.04, sampledAt: 100, speed: 0.4,
                now: 100.01 + Double(i) / 60, held: false)
        }
        precondition(clock.value <= 10.04 + 0.4 * 0.1 + 1e-9,
            "A missing physics snapshot must not let the camera drift forever")

        var occluded = PresentationClock()
        let visible = occluded.sample(sceneTime: 2, sampledAt: 0, speed: 2.4, now: 0.015, held: false)
        let restored = occluded.sample(sceneTime: 2, sampledAt: 0, speed: 2.4, now: 90, held: false)
        precondition(restored == visible, "No draw callbacks occur while occluded; the first visible frame must preserve camera phase")
        var brieflyHidden = PresentationClock()
        let beforeHide = brieflyHidden.sample(sceneTime: 2, sampledAt: 0, speed: 2.4, now: 0.01, held: false)
        brieflyHidden.reanchor(at: 0.12)
        let visibleAgain = brieflyHidden.sample(sceneTime: 2, sampledAt: 0, speed: 2.4, now: 0.121, held: false)
        precondition(visibleAgain >= beforeHide && visibleAgain - beforeHide <= 0.003,
            "Explicit visibility reanchor must exclude a short hidden interval too")
        var slowed = PresentationClock()
        var prior = slowed.sample(sceneTime: 10, sampledAt: 0, speed: 3, now: 0.1, held: false)
        precondition(abs(prior - 10.3) < 1e-12)
        var error = prior - 10
        for frame in 1...240 {
            let now = 0.1 + Double(frame) / 60
            let scene = 10 + Double(frame) / 60 * 0.1
            let current = slowed.sample(sceneTime: scene, sampledAt: now, speed: 0.1, now: now, held: false)
            precondition(current > prior && current - prior <= 0.1 / 60 * 1.5 + 1e-12,
                "A high-to-low speed change must slow the camera continuously, not freeze it for seconds")
            precondition(current - scene <= error + 1e-12, "Old prediction lead must shrink with fresh snapshots")
            prior = current; error = current - scene
        }
        precondition(error < 0.02, "The camera must converge back to the slower simulation clock")

        var flashPolicy = RetinaHistoryPolicy()
        _ = flashPolicy.advance(delta: 0, held: false)
        let flash = flashPolicy.advance(delta: 0.05, held: false)
        precondition(flash.attackAlphas.z > 0.30, "A 50 ms glint must visibly enter even the slow retinal layer")

        for halfLife in RetinaHistoryPolicy.attackHalfLives + RetinaHistoryPolicy.releaseHalfLives {
            var results: [Double] = []
            for cadence in [1.0 / 30, 1.0 / 60, 1.0 / 120] {
                var value = 0.0
                for _ in 0..<Int((1 / cadence).rounded()) {
                    let alpha = RetinaHistoryPolicy.alpha(delta: cadence, halfLife: halfLife)
                    value += (1 - value) * alpha
                }
                precondition(abs(value - (1 - pow(2, -1 / halfLife))) < 1e-12)
                results.append(value)
            }
            precondition((results.max()! - results.min()!) < 1e-12)
        }
        var retina = RetinaHistoryPolicy()
        let seed = retina.advance(delta: 1.0 / 60, held: false)
        precondition(seed.update && seed.seed && seed.attackAlphas == SIMD3<Float>(repeating: 1) && seed.releaseAlphas == SIMD3<Float>(repeating: 1))
        let active = retina.advance(delta: 0.02, held: false)
        precondition(active.update && !active.seed)
        let phase = active.time
        for dt in [0.016, 0.2, 40.0] {
            let pause = retina.advance(delta: dt, held: true)
            precondition(!pause.update && pause.time == phase && !pause.seed)
        }
        retina.invalidate()
        let resetHeld = retina.advance(delta: 10, held: true)
        precondition(resetHeld.seed && resetHeld.update && resetHeld.time == phase)
        let gap = retina.advance(delta: 1, held: false)
        precondition(!gap.seed && gap.update && abs(gap.time - phase - 1) < 1e-6,
            "A slow active frame ages existing reservoirs without reseeding them")
        let empty = retina.advance(delta: 0, held: false)
        precondition(!empty.update)
        for invalid in [Double.nan,Double.infinity,-1] {
            let ignored = retina.advance(delta: invalid, held: false)
            precondition(!ignored.update && ignored.time == empty.time
                && ignored.attackAlphas == .zero && ignored.releaseAlphas == .zero,
                "Invalid wall intervals cannot poison time or history coefficients")
        }
        let wide = RetinaHistoryPolicy.extent(width: 8192, height: 4096)
        precondition(wide.width * wide.height <= 350_000 && abs(Double(wide.width) / Double(wide.height) - 2) < 0.01)
        print("PASS presentation clock: 3 cadences, publication jitter, command republish, hold/resume, speed, bounded stale lead")
        print("PASS retinal history: exact exponential half-lives, reset/held state, slow active frames, bounded memory")
    }
}
