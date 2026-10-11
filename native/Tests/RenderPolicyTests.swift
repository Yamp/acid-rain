import Foundation

@main struct RenderPolicyTests {
    static func main() {
        let hd = RenderPolicy.extent(width: 3840, height: 2160, quality: .high)
        precondition(hd.width * hd.height <= 1_650_000)
        precondition(abs(Double(hd.width) / Double(hd.height) - 16.0 / 9.0) < 0.01)
        let tiny = RenderPolicy.extent(width: 1, height: 1, quality: .high)
        precondition(tiny.width == 1 && tiny.height == 1)
        let zero = RenderPolicy.extent(width: 0, height: 0, quality: .low)
        precondition(zero.width == 1 && zero.height == 1)
        precondition(RenderQuality.high.next == .balanced)
        precondition(RenderQuality.low.next == .native)
        precondition(RenderQuality.native.next == .auto)
        precondition(RenderQuality.auto.next == .high)
        let native = RenderPolicy.extent(width: 3024, height: 1964, quality: .native)
        precondition(native.width == 3024 && native.height == 1964)
        var policy = RenderPolicy()
        for _ in 0..<29 { policy.observe(gpuMilliseconds: 40) }
        precondition(policy.automaticPixelBudget == 1_200_000)
        policy.observe(gpuMilliseconds: 40)
        precondition(policy.automaticPixelBudget < 1_200_000)
        let reduced = policy.automaticPixelBudget
        for _ in 0..<30 { policy.observe(gpuMilliseconds: 10) }
        precondition(policy.automaticPixelBudget == reduced, "avoid quick scale oscillation")
        for _ in 0..<150 { policy.observe(gpuMilliseconds: 3) }
        precondition(policy.automaticPixelBudget > reduced)
        var clock = CloudClock()
        let first = clock.consume(time: 0, origin: SIMD2<Double>(-1, -1), revision: 0)
        precondition(first.clear && first.delta == 0)
        let moving = clock.consume(time: 0.03, origin: SIMD2<Double>(-1 + 2.0 / 96, -1 - 2.0 / 64), revision: 0)
        precondition(abs(moving.delta - 0.03) < 1e-8 && moving.shift == SIMD2<Int32>(1, -1) && !moving.clear)
        let held = clock.consume(time: 0.03, origin: SIMD2<Double>(-1 + 2.0 / 96, -1 - 2.0 / 64), revision: 0)
        precondition(held.delta == 0 && held.shift == .zero && !held.clear, "Repeated display snapshots must not advance pigment")
        let cleared = clock.consume(time: 0.03, origin: .zero, revision: 1)
        precondition(cleared.clear && cleared.delta == 0)
        let late = clock.consume(time: 60, origin: .zero, revision: 1)
        precondition(late.delta <= 0.128, "Suspended display must not enqueue unbounded catch-up GPU work")
        let contact = CloudTransportPlan(delta: 0.06, endTime: 0.30, impactTimes: [0.29])
        let birthStep = contact.injections.firstIndex(where: { !$0.isEmpty })!
        let birthTime = 0.24 + Double(birthStep) * contact.substepDelta
        precondition(birthTime >= 0.29 && birthTime - 0.29 <= 0.008,
            "A contact at .29 must not be transported over the whole .24...30 frame")
        precondition(contact.substepCount == 8 && birthStep == 7)
        let reordered = CloudTransportPlan(delta: 0.06, endTime: 0.30, impactTimes: [0.299, 0.241, 0.23, 0.29])
        precondition(reordered.sourceOrder == [2, 1, 3, 0], "Spawn IDs need not order impact times")
        precondition(reordered.injections[0] == 0..<1, "Previously impacted sources start before transport")
        let paused = CloudTransportPlan(delta: 0, endTime: 0.30, impactTimes: [0.29])
        precondition(paused.substepCount == 0 && paused.injections == [0..<1])
        let repeated = CloudTransportPlan(delta: 0, endTime: 0.30, impactTimes: [])
        precondition(repeated.injections == [0..<0])
        var random: UInt64 = 0xD09C10
        for _ in 0..<300 {
            random = random &* 6364136223846793005 &+ 1
            let delta = 0.001 + Double(random % 127_000) / 1_000_000
            let now = 10.0, start = now - delta
            var times: [Double] = []
            for _ in 0..<17 {
                random = random &* 6364136223846793005 &+ 1
                times.append(start + delta * Double(random % 10_000) / 10_000)
            }
            let plan = CloudTransportPlan(delta: delta, endTime: now, impactTimes: times)
            precondition(plan.substepCount <= 16 && plan.substepDelta <= 0.008)
            var seen = Set<Int>()
            for (bucket, range) in plan.injections.enumerated() {
                let birth = start + Double(bucket) * plan.substepDelta
                for source in range {
                    let original = plan.sourceOrder[source]
                    precondition(seen.insert(original).inserted, "Every source is injected once")
                    precondition(birth + 1e-12 >= times[original] && birth - times[original] <= 0.008 + 1e-12,
                        "Variable frame cadence must never transport pigment before contact")
                }
            }
            precondition(seen.count == times.count)
        }
        print("PASS cloud source timing: contact boundary, reordered impacts, pause, 5100 variable-cadence events")
        print("PASS cloud clock: repeated frames, pause, precise world rebase, generation reset, bounded catch-up")
        print("PASS render extent, Retina pixel budget, quality cycle, adaptive hysteresis")
    }
}
