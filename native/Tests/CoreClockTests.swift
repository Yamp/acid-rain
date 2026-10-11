import Foundation

@main struct CoreClockTests {
    static func main() throws {
        let core = try CoreBridge(seed: 7)
        let initial = core.current()
        try core.stepOffline(delta: 1.0 / 60)
        let moved = core.current()
        precondition(moved.simulationSampleAt >= initial.simulationSampleAt)
        precondition(moved.simulationSampleAt <= moved.publishedAt)
        precondition(moved.state.frame > initial.state.frame)
        try core.commandOffline(2, value: 1.2)
        let speed = core.current()
        precondition(speed.simulationSampleAt == moved.simulationSampleAt)
        precondition(speed.state.uniforms.time.x == moved.state.uniforms.time.x)
        try core.commandOffline(14)
        precondition(core.current().simulationSampleAt == moved.simulationSampleAt,
            "Command-only publication must preserve the physics clock anchor")
        try core.commandOffline(1)
        try core.stepOffline(delta: 0.1)
        let held = core.current()
        precondition(held.simulationSampleAt == moved.simulationSampleAt)
        precondition(held.state.frame == moved.state.frame)
        try core.commandOffline(1)
        try core.stepOffline(delta: 1.0 / 60)
        let resumed = core.current()
        precondition(resumed.simulationSampleAt > moved.simulationSampleAt)
        precondition(resumed.state.frame > moved.state.frame)
        print("PASS actual CoreBridge sample clock: step-start timestamp, speed/burst command republish, pause/resume")
    }
}
