import Foundation
import simd

@main struct CameraTests {
    static func main() {
        testStableLens()
        testConservativeFrustum()
        testPresentation()
        testCubicPicking()
        var cases = 0
        for seconds: Float in [0, 30, 100, 1000, 10000] {
            for aspect: Float in [0.5, 1, 1.6, 3] {
                for elevation: Float in [0.45, 0.8, 1.1] {
                    var uniforms = ARUniforms()
                    uniforms.time.x = seconds
                    uniforms.camera = SIMD4<Float>(1.012, 0.987, seconds * 0.13, elevation)
                    uniforms.optics.x = 1.05
                    uniforms.world = SIMD4<Float>(0, -0.18, 0.13, 0)
                    WaterCamera.fill(&uniforms, aspect: aspect)
                    let eye = SIMD3<Float>(uniforms.eye.x, uniforms.eye.y, uniforms.eye.z)
                    let forward = SIMD3<Float>(uniforms.forward.x, uniforms.forward.y, uniforms.forward.z)
                    let right = SIMD3<Float>(uniforms.right.x, uniforms.right.y, uniforms.right.z)
                    let up = SIMD3<Float>(uniforms.up.x, uniforms.up.y, uniforms.up.z)
                    precondition(abs(simd_length(right) - 1) < 0.00001)
                    precondition(abs(simd_dot(right, up)) < 0.00001)
                    for x: Float in [-1, 1] { for y: Float in [-1, 1] {
                        let ray = forward + right * (x * uniforms.right.w) + up * (y * uniforms.up.w)
                        for h: Float in [-0.18, 0.13] {
                            let hit = eye + ray * ((h - eye.z) / ray.z)
                            precondition(hit.x >= 0.06 && hit.x <= 1.94 && hit.y >= 0.06 && hit.y <= 1.94,
                                         "World edge entered viewport at \(seconds), \(aspect), \(elevation): \(hit)")
                        }
                    }}
                    cases += 1
                }
            }
        }
        var picking = ARUniforms()
        picking.camera = SIMD4<Float>(1, 1, 0.23, 0.8)
        picking.optics.x = 0.9
        picking.world = SIMD4<Float>(0, -0.08, 0.08, 0)
        WaterCamera.fill(&picking, aspect: 1.6)
        let width = 64, height = 48
        var plane = [Float](repeating: 0, count: width * height * 4)
        for y in 0..<height { for x in 0..<width {
            plane[(y * width + x) * 4] = 0.03 * (2 * Float(x) / Float(width) - 1)
        }}
        for uv in [SIMD2<Float>(0.5, 0.5), SIMD2<Float>(0.1, 0.2), SIMD2<Float>(0.9, 0.8)] {
            let point = WaterCamera.surfacePoint(at: uv, uniforms: picking, water: plane, width: width, height: height)!
            let eye = SIMD3<Float>(picking.eye.x, picking.eye.y, picking.eye.z)
            let ray = SIMD3<Float>(picking.forward.x, picking.forward.y, picking.forward.z)
                + SIMD3<Float>(picking.right.x, picking.right.y, picking.right.z) * ((2 * uv.x - 1) * picking.right.w)
                + SIMD3<Float>(picking.up.x, picking.up.y, picking.up.z) * ((1 - 2 * uv.y) * picking.up.w)
            let distance = (0.03 * (eye.x - 1) - eye.z) / (ray.z - 0.03 * ray.x)
            let expected = eye + ray * distance
            precondition(simd_distance(point, SIMD2<Float>(expected.x, expected.y)) < 0.0001, "Click ray must hit the actual sloping water lattice")
        }
        precondition(WaterCamera.surfacePoint(at: SIMD2<Float>(-0.1, 0.5), uniforms: picking, water: plane, width: width, height: height) == nil)
        precondition(WaterCamera.surfacePoint(at: SIMD2<Float>(0.5, 0.5), uniforms: picking, water: [], width: width, height: height) == nil)
        print("PASS surface picking: sloping water, center and edge rays, invalid input")
        print("PASS fitted camera: \(cases) poses, rotation, portrait and ultrawide, both height bounds")
    }

    static func testCubicPicking() {
        // Keep the ray's crest coordinate fixed and tilt its other axis: the
        // returned horizontal point then independently reveals the hit height.
        let fixtures: [([Float], Float, Float)] = [
            ([0, 0.08, 0.08, 0], 0.75, 0.09),
            ([0.08, 0, 0, 0], 0.125, 0.06375),
            ([0, 0, 0, 0.08], 1.625, 0.085625),
        ]
        for axis in 0..<2 {
            for (profile, coordinate, height) in fixtures {
                var u = ARUniforms()
                u.eye = SIMD4<Float>(0.25, 0.25, 1, 1)
                u.eye[axis] = coordinate
                u.forward = SIMD4<Float>(0, 0, -1, 0)
                u.forward[1 - axis] = 0.5
                u.world = SIMD4<Float>(0, -0.12, 0.12, 0)
                var water = [Float](repeating: 0, count: 4 * 4 * 4)
                for y in 0..<4 { for x in 0..<4 {
                    water[(y * 4 + x) * 4] = profile[axis == 0 ? x : y]
                }}
                let point = WaterCamera.surfacePoint(at: SIMD2<Float>(0.5, 0.5),
                    uniforms: u, water: water, width: 4, height: 4)!
                let expected = 0.25 + 0.5 * (1 - height)
                precondition(abs(point[axis] - coordinate) < 0.00001)
                precondition(abs(point[1 - axis] - expected) < 0.00001,
                    "Cubic picking axis \(axis), coordinate \(coordinate): got \(point[1 - axis]), expected \(expected)")
            }
        }
        print("PASS cubic surface picking: thin crest and clamped ghost samples on both axes")
    }

    static func testStableLens() {
        func aperture(roll: Float = 0.5061, time: Float = 7.25,
                      target: SIMD2<Float> = SIMD2(1, 1),
                      heights: SIMD2<Float> = SIMD2(-0.03, 0.03)) -> SIMD2<Float> {
            var u = ARUniforms()
            u.time.x = time
            u.camera = SIMD4(target.x, target.y, roll, 70 * .pi / 180)
            u.optics.x = 61 * .pi / 180
            u.world = SIMD4(0, heights.x, heights.y, 0)
            WaterCamera.fill(&u, aspect: 1.6)
            return SIMD2(u.right.w, u.up.w)
        }
        let baseline = aperture()
        precondition(simd_distance(baseline, aperture(heights: SIMD2(-0.08, 0.08))) < 0.0000001,
                     "A splash must not zoom the camera through global water extrema")
        for roll: Float in stride(from: -Float.pi, through: Float.pi, by: 0.03) {
            precondition(simd_distance(baseline, aperture(roll: roll)) < 0.0000001,
                         "Rolling past a frustum corner must not change lens aperture")
        }
        for seconds: Float in stride(from: 0, through: 180, by: 0.5) {
            precondition(simd_distance(baseline, aperture(time: seconds)) < 0.0000001,
                         "Orbit heading must not create periodic zoom")
        }
        for x: Float in [-1 / 96, 0, 1 / 96] {
            for y: Float in [-1 / 64, 0, 1 / 64] {
                precondition(simd_distance(baseline, aperture(target: SIMD2(1 + x, 1 + y))) < 0.0000001,
                             "Rebasing the water window must not change lens aperture")
            }
        }
        print("PASS invariant lens: water extrema, full roll/orbit and snapped patch origins")
    }

    static func pose(time: Float = 0, target: SIMD2<Float> = SIMD2(1, 1)) -> ARUniforms {
        var u = ARUniforms()
        u.time.x = time
        u.camera = SIMD4(target.x, target.y, time * 4 * .pi / 180, 70 * .pi / 180)
        u.optics.x = 61 * .pi / 180
        u.world = SIMD4(0, -0.03, 0.03, 0)
        return u
    }

    static func testConservativeFrustum() {
        var cases = 0, intersections = 0
        let slabs: [SIMD2<Float>] = [SIMD2(-0.16, 0.12), SIMD2(-0.35, 0.25), SIMD2(-0.55, 0.60)]
        let targets: [SIMD2<Float>] = [SIMD2(1, 1), SIMD2(1 + 1 / 96, 1 - 1 / 64), SIMD2(0.96, 1), SIMD2(1.06, 1.02)]
        for seconds: Float in stride(from: 0, through: 180, by: 6) {
            for aspect: Float in [0.5, 1, 1.6, 3] {
                for elevation: Float in [0.45, 0.8, 1.1] {
                    for heights in slabs {
                        for target in targets {
                            var u = pose(time: seconds, target: target)
                            u.camera.z = seconds * 0.13
                            u.camera.w = elevation
                            u.world.y = heights.x; u.world.z = heights.y
                            WaterCamera.fill(&u, aspect: aspect)
                            precondition(abs(u.right.w / u.up.w - aspect) < 0.00001)
                            let eye = SIMD3(u.eye.x, u.eye.y, u.eye.z)
                            let forward = SIMD3(u.forward.x, u.forward.y, u.forward.z)
                            let right = SIMD3(u.right.x, u.right.y, u.right.z)
                            let up = SIMD3(u.up.x, u.up.y, u.up.z)
                            for x: Float in [-1, -0.5, 0, 0.5, 1] {
                                for y: Float in [-1, -0.5, 0, 0.5, 1] {
                                    let ray = forward + right * (x * u.right.w) + up * (y * u.up.w)
                                    precondition(ray.z < 0)
                                    for h in [heights.x, heights.y] {
                                        let distance = (h - eye.z) / ray.z
                                        precondition(distance > 0)
                                        let hit = eye + ray * distance
                                        precondition(hit.x >= 0.05998 && hit.y >= 0.05998 && hit.x <= 1.94002 && hit.y <= 1.94002,
                                                     "Unsafe cone: \(seconds),\(aspect),\(elevation),\(target),\(heights): \(hit)")
                                        intersections += 1
                                    }
                                }
                            }
                            cases += 1
                        }
                    }
                }
            }
        }
        print("PASS conservative frustum: \(cases) poses, \(intersections) intersections, normal and emergency slabs")
    }

    static func testPresentation() {
        let origin = SIMD2<Double>(-1, -1)
        var whole = CameraPresentation(), split = CameraPresentation()
        var a = pose(), b = pose()
        whole.fill(&a, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0, delta: 0, held: false, aspect: 1.6)
        split.fill(&b, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0, delta: 0, held: false, aspect: 1.6)
        a = pose(target: SIMD2(1.01, 1.005)); a.camera.w += 0.05; a.optics.x -= 0.08; a.camera.z += 0.03
        let source = a
        whole.fill(&a, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0.1, delta: 0.1, held: false, aspect: 1.6)
        for index in 1...6 {
            b = source
            split.fill(&b, origin: origin, snapshotSceneTime: 0, presentationSceneTime: Float(index) / 60, delta: 1 / 60, held: false, aspect: 1.6)
        }
        precondition(simd_distance(a.camera, b.camera) < 0.000001, "Damping must not depend on frame partition")
        precondition(abs(a.optics.x - b.optics.x) < 0.000001)
        let heldWorldEye = SIMD3<Double>(Double(a.eye.x) + origin.x, Double(a.eye.y) + origin.y, Double(a.eye.z))
        let heldRight = a.right
        let rebasedOrigin = origin + SIMD2(2 / 96, -2 / 64)
        var held = pose(time: 100, target: SIMD2(1.5, 0.5))
        whole.fill(&held, origin: rebasedOrigin, snapshotSceneTime: 100, presentationSceneTime: 100,
                   delta: 0.1, held: true, aspect: 1.6)
        let afterWorldEye = SIMD3<Double>(Double(held.eye.x) + rebasedOrigin.x, Double(held.eye.y) + rebasedOrigin.y, Double(held.eye.z))
        precondition(simd_distance(heldWorldEye, afterWorldEye) < 0.000001, "Holding/rebasing must not move the camera in world space")
        precondition(simd_distance(heldRight, held.right) < 0.000001)

        var moving = CameraPresentation()
        var previousRaw = 0.0, previousSmooth = 0.0, previousRawVelocity = 0.0, previousSmoothVelocity = 0.0
        var maxRawVelocityJump = 0.0, maxSmoothVelocityJump = 0.0, maxTargetOffset: Float = 0
        var aperture: Float?
        for frame in 0...360 {
            let time = Double(frame) / 60
            let sampleTime = floor(time * 30 + 0.000001) / 30
            let worldTarget = SIMD2(0.13 * sampleTime, 0.02 * sampleTime)
            let snappedOrigin = SIMD2(-1 + (worldTarget.x / (2 / 96)).rounded() * (2 / 96),
                                      -1 + (worldTarget.y / (2 / 64)).rounded() * (2 / 64))
            var u = pose(time: Float(sampleTime * 2.4), target: SIMD2(Float(worldTarget.x - snappedOrigin.x), Float(worldTarget.y - snappedOrigin.y)))
            moving.fill(&u, origin: snappedOrigin, snapshotSceneTime: Float(sampleTime * 2.4),
                        presentationSceneTime: Float(time * 2.4), delta: 1 / 60, held: false, aspect: 1.6)
            let smooth = snappedOrigin.x + Double(u.camera.x)
            let rawVelocity = (worldTarget.x - previousRaw) * 60
            let smoothVelocity = (smooth - previousSmooth) * 60
            if frame > 2 {
                maxRawVelocityJump = max(maxRawVelocityJump, abs(rawVelocity - previousRawVelocity))
                maxSmoothVelocityJump = max(maxSmoothVelocityJump, abs(smoothVelocity - previousSmoothVelocity))
            }
            previousRaw = worldTarget.x; previousSmooth = smooth
            previousRawVelocity = rawVelocity; previousSmoothVelocity = smoothVelocity
            maxTargetOffset = max(maxTargetOffset, simd_length(SIMD2(u.camera.x - 1, u.camera.y - 1)))
            if let aperture { precondition(abs(aperture - u.right.w) < 0.0000001, "Presented travel/rebase changed aperture") }
            else { aperture = u.right.w }
        }
        precondition(maxTargetOffset < WaterCamera.targetReserve)
        precondition(maxSmoothVelocityJump < maxRawVelocityJump * 0.15, "Pose smoothing must remove snapshot velocity jumps")

        var emergency = CameraPresentation()
        var normal = pose()
        emergency.fill(&normal, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0, delta: 0, held: false, aspect: 1.6)
        var extreme = pose(); extreme.world.y = -0.4
        emergency.fill(&extreme, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0, delta: 1 / 60, held: false, aspect: 1.6)
        var release = pose()
        emergency.fill(&release, origin: origin, snapshotSceneTime: 0, presentationSceneTime: 0, delta: 1 / 60, held: false, aspect: 1.6)
        precondition(extreme.right.w < release.right.w && release.right.w < normal.right.w)
        precondition(release.right.w - extreme.right.w < (normal.right.w - extreme.right.w) * 0.02,
                     "Emergency aperture must release gradually")
        print("PASS camera presentation: cadence invariance, held/rebased world pose, 30→60Hz source velocity jump \(maxRawVelocityJump)→\(maxSmoothVelocityJump) m/s; target residual \(maxTargetOffset) m; gradual emergency release")
    }
}
