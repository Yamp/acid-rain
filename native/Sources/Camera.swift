import Foundation
import simd

enum WaterCamera {
    static let nominalHeightEnvelope = SIMD2<Float>(-0.16, 0.12)
    static let targetReserve: Float = 0.04
    /// Top-left normalized drawable coordinates, using the same physical lattice as Metal.
    static func surfacePoint(at uv: SIMD2<Float>, uniforms u: ARUniforms,
                             water: [Float], width: Int, height: Int) -> SIMD2<Float>? {
        guard uv.x.isFinite, uv.y.isFinite, (0...1).contains(uv.x), (0...1).contains(uv.y),
              width > 1, height > 1, water.count == width * height * 4 else { return nil }
        let eye = SIMD3<Float>(u.eye.x, u.eye.y, u.eye.z)
        let ray = SIMD3<Float>(u.forward.x, u.forward.y, u.forward.z)
            + SIMD3<Float>(u.right.x, u.right.y, u.right.z) * ((2 * uv.x - 1) * u.right.w)
            + SIMD3<Float>(u.up.x, u.up.y, u.up.z) * ((1 - 2 * uv.y) * u.up.w)
        guard ray.z < -0.0001 else { return nil }
        func cubic(_ h: SIMD4<Float>, _ t: Float) -> Float {
            // Same coefficients as waterCubicCoefficients in Caustics.metal.
            let a = 0.5 * (-h.x + 3 * h.y - 3 * h.z + h.w)
            let b = h.x - 2.5 * h.y + 2 * h.z - 0.5 * h.w
            let c = 0.5 * (h.z - h.x)
            return ((a * t + b) * t + c) * t + h.y
        }
        func residual(_ distance: Float) -> Float {
            let p = eye + ray * distance
            let x = p.x * 0.5 * Float(width), y = p.y * 0.5 * Float(height)
            // Floor the unbounded lattice coordinate; clamp ghost samples only.
            let ix = Int(floor(x)), iy = Int(floor(y))
            let fx = x - Float(ix), fy = y - Float(iy)
            var rows = SIMD4<Float>(repeating: 0)
            for row in 0..<4 {
                let sy = max(0, min(height - 1, iy + row - 1))
                var samples = SIMD4<Float>(repeating: 0)
                for column in 0..<4 {
                    let sx = max(0, min(width - 1, ix + column - 1))
                    samples[column] = water[(sy * width + sx) * 4]
                }
                rows[row] = cubic(samples, fx)
            }
            return p.z - cubic(rows, fy)
        }
        var near = max(0, (u.world.z + 0.00001 - eye.z) / ray.z)
        var far = (u.world.y - 0.00001 - eye.z) / ray.z
        guard far > near, residual(near) >= 0, residual(far) <= 0 else { return nil }
        // Match the primary Metal surface intersection, including its lattice interpolation.
        var a = residual(near), b = residual(far)
        for _ in 0..<20 {
            let weight = max(0.05, min(0.95, a / max(a - b, 1e-9)))
            let middle = near + (far - near) * weight
            let value = residual(middle)
            if value > 0 { near = middle; a = value } else { far = middle; b = value }
        }
        let point = eye + ray * ((near + far) * 0.5)
        guard point.x >= 0, point.y >= 0, point.x <= 2, point.y <= 2 else { return nil }
        return SIMD2<Float>(point.x, point.y)
    }

    static func fill(_ uniforms: inout ARUniforms, aspect: Float,
                     heightEnvelope: SIMD2<Float>? = nil) {
        let target = SIMD3<Float>(uniforms.camera.x, uniforms.camera.y, 0)
        let heading = uniforms.time.x / 180 * (2 * Float.pi)
        let requestedEnvelope = heightEnvelope ?? nominalHeightEnvelope
        let heights = SIMD2(min(requestedEnvelope.x, uniforms.world.y - 0.002),
                            max(requestedEnvelope.y, uniforms.world.z + 0.002))
        let offset = simd_length(SIMD2(target.x - 1, target.y - 1))
        let radius = max(0.005, 0.94 - max(targetReserve, offset))
        // A complete rotation/roll fits inside this circle. Normal patch snaps
        // and small presentation lag remain inside a fixed target reserve, so
        // neither can change aperture. Only extraordinary heights/targets invoke
        // the safety fallback; ordinary rain never participates in lens fitting.
        let requiredElevation = atan(max(abs(heights.x), abs(heights.y)) / radius) + 0.01
        let elevation = max(uniforms.camera.w, requiredElevation)
        let s = sin(elevation), c = cos(elevation)
        let orbitDistance = max(1, (heights.y + 0.08) / s)
        let eye = target + orbitDistance * SIMD3<Float>(-c * cos(heading), -c * sin(heading), s)
        let forward = simd_normalize(target - eye)
        let basisRight = simd_normalize(SIMD3<Float>(forward.y, -forward.x, 0))
        let basisUp = simd_cross(basisRight, forward)
        let roll = uniforms.camera.z
        let right = basisRight * cos(roll) + basisUp * sin(roll)
        let up = basisUp * cos(roll) - basisRight * sin(roll)
        let halfWidth = tan(uniforms.optics.x * 0.5)
        let halfHeight = halfWidth / max(0.1, aspect)
        // The intersection of a diagonal ray cone with a horizontal plane is
        // an ellipse. Its farthest points from the target lie on its major axis.
        // Solving those two endpoints against radius R gives these closed-form
        // limits on tan(cone half-angle). They do not depend on heading or roll.
        var coneLimit: Float = s / max(c, 0.00001) * 0.99
        for h in [heights.x, heights.y] {
            for sign: Float in [-1, 1] {
                let numerator = radius * s + sign * h * c
                let denominator = orbitDistance - h * s + sign * radius * c
                if denominator > 0 {
                    coneLimit = min(coneLimit, max(0.00001, numerator / denominator))
                }
            }
        }
        let requestedCone = hypot(halfWidth, halfHeight)
        // C1 smooth minimum removes the derivative kink when a slowly evolving
        // scene FOV crosses the safety limit. It never exceeds either bound.
        let rounding: Float = 0.02
        let blend = max(0, 1 - abs(requestedCone - coneLimit) / rounding)
        let cone = min(requestedCone, coneLimit) - 0.25 * rounding * blend * blend
        let scale = max(0.00001, cone) / max(0.00001, requestedCone)
        uniforms.eye = SIMD4<Float>(eye, 1)
        uniforms.forward = SIMD4<Float>(forward, 0)
        uniforms.right = SIMD4<Float>(right, halfWidth * scale)
        uniforms.up = SIMD4<Float>(up, halfHeight * scale)
    }
}

/// Smooth camera parameters in world space, independently of water-grid rebases.
/// The exact critically damped step is stable and preserves velocity when a new
/// simulation snapshot arrives. Orbit/roll phase comes from the display clock.
struct CameraPresentation {
    private var pose: SIMD4<Double>?
    private var velocity = SIMD4<Double>(repeating: 0)
    private var rollResidual = 0.0
    private var rollVelocity = 0.0
    private var lastSceneTime: Float = 0
    private var envelope = SIMD2(Double(WaterCamera.nominalHeightEnvelope.x), Double(WaterCamera.nominalHeightEnvelope.y))

    mutating func fill(_ uniforms: inout ARUniforms, origin: SIMD2<Double>,
                       snapshotSceneTime: Float, presentationSceneTime: Float,
                       delta: Double, held: Bool, aspect: Float) {
        let target = SIMD4(origin.x + Double(uniforms.camera.x),
                           origin.y + Double(uniforms.camera.y),
                           Double(uniforms.camera.w), Double(uniforms.optics.x))
        let rollRate = 4 * Double.pi / 180
        let desiredResidual = Double(uniforms.camera.z) - Double(snapshotSceneTime) * rollRate
        let dt = delta.isFinite ? min(0.1, max(0, delta)) : 0
        if pose == nil {
            pose = target
            rollResidual = desiredResidual
            lastSceneTime = presentationSceneTime
        } else if !held {
            let omega = 1.0 / 0.06
            let decay = exp(-omega * dt)
            let displacement = pose! - target
            let transient = velocity + omega * displacement
            pose = target + (displacement + transient * dt) * decay
            velocity = (velocity - omega * transient * dt) * decay
            let rollDisplacement = rollResidual - desiredResidual
            let rollTransient = rollVelocity + omega * rollDisplacement
            rollResidual = desiredResidual + (rollDisplacement + rollTransient * dt) * decay
            rollVelocity = (rollVelocity - omega * rollTransient * dt) * decay
            lastSceneTime = presentationSceneTime
        }
        // Rare out-of-envelope waves must be fitted immediately for safety, but
        // releasing that emergency allowance is gradual. Normal extrema leave
        // this envelope exactly constant, including in paused/help frames.
        let desiredEnvelope = SIMD2(min(Double(WaterCamera.nominalHeightEnvelope.x), Double(uniforms.world.y) - 0.002),
                                    max(Double(WaterCamera.nominalHeightEnvelope.y), Double(uniforms.world.z) + 0.002))
        let release = held ? 1 : exp(-dt / 2)
        envelope.x = min(desiredEnvelope.x, desiredEnvelope.x + (envelope.x - desiredEnvelope.x) * release)
        envelope.y = max(desiredEnvelope.y, desiredEnvelope.y + (envelope.y - desiredEnvelope.y) * release)
        let presented = pose!
        uniforms.camera = SIMD4(Float(presented.x - origin.x), Float(presented.y - origin.y),
                                Float(Double(lastSceneTime) * rollRate + rollResidual), Float(presented.z))
        uniforms.optics.x = Float(presented.w)
        uniforms.time.x = lastSceneTime
        WaterCamera.fill(&uniforms, aspect: aspect, heightEnvelope: SIMD2(Float(envelope.x), Float(envelope.y)))
    }
}
