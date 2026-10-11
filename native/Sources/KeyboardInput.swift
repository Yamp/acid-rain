import AppKit
import MetalKit

struct NativeKeyEvent {
    let key: String
    let keyCode: UInt16
    let modifierFlags: NSEvent.ModifierFlags
    let isARepeat: Bool

    init(_ event: NSEvent) {
        key = Self.normalizedKey(event)
        keyCode = event.keyCode
        modifierFlags = event.modifierFlags
        isARepeat = event.isARepeat
    }

    private static func normalizedKey(_ event: NSEvent) -> String {
        let text = event.characters ?? ""
        // Preserve the typed shortcut on Latin layouts. Non-Latin input still
        // controls the simulation through the same physical keys; no system
        // input source is changed. Control-C often arrives as U+0003.
        if !text.isEmpty && text.unicodeScalars.allSatisfy({ $0.value >= 32 && $0.value < 127 }) {
            return text
        }
        if text == "\t" { return text }
        let physical: [UInt16: String] = [
            0: "a", 45: "n", 46: "m",
            35: "p", 7: "x", 31: "o", 4: "h", 12: "q", 5: "g", 17: "t", 2: "d", 3: "f", 8: "c",
            13: "w", 40: "k", 14: "e", 34: "i", 38: "j", 37: "l", 1: "s",
            15: "r", 9: "v", 11: "b", 33: "[", 30: "]", 43: ",", 47: ".",
            24: "=", 27: "-", 18: "1", 19: "2", 20: "3", 21: "4", 29: "0",
            49: " ", 48: "\t", 44: "/"
        ]
        guard let key = physical[event.keyCode] else { return text }
        if event.modifierFlags.contains(.shift) {
            if ["r", "b", "j", "n"].contains(key) { return key.uppercased() }
            if key == "=" { return "+" }
            if key == "/" { return "?" }
        }
        return key
    }
}

final class WaterView: MTKView {
    var handleKey: ((NativeKeyEvent) -> Void)?
    var handleClick: ((SIMD2<Float>) -> Void)?
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) { handleKey?(NativeKeyEvent(event)) }
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        guard bounds.width > 0, bounds.height > 0 else { return }
        let point = convert(event.locationInWindow, from: nil)
        let y = Float((point.y - bounds.minY) / bounds.height)
        handleClick?(SIMD2<Float>(Float((point.x - bounds.minX) / bounds.width), isFlipped ? y : 1 - y))
    }
}
