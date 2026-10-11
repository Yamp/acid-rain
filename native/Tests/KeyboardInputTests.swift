import AppKit
import MetalKit

@main struct KeyboardInputTests {
    static func main() {
        let view = WaterView(frame: .zero, device: nil)
        var routed: NativeKeyEvent?
        view.handleKey = { routed = $0 }
        let cases: [(String, UInt16, NSEvent.ModifierFlags, String)] = [
            ("n", 45, [], "n"), ("т", 45, [], "n"), ("Т", 45, [.shift], "N"),
            ("m", 46, [], "m"), ("ь", 46, [], "m"),
            ("a", 0, [], "a"), ("ф", 0, [], "a"), ("Ф", 0, [.shift], "a"),
            ("p", 35, [], "p"), ("з", 35, [], "p"), ("x", 7, [], "x"), ("ч", 7, [], "x"), ("o", 31, [], "o"), ("щ", 31, [], "o"), ("h", 4, [], "h"), ("р", 4, [], "h"), ("", 4, [], "h"),
            ("й", 12, [], "q"), ("п", 5, [], "g"), ("е", 17, [], "t"),
            ("в", 2, [], "d"), ("а", 3, [], "f"), ("с", 8, [], "c"),
            ("ц", 13, [], "w"), ("л", 40, [], "k"), ("у", 14, [], "e"),
            ("ш", 34, [], "i"), ("о", 38, [], "j"), ("д", 37, [], "l"),
            ("ы", 1, [], "s"), ("к", 15, [], "r"), ("м", 9, [], "v"),
            ("и", 11, [], "b"), ("К", 15, [.shift], "R"), ("И", 11, [.shift], "B"),
            ("О", 38, [.shift], "J"), ("R", 15, [.shift], "R"),
            ("х", 33, [], "["), ("ъ", 30, [], "]"), ("б", 43, [], ","),
            ("ю", 47, [], "."), ("+", 24, [.shift], "+"), ("№", 20, [], "3"),
            (" ", 49, [], " "), ("\t", 48, [], "\t"), ("?", 44, [.shift], "?"),
            ("\u{3}", 8, [.control], "c"), ("\u{F702}", 123, [], "\u{F702}")
        ]
        for (characters, code, modifiers, expected) in cases {
            let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: modifiers,
                timestamp: 0, windowNumber: 0, context: nil, characters: characters,
                charactersIgnoringModifiers: characters, isARepeat: false, keyCode: code)!
            view.keyDown(with: event)
            precondition(routed?.key == expected, "Expected \(characters.debugDescription) code \(code) to route as \(expected), got \(routed?.key ?? "nil")")
            precondition(routed?.keyCode == code && routed?.modifierFlags == modifiers)
        }
        print("PASS \(cases.count) keyboard cases through WaterView: ASCII, Russian layout, Shift resets, Control, arrows")
    }
}
