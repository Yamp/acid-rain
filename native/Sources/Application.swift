import AppKit
import MetalKit
import QuartzCore

final class AppController: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var window: NSWindow!
    private var view: WaterView!
    private var renderer: MetalRenderer!
    private var core: CoreBridge!
    private let options: LaunchOptions
    private var hud = NSTextField(labelWithString: "")
    private var hudPanel = NSVisualEffectView()
    private var helpPanel = NSVisualEffectView()
    private var helpVisible = false
    private var hudUntil = CACurrentMediaTime() + 8
    private var lastHUDUpdate: Double = 0
    private var tickCount = 0
    private var began = CACurrentMediaTime()
    private var smokeTimer: Timer?
    private var controlTimer: Timer?
    private var controlIndex = 0

    init(options: LaunchOptions) { self.options = options }

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            core = try CoreBridge(seed: options.seed)
            try options.apply(to: core)
            renderer = try MetalRenderer(core: core, libraryURL: try LaunchOptions.libraryURL())
            renderer.quality = options.quality
            renderer.cloudStudy = options.dropDemo || options.cloudStudy
            renderer.showFallingDrops = options.showFallingDrops
            renderer.subjectiveIntensity = options.subjectiveIntensity
            renderer.recordMeasurements = options.benchmark != nil
            makeMenu()
            window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: options.width, height: options.height),
                styleMask: [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView],
                backing: .buffered, defer: false)
            window.title = "Acid Rain · Metal"
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.backgroundColor = .black
            window.minSize = NSSize(width: 640, height: 420)
            window.collectionBehavior = [.fullScreenPrimary]
            window.isReleasedWhenClosed = false
            window.delegate = self
            window.center()
            view = WaterView(frame: window.contentView!.bounds, device: renderer.device)
            view.autoresizingMask = [.width, .height]
            view.colorPixelFormat = .bgra8Unorm_srgb
            view.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
            view.depthStencilPixelFormat = .invalid
            view.framebufferOnly = true
            view.preferredFramesPerSecond = 60
            view.clearColor = MTLClearColorMake(0, 0, 0, 1)
            view.delegate = renderer
            view.enableSetNeedsDisplay = false
            view.isPaused = false
            view.handleKey = { [weak self] event in self?.handle(event) }
            view.handleClick = { [weak self] uv in
                guard let self, !self.helpVisible else { return }
                self.renderer.drop(at: uv); self.showHUD()
            }
            window.contentView = view
            makeOverlays()
            renderer.onFailure = { [weak self] message in self?.reportFailure(message) }
            core.onFailure = { [weak self] message in self?.reportFailure(message) }
            renderer.onFrame = { [weak self] frame, stats in
                self?.tickCount += 1
                self?.updateHUD(frame, measurement: stats)
            }
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(view)
            NSApp.activate(ignoringOtherApps: true)
            core.start()
            began = CACurrentMediaTime()
            if options.fullscreen { window.toggleFullScreen(nil) }
            if let seconds = options.smoke {
                smokeTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in self?.finishSmoke() }
            }
            if options.exerciseControls {
                controlTimer = Timer.scheduledTimer(withTimeInterval: 0.30, repeats: true) { [weak self] _ in self?.exerciseNextControl() }
            }
        } catch { reportFailure(String(describing: error)) }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func windowWillClose(_ notification: Notification) { NSApp.terminate(nil) }
    func windowDidResize(_ notification: Notification) { showHUD() }
    func windowDidChangeOcclusionState(_ notification: Notification) {
        guard core != nil, view != nil else { return }
        let visible = window.occlusionState.contains(.visible)
        renderer.setOccluded(!visible)
        view.isPaused = !visible
        core.setHeld(helpVisible || !visible)
    }

    private func makeMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem(); menu.addItem(appItem)
        let appMenu = NSMenu(title: "Acid Rain"); appItem.submenu = appMenu
        appMenu.addItem(withTitle: "О программе Acid Rain", action: #selector(showAbout), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Выйти из Acid Rain", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let viewItem = NSMenuItem(); menu.addItem(viewItem)
        let viewMenu = NSMenu(title: "Вид"); viewItem.submenu = viewMenu
        let full = viewMenu.addItem(withTitle: "Полный экран", action: #selector(toggleFullscreen), keyEquivalent: "f")
        full.target = self
        let help = viewMenu.addItem(withTitle: "Управление", action: #selector(toggleHelp), keyEquivalent: "h")
        help.target = self
        NSApp.mainMenu = menu
    }

    @objc private func showAbout() {
        NSApp.orderFrontStandardAboutPanel(options: [.applicationName: "Acid Rain · Metal",
            .applicationVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.2.0", .credits: NSAttributedString(string: "Живая вода, цветной свет и бесконечное дно.\nRust simulation · native Metal rendering")])
    }
    @objc private func toggleFullscreen() { window.toggleFullScreen(nil) }
    @objc private func toggleHelp() {
        helpVisible.toggle()
        helpPanel.isHidden = !helpVisible
        renderer.helpVisible = helpVisible
        renderer.resetHistory(preserveRetina: true)
        core.setHeld(helpVisible || !window.occlusionState.contains(.visible))
        showHUD()
    }

    private func makeOverlays() {
        for panel in [hudPanel, helpPanel] {
            panel.material = .hudWindow
            panel.blendingMode = .withinWindow
            panel.state = .active
            panel.wantsLayer = true
            panel.layer?.cornerRadius = 14
            panel.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(panel)
        }
        hud.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        hud.textColor = NSColor.white.withAlphaComponent(0.9)
        hud.maximumNumberOfLines = 3
        hud.translatesAutoresizingMaskIntoConstraints = false
        hudPanel.addSubview(hud)
        NSLayoutConstraint.activate([
            hudPanel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 22),
            hudPanel.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -22),
            hudPanel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -22),
            hud.leadingAnchor.constraint(equalTo: hudPanel.leadingAnchor, constant: 15),
            hud.trailingAnchor.constraint(equalTo: hudPanel.trailingAnchor, constant: -15),
            hud.topAnchor.constraint(equalTo: hudPanel.topAnchor, constant: 11),
            hud.bottomAnchor.constraint(equalTo: hudPanel.bottomAnchor, constant: -11)
        ])
        let helpString = """
        ACID RAIN · METAL
        Живая вода и цветной свет

        Пробел             Пауза / продолжить
        [ / ]  или  ← / →   Скорость · s — сброс к 2.4×
        ↑ / ↓              Интенсивность дождя
        , / .              Диаметр капель · R — сброс дождя
        r                  Всплеск цветных капель
        o / клик           Одна капля в воду
        a                  Показать / скрыть падающие капли
        x                  Очистить краситель и капли
        p                  Наблюдение капли / обычный свет
        v / b              Обесцвечивание · B — сброс
        − / +              Интенсивность трипа
        j / l              RGB-сдвиг · J — сброс к 2 px
        e                  RGB-расслоение и эффекты сетчатки
        n / m              Субъективные эффекты · N — сброс
        i                  Смешивание кадров

        g / Tab            Следующая сцена · t — Journey
        d                  Сток / обычная вода
        f                  Пороги / обычная вода
        c                  Движение камеры вдоль воды
        w                  Течение воды
        k                  Следующее качество
        0 / 1 / 2 / 3 / 4  Auto / High / Balanced / Low / Native

        ⌘ F                Полный экран
        h / ?              Скрыть / показать управление
        q / Esc / ⌘ Q      Выход (Esc сперва закрывает справку)

        Справка удерживает анимацию. High ограничивает число
        пикселей на Retina; Auto подстраивает его под время GPU.
        """
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        scroll.translatesAutoresizingMaskIntoConstraints = false
        let helpText = NSTextView(frame: NSRect(x: 0, y: 0, width: 538, height: 640))
        helpText.string = helpString
        helpText.isEditable = false
        helpText.isSelectable = false
        helpText.drawsBackground = false
        helpText.isVerticallyResizable = true
        helpText.isHorizontallyResizable = false
        helpText.autoresizingMask = .width
        helpText.textContainer?.widthTracksTextView = true
        helpText.textContainerInset = .zero
        helpText.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        helpText.textColor = .white
        helpText.setAccessibilityLabel("Управление Acid Rain")
        scroll.documentView = helpText
        helpPanel.addSubview(scroll)
        let preferredHelpHeight = helpPanel.heightAnchor.constraint(equalToConstant: 650)
        preferredHelpHeight.priority = .defaultHigh
        preferredHelpHeight.isActive = true
        NSLayoutConstraint.activate([
            helpPanel.heightAnchor.constraint(lessThanOrEqualTo: view.heightAnchor, constant: -80),
            helpPanel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            helpPanel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            helpPanel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
            helpPanel.widthAnchor.constraint(equalToConstant: 590),
            scroll.leadingAnchor.constraint(equalTo: helpPanel.leadingAnchor, constant: 26),
            scroll.trailingAnchor.constraint(equalTo: helpPanel.trailingAnchor, constant: -26),
            scroll.topAnchor.constraint(equalTo: helpPanel.topAnchor, constant: 24),
            scroll.bottomAnchor.constraint(equalTo: helpPanel.bottomAnchor, constant: -24)
        ])
        helpPanel.isHidden = true
    }

    private func updateHUD(_ frame: SimulationFrame, measurement: RenderMeasurement?) {
        let now = CACurrentMediaTime()
        guard now - lastHUDUpdate > 0.2 else { return }
        lastHUDUpdate = now
        hudPanel.isHidden = now > hudUntil && !helpVisible && frame.state.paused == 0
        let s = frame.state
        let scenes = ["Journey", "Acid", "Aurora", "Storm", "Abyss", "Lava", "Prism"]
        let modes = ["Вода", "Сток", "Пороги"]
        let performance = measurement.map { String(format: " · %.1f ms GPU · %d×%d", $0.gpuMilliseconds, $0.width, $0.height) } ?? ""
        hud.stringValue = String(format: "%@ · %@ · %.2f× · Trip %.1f · LSD %.1f · %@%@\nДождь %.0f/s · Капли %.2f× · Bleach %.2f× · RGB %.2f px · Полёт %@ · h — управление",
            scenes[min(6, Int(s.scene))], modes[min(2, Int(s.mode))], s.speed, s.uniforms.time.z, renderer.subjectiveIntensity,
            renderer.quality.name + (renderer.cloudStudy ? " · Капля" : ""), s.paused != 0 ? " · Пауза" : performance,
            s.rain, s.drop_size, s.bleach, s.uniforms.time.w, renderer.showFallingDrops ? "вкл" : "скрыт")
    }

    private func showHUD() { hudUntil = CACurrentMediaTime() + 5; hudPanel.isHidden = false }

    private func handle(_ event: NativeKeyEvent) {
        if event.modifierFlags.contains(.command) { return }
        let key = event.key
        if event.modifierFlags.contains(.control), key.lowercased() == "c" { NSApp.terminate(nil); return }
        let toggles = ["h", "?", " ", "e", "i", "c", "w", "f", "d", "k", "R", "B", "J", "N", "s", "o", "O", "x", "X", "p", "P", "a", "A"]
        if event.isARepeat && toggles.contains(key) { return }
        handleKey(key, keyCode: event.keyCode)
    }

    private func handleKey(_ key: String, keyCode: UInt16 = 0) {
        showHUD()
        let s = core.current().state
        func command(_ id: UInt32, _ value: Float = 0) { core.command(id, value: value) }
        func adjust(_ id: UInt32, _ delta: Float) { core.adjust(id, by: delta) }
        switch key {
        case "h", "?": toggleHelp()
        case "q": NSApp.terminate(nil)
        case " ": command(1); renderer.resetHistory(preserveRetina: true)
        case "e": command(13); renderer.resetHistory()
        case "i": renderer.interpolation.toggle(); renderer.resetHistory()
        case "c": command(11)
        case "w": command(12)
        case "d": command(9, s.mode == 1 ? 0 : 1); renderer.resetHistory()
        case "f": command(9, s.mode == 2 ? 0 : 2); renderer.resetHistory()
        case "g", "\t": command(10, Float((s.scene + 1) % 7)); renderer.resetHistory()
        case "t": command(10, 0); renderer.resetHistory()
        case "k": renderer.quality = renderer.quality.next; command(8, renderer.quality.shaderLevel)
        case "0", "1", "2", "3", "4": renderer.quality = RenderQuality(rawValue: Int(key)!)!; command(8, renderer.quality.shaderLevel)
        case "+", "=": adjust(6, 0.1)
        case "-": adjust(6, -0.1)
        case "j": adjust(7, -0.25)
        case "l": adjust(7, 0.25)
        case "J": command(7, 2)
        case "[": adjust(2, -0.1)
        case "]": adjust(2, 0.1)
        case "s": command(2, 2.4)
        case ",": adjust(4, -0.25)
        case ".": adjust(4, 0.25)
        case "R": command(3, 32); command(4, 1.5)
        case "v": adjust(5, -0.25)
        case "b": adjust(5, 0.25)
        case "B": command(5, 1)
        case "r": command(14)
        case "o", "O": renderer.drop(at: SIMD2<Float>(0.5, 0.5))
        case "x", "X": command(16); renderer.resetHistory()
        case "p", "P": renderer.cloudStudy.toggle(); renderer.resetHistory()
        case "a", "A": renderer.showFallingDrops.toggle(); renderer.resetHistory()
        case "n": renderer.subjectiveIntensity = max(0, renderer.subjectiveIntensity - 0.1)
        case "m": renderer.subjectiveIntensity = min(2, renderer.subjectiveIntensity + 0.1)
        case "N": renderer.subjectiveIntensity = 1
        default:
            switch keyCode {
            case 53: if helpVisible { toggleHelp() } else { NSApp.terminate(nil) }
            case 123: adjust(2, -0.1)
            case 124: adjust(2, 0.1)
            case 125: adjust(3, -2)
            case 126: adjust(3, 2)
            default: break
            }
        }
    }

    private func exerciseNextControl() {
        let keys = ["h", "h", " ", " ", "d", "d", "f", "f", "k", "k", "k", "k", "k", "4", "1", "0", "2", "3", "1", "r", "o", "g", "t", "c", "c", "w", "w", "j", "l", "J", "e", "e", "i", "i", "]", "[", "s", ".", ",", "R", "b", "v", "B", "+", "-", "a", "a", "n", "m", "N"]
        guard controlIndex < keys.count else { controlTimer?.invalidate(); return }
        handleKey(keys[controlIndex]); controlIndex += 1
        if controlIndex == 10 { window.setContentSize(NSSize(width: 960, height: 600)) }
        if controlIndex == 20 { window.setContentSize(NSSize(width: options.width, height: options.height)) }
    }

    private func finishSmoke() {
        do {
            guard tickCount > 0, core.currentFailure() == nil else { throw NativeFailure("Native window smoke did not render successfully.") }
            if let path = options.benchmark {
                var details: [String: Any] = ["mode": "window", "submitted_frames": tickCount,
                    "wall_seconds": CACurrentMediaTime() - began, "controls_exercised": controlIndex,
                    "drawable_width": Int(view.drawableSize.width), "drawable_height": Int(view.drawableSize.height)]
                for (key, value) in renderer.presentationSummary() { details[key] = value }
                try writeBenchmark(path: path, measurements: renderer.measurements(), device: renderer.device.name, extra: details)
            }
            print("PASS native window smoke: \(tickCount) submitted frames, \(controlIndex) controls, \(renderer.device.name)")
            NSApp.terminate(nil)
        } catch { reportFailure(String(describing: error)) }
    }

    private func reportFailure(_ message: String) {
        FileHandle.standardError.write(Data(("Acid Rain Metal: \(message)\n").utf8))
        if options.smoke != nil { exit(1) }
        let alert = NSAlert(); alert.messageText = "Не удалось запустить Acid Rain"
        alert.informativeText = message; alert.alertStyle = .critical
        alert.runModal(); NSApp.terminate(nil)
    }
}
