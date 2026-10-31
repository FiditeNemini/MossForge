import SwiftUI
import Tweakbench

/// The MossForge macOS app — composition root. One `AppStore`
/// holds the engine (`MossBackend` when MOSS weights are found, else the canned `StubBackend`) and is
/// shared through the environment to every screen.
@main
struct MossForgeApp: App {
    @State private var store = AppStore()

    init() {
        // Launched via `swift run` there is no bundle: make it a regular foreground app with a menu bar.
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async { NSApplication.shared.activate(ignoringOtherApps: true) }
    }

    var body: some Scene {
        WindowGroup("MossForge") {
            RootView()
                .environment(store)
                .frame(minWidth: 1100, minHeight: 720)
                .tweakbench()  // live layout tweaking hook; a no-op with the bundled stand-in
        }
    }
}
