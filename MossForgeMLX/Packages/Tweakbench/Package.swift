// swift-tools-version: 6.0
import PackageDescription

// A no-op stand-in for Tweakbench, a private live-layout tweaking tool used while developing
// MossForge. The app tags its views with `.tweakable("page.region.element")`; here those tags
// do nothing, so the app builds and looks the same without the tool.
let package = Package(
    name: "Tweakbench",
    platforms: [.macOS(.v14)],
    products: [.library(name: "Tweakbench", targets: ["Tweakbench"])],
    targets: [.target(name: "Tweakbench")]
)
