import SwiftUI

/// The top-level surfaces, in sidebar order. (Merge lives in Models → Tools.)
enum Screen: String, CaseIterable, Identifiable {
    case generate, characterStudio, gallery, models, train, server, help, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .generate: "Generate"
        case .characterStudio: "Character Studio"
        case .gallery: "Gallery"
        case .models: "Models"
        case .train: "Train"
        case .server: "Server"
        case .help: "Help"
        case .settings: "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .generate: "sparkles"
        case .characterStudio: "person.2.crop.square.stack"
        case .gallery: "photo.on.rectangle"
        case .models: "shippingbox"
        case .train: "dumbbell"
        case .server: "server.rack"
        case .help: "questionmark.bubble"
        case .settings: "gearshape"
        }
    }
}

/// The custom-theme sidebar lists every screen.
extension Screen: SidebarScreen {}
