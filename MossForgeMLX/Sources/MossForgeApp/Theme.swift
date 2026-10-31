import AppKit
import CoreGraphics
import MossForgeCore
import SwiftUI

// The theme engine. A theme is data (`ThemeSpec`, in the Core module, where its legibility is
// tested); this file turns it into SwiftUI: one `.appTheme(_:)` at the root sets the tint,
// colour scheme, type, card style, and window backdrop, and screens read `\.theme` for the
// few tokens that aren't expressible as semantic styles (accent rings, image wells).

extension Color {
    public init(_ c: ThemeColor) {
        self.init(.sRGB, red: c.red, green: c.green, blue: c.blue, opacity: c.alpha)
    }
}

extension Font.Design {
    public init(_ t: ThemeTypeface) {
        switch t {
        case .standard: self = .default
        case .serif: self = .serif
        case .rounded: self = .rounded
        case .monospaced: self = .monospaced
        }
    }
}

/// A theme resolved for SwiftUI. `palette == nil` is the native look: every token falls back
/// to the system colour it replaced, so the Native theme renders exactly as before.
public struct Theme: Equatable, Sendable {
    public let spec: ThemeSpec

    public init(_ id: ThemeID) { spec = ThemeSpec.spec(for: id) }

    public var palette: ThemePalette? { spec.palette }
    public var isNative: Bool { palette == nil }

    public var accent: Color { palette.map { Color($0.accent) } ?? .accentColor }
    public var well: Color { palette.map { Color($0.well) } ?? Color.black.opacity(0.2) }
    public var cornerRadius: CGFloat { CGFloat(spec.cornerRadius) }
    public var headingDesign: Font.Design { Font.Design(spec.headingTypeface) }
    public var bodyDesign: Font.Design { Font.Design(spec.bodyTypeface) }

    public var colorScheme: ColorScheme? {
        switch spec.appearance {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }
}

private struct ThemeKey: EnvironmentKey {
    static let defaultValue = Theme(.default)
}

extension EnvironmentValues {
    public var theme: Theme {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

extension View {
    /// Apply a theme to a view tree (the app root, or a Settings preview card).
    public func appTheme(_ theme: Theme) -> some View { modifier(ThemeModifier(theme: theme)) }
}

/// One fixed modifier chain for every theme — only the values change. Branching here would wrap
/// the whole `NavigationSplitView` in `_ConditionalContent`, so switching between Native and a
/// custom theme would tear down and rebuild the window's split view and toolbar; AppKit's
/// sidebar/titlebar integration is fragile under that. Native passes the neutral values.
private struct ThemeModifier: ViewModifier {
    let theme: Theme

    func body(content: Content) -> some View {
        let palette = theme.palette
        content
            .environment(\.theme, theme)
            .tint(palette.map { Color($0.accent) })
            .fontDesign(palette == nil ? nil : theme.bodyDesign)
            .foregroundStyle(palette.map { Color($0.text) } ?? Color.primary)
            .groupBoxStyle(ThemedGroupBoxStyle(theme: theme))
            .scrollContentBackground(palette == nil ? .automatic : .hidden)
            .background {
                if let palette { ThemeBackdrop(palette: palette).ignoresSafeArea() }
            }
            // Not `.preferredColorScheme`: after a theme forced `.dark`, passing `nil` (Native = follow the
            // system) doesn't hand the window back to the system, so Native kept a dark window under
            // light-mode text. AppKit's appearance resets cleanly. (From MossForge 9f63af5.)
            .background(AppearanceSync(scheme: theme.colorScheme))
    }
}

/// Applies the theme's appearance to the app and its windows (`nil` = follow the system).
private struct AppearanceSync: NSViewRepresentable {
    let scheme: ColorScheme?

    func makeNSView(context: Context) -> NSView { NSView() }

    func updateNSView(_ view: NSView, context: Context) {
        let appearance: NSAppearance? = switch scheme {
        case .dark?: NSAppearance(named: .darkAqua)
        case .light?: NSAppearance(named: .aqua)
        default: nil
        }
        DispatchQueue.main.async {
            NSApp.appearance = appearance
            for window in NSApp.windows { window.appearance = appearance }
        }
    }
}

/// The window canvas: base colour, two soft light sources, optional film grain.
public struct ThemeBackdrop: View {
    let palette: ThemePalette

    public init(palette: ThemePalette) { self.palette = palette }

    public var body: some View {
        GeometryReader { geo in
            let reach = max(geo.size.width, geo.size.height)
            ZStack {
                Color(palette.canvas)
                RadialGradient(
                    colors: [Color(palette.glowA), Color(palette.glowA).opacity(0)],
                    center: .topLeading, startRadius: 0, endRadius: reach * 0.75)
                RadialGradient(
                    colors: [Color(palette.glowB), Color(palette.glowB).opacity(0)],
                    center: .bottomTrailing, startRadius: 0, endRadius: reach * 0.7)
                if palette.grain > 0 {
                    Rectangle()
                        .fill(ImagePaint(image: Image(decorative: GrainTile.image, scale: 2)))
                        .opacity(palette.grain)
                        .blendMode(.overlay)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// A tiling monochrome noise texture, generated once (deterministic, so it never shimmers).
private enum GrainTile {
    static let image: CGImage = {
        let side = 192
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        var pixels = [UInt8](repeating: 0, count: side * side)
        for i in pixels.indices {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            pixels[i] = UInt8(truncatingIfNeeded: state >> 56)
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(
            width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: side,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }()
}

/// Cards: the stock group box for Native; for a custom theme, the palette surface (or glass over the glows), a hairline, and a
/// heading in the theme's display face.
public struct ThemedGroupBoxStyle: GroupBoxStyle {
    let theme: Theme

    public init(theme: Theme) { self.theme = theme }

    @ViewBuilder public func makeBody(configuration: Configuration) -> some View {
        if theme.isNative {
            GroupBox(configuration)  // the stock style, untouched
        } else {
            themed(configuration)
        }
    }

    private func themed(_ configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            configuration.label
                .font(.subheadline.weight(.semibold))
                .fontDesign(theme.headingDesign)
            configuration.content
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background { surface }
    }

    @ViewBuilder private var surface: some View {
        let shape = RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
        if let p = theme.palette {
            ZStack {
                if p.glassCards {
                    shape.fill(.ultraThinMaterial)
                    shape.fill(Color(p.surface).opacity(0.45))
                } else {
                    shape.fill(Color(p.surface))
                        .shadow(color: .black.opacity(p.canvas.relativeLuminance > 0.5 ? 0.06 : 0.25), radius: 6, y: 2)
                }
                shape.strokeBorder(
                    p.glassCards
                        ? AnyShapeStyle(LinearGradient(
                            colors: [Color(p.stroke).opacity(1.6), Color(p.stroke).opacity(0.4)],
                            startPoint: .topLeading, endPoint: .bottomTrailing))
                        : AnyShapeStyle(Color(p.stroke)),
                    lineWidth: 1)
            }
        }
    }
}

/// A sidebar destination. INSTANTIATE: make the app's `Screen` enum conform.
public protocol SidebarScreen: Hashable, CaseIterable, Identifiable where AllCases: RandomAccessCollection {
    var title: String { get }
    /// SF Symbol name.
    var symbol: String { get }
}

/// The sidebar for a custom theme: the screens as rows, the selection filled with the theme
/// accent. Arrow keys move the selection, like the stock sidebar. Native keeps the stock
/// `List` sidebar (see TEMPLATE.md).
public struct ThemedSidebar<Screen: SidebarScreen>: View {
    @Binding var selection: Screen
    @Environment(\.theme) private var theme
    @FocusState private var focused: Bool
    @State private var hovered: Screen?

    public init(selection: Binding<Screen>) { _selection = selection }

    public var body: some View {
        // A plain top-aligned stack, not a ScrollView: in the split view's sidebar a ScrollView under the
        // full-size titlebar draws its rows below the titlebar inset but hit-tests them without it, so
        // hover/clicks landed ~2 rows off. The few rows never need to scroll. (From MossForge.)
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Screen.allCases) { screen in
                row(screen)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onKeyPress(.upArrow) { move(-1) }
        .onKeyPress(.downArrow) { move(1) }
    }

    private func row(_ screen: Screen) -> some View {
        let selected = screen == selection
        let shape = RoundedRectangle(cornerRadius: min(theme.cornerRadius, 8), style: .continuous)
        return Button {
            selection = screen
            focused = true
        } label: {
            HStack(spacing: 9) {
                Image(systemName: screen.symbol)
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: 18)
                    .foregroundStyle(selected ? AnyShapeStyle(.white) : AnyShapeStyle(theme.accent))
                Text(screen.title)
                    .font(.body.weight(selected ? .semibold : .regular))
                    .foregroundStyle(selected ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background {
                if selected {
                    shape.fill(theme.accent)
                } else if hovered == screen {
                    shape.fill(.primary.opacity(0.07))
                }
            }
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 ? screen : (hovered == screen ? nil : hovered) }
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    private func move(_ delta: Int) -> KeyPress.Result {
        let all = Array(Screen.allCases)
        guard let i = all.firstIndex(of: selection) else { return .ignored }
        let next = min(max(i + delta, 0), all.count - 1)
        selection = all[next]
        return .handled
    }
}
