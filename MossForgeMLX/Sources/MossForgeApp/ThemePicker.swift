import AppKit
import MossForgeCore
import SwiftUI

/// Settings → Appearance: one live preview card per theme. Choosing one applies instantly.
public struct ThemePicker: View {
    @Binding var selection: ThemeID

    public init(selection: Binding<ThemeID>) { _selection = selection }

    public var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140, maximum: 240), spacing: 12)], spacing: 12) {
            ForEach(ThemeID.allCases) { id in
                ThemeCard(theme: Theme(id), selected: selection == id)
                    .onTapGesture { withAnimation(.easeInOut(duration: 0.25)) { selection = id } }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(Theme(id).spec.name) theme")
                    .accessibilityAddTraits(selection == id ? [.isButton, .isSelected] : .isButton)
                    .accessibilityAction { selection = id }
            }
        }
        .padding(.vertical, 4)
    }
}

private struct ThemeCard: View {
    let theme: Theme
    let selected: Bool
    @Environment(\.theme) private var current

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ThemeMiniature(theme: theme)
                .frame(height: 104)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(selected ? current.accent : Color.primary.opacity(0.12), lineWidth: selected ? 2.5 : 1)
                }
            HStack(spacing: 6) {
                Text(theme.spec.name).font(.callout.weight(.semibold)).fontDesign(theme.headingDesign)
                if selected {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(current.accent).font(.caption)
                }
            }
            Text(theme.spec.blurb)
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .lineLimit(3, reservesSpace: true)
        }
        .contentShape(Rectangle())
    }
}

/// A tiny window in the theme's own colours: sidebar, a card with a heading, a slider, and a
/// prominent button — enough to judge the palette and type before switching.
private struct ThemeMiniature: View {
    let theme: Theme

    private var canvas: Color { theme.palette.map { Color($0.canvas) } ?? Color(nsColor: .windowBackgroundColor) }
    private var surface: Color { theme.palette.map { Color($0.surface) } ?? Color(nsColor: .controlBackgroundColor) }
    private var stroke: Color { theme.palette.map { Color($0.stroke) } ?? Color(nsColor: .separatorColor) }
    private var text: Color { theme.palette.map { Color($0.text) } ?? Color(nsColor: .labelColor) }
    private var sidebar: Color { text.opacity(0.06) }

    var body: some View {
        ZStack {
            if let p = theme.palette { ThemeBackdrop(palette: p) } else { canvas }
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(0..<5) { i in
                        Capsule().fill(i == 1 ? theme.accent : text.opacity(0.22))
                            .frame(width: i == 1 ? 30 : 24 + CGFloat(i % 2) * 6, height: 4)
                    }
                    Spacer()
                }
                .padding(.top, 12).padding(.leading, 9)
                .frame(width: 46, alignment: .topLeading)
                .frame(maxHeight: .infinity)
                .background(sidebar)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Aa").font(.system(size: 17, weight: .semibold, design: theme.headingDesign))
                        .foregroundStyle(text)
                    VStack(alignment: .leading, spacing: 6) {
                        Capsule().fill(text.opacity(0.35)).frame(width: 56, height: 3)
                        ZStack(alignment: .leading) {
                            Capsule().fill(text.opacity(0.15)).frame(height: 3)
                            Capsule().fill(theme.accent).frame(width: 44, height: 3)
                            Circle().fill(Color.white).frame(width: 9, height: 9)
                                .shadow(color: .black.opacity(0.3), radius: 1).offset(x: 40)
                        }
                        RoundedRectangle(cornerRadius: 4).fill(theme.accent).frame(width: 46, height: 13)
                            .overlay(Capsule().fill(Color.white.opacity(0.9)).frame(width: 24, height: 2.5))
                    }
                    .padding(8)
                    .background {
                        RoundedRectangle(cornerRadius: min(theme.cornerRadius, 8), style: .continuous)
                            .fill(theme.palette?.glassCards == true ? AnyShapeStyle(surface.opacity(0.55)) : AnyShapeStyle(surface))
                        RoundedRectangle(cornerRadius: min(theme.cornerRadius, 8), style: .continuous)
                            .strokeBorder(stroke)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
        }
        .environment(\.colorScheme, theme.colorScheme ?? systemScheme)
    }

    /// Native follows the system, so its card previews whatever the Mac is set to.
    @Environment(\.colorScheme) private var systemScheme
}
