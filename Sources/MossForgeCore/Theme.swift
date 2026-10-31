import Foundation

/// The app's selectable visual themes. Stored in settings by raw value. INSTANTIATE: keep Native
/// as the default; add, rename or retune the custom themes — ThemeTests keeps each one legible.
public enum ThemeID: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Stock macOS: system colours, system accent, follows light/dark.
    case native
    /// A photographic darkroom: warm charcoal, safelight-orange accent, editorial serif headings.
    case darkroom
    /// Night sky: deep indigo with teal and violet glows, glass cards, rounded type.
    case aurora
    /// A gallery wall: warm paper, white cards, vermilion ink accent, serif headings.
    case gallery

    public static let `default`: ThemeID = .native
    public var id: String { rawValue }

    /// Decode a persisted value, falling back to the default for a missing or retired theme.
    public init(storedValue: String?) {
        self = storedValue.flatMap(ThemeID.init(rawValue:)) ?? .default
    }
}

/// An sRGB colour as plain data, with the WCAG 2.x luminance / contrast maths.
public struct ThemeColor: Sendable, Equatable, Hashable {
    public var red: Double
    public var green: Double
    public var blue: Double
    public var alpha: Double

    public init(red: Double, green: Double, blue: Double, alpha: Double = 1) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }

    /// `#RRGGBB` or `RRGGBB`; `nil` for anything else.
    public init?(hex: String, alpha: Double = 1) {
        let digits = hex.hasPrefix("#") ? hex.dropFirst() : Substring(hex)
        guard digits.count == 6, let v = UInt32(digits, radix: 16) else { return nil }
        self.init(
            red: Double((v >> 16) & 0xFF) / 255, green: Double((v >> 8) & 0xFF) / 255,
            blue: Double(v & 0xFF) / 255, alpha: alpha)
    }

    /// WCAG relative luminance (0 = black, 1 = white). Ignores alpha.
    public var relativeLuminance: Double {
        func linear(_ c: Double) -> Double { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    }

    /// WCAG contrast ratio, 1…21.
    public static func contrast(_ a: ThemeColor, _ b: ThemeColor) -> Double {
        let (la, lb) = (a.relativeLuminance, b.relativeLuminance)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    public func opacity(_ alpha: Double) -> ThemeColor {
        ThemeColor(red: red, green: green, blue: blue, alpha: alpha)
    }
}

/// Which system appearance a theme runs under.
public enum ThemeAppearance: String, Sendable {
    case system, light, dark
}

/// Typeface family for headings / body (maps to SwiftUI `Font.Design`).
public enum ThemeTypeface: String, Sendable {
    case standard, serif, rounded, monospaced
}

/// A custom theme's colour tokens. Opaque colours are the ones the legibility tests check.
public struct ThemePalette: Sendable, Equatable {
    /// Window background.
    public var canvas: ThemeColor
    /// Soft light sources painted over the canvas (top-leading, bottom-trailing).
    public var glowA: ThemeColor
    public var glowB: ThemeColor
    /// Card (group box) fill and hairline.
    public var surface: ThemeColor
    public var stroke: ThemeColor
    /// Image wells (preview / canvas areas) — darker or lighter than a card.
    public var well: ThemeColor
    /// Primary text; secondary/tertiary derive from it.
    public var text: ThemeColor
    public var accent: ThemeColor
    /// Film-grain overlay strength (0 = none).
    public var grain: Double
    /// Cards are translucent glass over the glows rather than solid.
    public var glassCards: Bool
}

/// Everything that defines a theme: identity, appearance, palette, type, and shape.
public struct ThemeSpec: Sendable, Equatable {
    public var id: ThemeID
    public var name: String
    public var blurb: String
    public var appearance: ThemeAppearance
    /// `nil` = the native look (system colours and accent).
    public var palette: ThemePalette?
    public var headingTypeface: ThemeTypeface
    public var bodyTypeface: ThemeTypeface
    public var cornerRadius: Double

    public static func spec(for id: ThemeID) -> ThemeSpec {
        switch id {
        case .native: native
        case .darkroom: darkroom
        case .aurora: aurora
        case .gallery: gallery
        }
    }

    private static func c(_ hex: String, _ alpha: Double = 1) -> ThemeColor { ThemeColor(hex: hex, alpha: alpha)! }

    static let native = ThemeSpec(
        id: .native, name: "Native", blurb: "Stock macOS — system colours, follows light and dark.",
        appearance: .system, palette: nil, headingTypeface: .standard, bodyTypeface: .standard,
        cornerRadius: 8)

    static let darkroom = ThemeSpec(
        id: .darkroom, name: "Darkroom",
        blurb: "Warm charcoal under a safelight glow, with a touch of film grain.",
        appearance: .dark,
        palette: ThemePalette(
            canvas: c("#12100E"), glowA: c("#E3581C", 0.14), glowB: c("#8A5A2B", 0.10),
            surface: c("#1D1A17"), stroke: c("#EFE7DC", 0.08), well: c("#0B0A09"),
            text: c("#EFE7DC"), accent: c("#E3581C"), grain: 0.05, glassCards: false),
        headingTypeface: .serif, bodyTypeface: .standard, cornerRadius: 6)

    static let aurora = ThemeSpec(
        id: .aurora, name: "Aurora",
        blurb: "Deep night indigo lit by teal and violet — glass cards, rounded type.",
        appearance: .dark,
        palette: ThemePalette(
            canvas: c("#070A17"), glowA: c("#17C3B2", 0.32), glowB: c("#6D5AF0", 0.30),
            surface: c("#10142A"), stroke: c("#E6EAFF", 0.12), well: c("#04060F"),
            text: c("#E6EAFF"), accent: c("#8271EF"), grain: 0, glassCards: true),
        headingTypeface: .rounded, bodyTypeface: .rounded, cornerRadius: 14)

    static let gallery = ThemeSpec(
        id: .gallery, name: "Gallery",
        blurb: "Warm paper and white cards, like a gallery wall — vermilion ink accent.",
        appearance: .light,
        palette: ThemePalette(
            canvas: c("#F3EFE7"), glowA: c("#FFFFFF", 0.70), glowB: c("#E9DFCD", 0.60),
            surface: c("#FFFFFF"), stroke: c("#1C1A17", 0.10), well: c("#E7E1D6"),
            text: c("#1C1A17"), accent: c("#B4432A"), grain: 0.025, glassCards: false),
        headingTypeface: .serif, bodyTypeface: .standard, cornerRadius: 4)
}
