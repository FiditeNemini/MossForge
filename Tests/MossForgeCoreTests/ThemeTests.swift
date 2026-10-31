import Testing

@testable import MossForgeCore

/// The app's visual themes are data, so their legibility is testable: every theme must meet
/// WCAG AA for body text, keep its accent visible on its cards, and keep the white label on a
/// prominent (accent-filled) button readable.
@Suite struct ThemeTests {
    @Test func hexParsesAndRelativeLuminanceMatchesWCAG() throws {
        let white = try #require(ThemeColor(hex: "#FFFFFF"))
        let black = try #require(ThemeColor(hex: "000000"))
        #expect(white.relativeLuminance == 1)
        #expect(black.relativeLuminance == 0)
        #expect(abs(ThemeColor.contrast(white, black) - 21) < 1e-9)
        #expect(ThemeColor(hex: "#12345") == nil)
        #expect(ThemeColor(hex: "#GG0000") == nil)
    }

    @Test func nativeIsTheDefaultAndDefersToTheSystem() {
        #expect(ThemeID.default == .native)
        let native = ThemeSpec.spec(for: .native)
        #expect(native.appearance == .system)
        #expect(native.palette == nil)  // system colours + accent, exactly today's look
    }

    @Test func everyThemeHasAUniqueNameAndBlurb() {
        let specs = ThemeID.allCases.map(ThemeSpec.spec(for:))
        #expect(Set(specs.map(\.name)).count == specs.count)
        #expect(specs.allSatisfy { !$0.blurb.isEmpty })
    }

    @Test(arguments: ThemeID.allCases.filter { $0 != .native })
    func customThemesAreLegible(_ id: ThemeID) throws {
        let p = try #require(ThemeSpec.spec(for: id).palette)
        let white = ThemeColor(hex: "#FFFFFF")!
        // Body text: WCAG AA (4.5:1) on both the window canvas and the cards.
        #expect(ThemeColor.contrast(p.text, p.canvas) >= 4.5)
        #expect(ThemeColor.contrast(p.text, p.surface) >= 4.5)
        // macOS draws menu pickers, bordered buttons and links with their *label* in the tint,
        // so the accent is also text: AA (4.5:1) on cards and canvas.
        #expect(ThemeColor.contrast(p.accent, p.surface) >= 4.5)
        #expect(ThemeColor.contrast(p.accent, p.canvas) >= 4.5)
        // Prominent buttons / selected segments draw a white label on the accent. A light theme
        // meets AA; on a dark canvas both can't hold at once (text needs a light accent, a white
        // label a dark one), so a dark theme makes Apple's trade — at least macOS's own dark-mode
        // blue (#0A84FF, 3.65:1).
        let dark = ThemeSpec.spec(for: id).appearance == .dark
        #expect(ThemeColor.contrast(white, p.accent) >= (dark ? 3.65 : 4.5))
    }

    @Test func darkModeSystemBlueIsTheLabelFloor() {
        // Pins the reference the dark-theme floor above is taken from.
        let blue = ThemeColor(hex: "#0A84FF")!
        #expect(abs(ThemeColor.contrast(ThemeColor(hex: "#FFFFFF")!, blue) - 3.65) < 0.01)
    }

    @Test(arguments: ThemeID.allCases.filter { $0 != .native })
    func appearanceMatchesThePalette(_ id: ThemeID) throws {
        // A dark canvas must force dark mode (so system controls draw light-on-dark), and vice versa.
        let spec = ThemeSpec.spec(for: id)
        let p = try #require(spec.palette)
        let dark = p.canvas.relativeLuminance < 0.18
        #expect(spec.appearance == (dark ? .dark : .light))
    }

    @Test func unknownStoredIDFallsBackToDefault() {
        #expect(ThemeID(storedValue: "aurora") == .aurora)
        #expect(ThemeID(storedValue: "neon-2031") == .native)
        #expect(ThemeID(storedValue: nil) == .native)
    }
}
