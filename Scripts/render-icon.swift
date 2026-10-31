// Renders the MossForge app icon — "forged growth": a deep-forest squircle lit from below by a forge
// glow, a luminous moss leaf rising from a gold four-point forge spark, a few sparks drifting up.
// AppKit only (no SVG rasteriser). Follows the macOS icon grid: 824 px body centred on a 1024 canvas
// with a soft drop shadow. Then `Scripts/make-icons.sh` cuts the AppIcon set and the .icns.
//
//   swift Scripts/render-icon.swift [out.png]     (default: branding/icon-1024.png)
import AppKit

let size = 1024.0
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "branding/icon-1024.png"

func color(_ hex: UInt32, _ a: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
        blue: CGFloat(hex & 0xFF) / 255, alpha: a)
}
/// Top-left-origin point (design coordinates) → AppKit's bottom-left origin.
func p(_ x: Double, _ y: Double) -> NSPoint { NSPoint(x: x, y: size - y) }

let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size), bitsPerSample: 8, samplesPerPixel: 4,
    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let ctx = NSGraphicsContext.current!.cgContext

// MARK: body + shadow (macOS grid: 100 px margin, ~185 px corner)
let body = NSRect(x: 100, y: 100, width: 824, height: 824)
let squircle = NSBezierPath(roundedRect: body, xRadius: 186, yRadius: 186)
NSGraphicsContext.saveGraphicsState()
let shadow = NSShadow()
shadow.shadowColor = color(0x000000, 0.45)
shadow.shadowBlurRadius = 28
shadow.shadowOffset = NSSize(width: 0, height: -12)
shadow.set()
color(0x0B2318).setFill()
squircle.fill()
NSGraphicsContext.restoreGraphicsState()

NSGraphicsContext.saveGraphicsState()
squircle.addClip()
// Forest gradient, top-left → bottom-right.
NSGradient(colors: [color(0x123B2A), color(0x0E2E22), color(0x081A13)], atLocations: [0, 0.55, 1], colorSpace: .sRGB)!
    .draw(in: body, angle: -60)
// Forge glow from below.
NSGradient(colors: [color(0xFB923C, 0.62), color(0xF97316, 0.26), color(0xC2410C, 0)], atLocations: [0, 0.4, 1],
    colorSpace: .sRGB)!
    .draw(fromCenter: p(512, 900), radius: 0, toCenter: p(512, 900), radius: 520, options: [])
// Cool moss bloom behind the leaf.
NSGradient(colors: [color(0x4ADE80, 0.20), color(0x4ADE80, 0)], atLocations: [0, 1], colorSpace: .sRGB)!
    .draw(fromCenter: p(520, 430), radius: 0, toCenter: p(520, 430), radius: 360, options: [])
// Faint rim light along the top edge.
NSGradient(colors: [color(0xFFFFFF, 0.10), color(0xFFFFFF, 0)], atLocations: [0, 1], colorSpace: .sRGB)!
    .draw(in: NSRect(x: 100, y: 760, width: 824, height: 164), angle: -90)

// MARK: leaf — a tilted vesica from the spark (bottom) to the tip (upper right)
let base = p(470, 760), tip = p(650, 250)
let leaf = NSBezierPath()
leaf.move(to: base)
leaf.curve(to: tip, controlPoint1: p(250, 610), controlPoint2: p(360, 300))
leaf.curve(to: base, controlPoint1: p(780, 420), controlPoint2: p(650, 690))
leaf.close()
NSGraphicsContext.saveGraphicsState()
let glow = NSShadow()
glow.shadowColor = color(0x86EFAC, 0.55)
glow.shadowBlurRadius = 60
glow.shadowOffset = .zero
glow.set()
color(0x4ADE80).setFill()
leaf.fill()
NSGraphicsContext.restoreGraphicsState()
NSGraphicsContext.saveGraphicsState()
leaf.addClip()
NSGradient(colors: [color(0xD9F99D), color(0x84CC16), color(0x22C55E), color(0x15803D)],
    atLocations: [0, 0.35, 0.7, 1], colorSpace: .sRGB)!
    .draw(from: tip, to: base, options: [])
// Darker left half (the leaf folds along its vein).
let half = NSBezierPath()
half.move(to: base)
half.curve(to: tip, controlPoint1: p(250, 610), controlPoint2: p(360, 300))
half.curve(to: base, controlPoint1: p(560, 380), controlPoint2: p(500, 600))
half.close()
color(0x052E16, 0.28).setFill()
half.fill()
// Warm forge light on the lower leaf.
NSGradient(colors: [color(0xFBBF24, 0.45), color(0xFBBF24, 0)], atLocations: [0, 1], colorSpace: .sRGB)!
    .draw(fromCenter: base, radius: 0, toCenter: base, radius: 300, options: [])
NSGraphicsContext.restoreGraphicsState()

// Vein + side veins.
let vein = NSBezierPath()
vein.move(to: base)
vein.curve(to: p(640, 275), controlPoint1: p(500, 600), controlPoint2: p(560, 380))
vein.lineWidth = 10
vein.lineCapStyle = .round
color(0xF7FEE7, 0.85).setStroke()
vein.stroke()
// Side veins branch upward from points on the midrib (the same cubic), alternating sides, rotated ±40°
// from its tangent and shorter toward the tip.
let vein0 = (470.0, 760.0), vein1 = (500.0, 600.0), vein2 = (560.0, 380.0), vein3 = (640.0, 275.0)
func cubic(_ t: Double, _ a: Double, _ b: Double, _ c: Double, _ d: Double) -> Double {
    let u = 1 - t
    return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d
}
func cubicTangent(_ t: Double, _ a: Double, _ b: Double, _ c: Double, _ d: Double) -> Double {
    let u = 1 - t
    return 3 * u * u * (b - a) + 6 * u * t * (c - b) + 3 * t * t * (d - c)
}
for (i, t) in [0.28, 0.40, 0.52, 0.64, 0.76].enumerated() {
    let sx = cubic(t, vein0.0, vein1.0, vein2.0, vein3.0), sy = cubic(t, vein0.1, vein1.1, vein2.1, vein3.1)
    var tx = cubicTangent(t, vein0.0, vein1.0, vein2.0, vein3.0), ty = cubicTangent(t, vein0.1, vein1.1, vein2.1, vein3.1)
    let n = (tx * tx + ty * ty).squareRoot()
    (tx, ty) = (tx / n, ty / n)
    let side = i % 2 == 0 ? -1.0 : 1.0  // −1 = left (wider) half
    let ang = side * 42 * Double.pi / 180
    let (dx, dy) = (tx * cos(ang) - ty * sin(ang), tx * sin(ang) + ty * cos(ang))
    let l = (side < 0 ? 175 : 125) * (1.12 - t)
    let b = NSBezierPath()
    b.move(to: p(sx, sy))
    // A gentle arc: start along the branch direction, bend slightly toward the tip.
    b.curve(to: p(sx + dx * l + tx * l * 0.15, sy + dy * l + ty * l * 0.15),
        controlPoint1: p(sx + dx * l * 0.45, sy + dy * l * 0.45),
        controlPoint2: p(sx + dx * l * 0.85 + tx * l * 0.05, sy + dy * l * 0.85 + ty * l * 0.05))
    b.lineWidth = 7 - t * 3.5
    b.lineCapStyle = .round
    color(0xF7FEE7, 0.40).setStroke()
    b.stroke()
}

// MARK: forge spark at the leaf's base — a gold four-point star with a hot core
func star(_ c: NSPoint, _ r: Double, _ waist: Double) -> NSBezierPath {
    let s = NSBezierPath()
    for i in 0..<8 {
        let a = Double(i) * .pi / 4 + .pi / 2
        let rr = i % 2 == 0 ? r : waist
        let pt = NSPoint(x: c.x + rr * cos(a), y: c.y + rr * sin(a))
        i == 0 ? s.move(to: pt) : s.line(to: pt)
    }
    s.close()
    return s
}
let sparkCenter = p(470, 772)
NSGradient(colors: [color(0xFDE68A, 0.9), color(0xF59E0B, 0.35), color(0xF59E0B, 0)], atLocations: [0, 0.35, 1],
    colorSpace: .sRGB)!
    .draw(fromCenter: sparkCenter, radius: 0, toCenter: sparkCenter, radius: 170, options: [])
NSGraphicsContext.saveGraphicsState()
let sg = NSShadow()
sg.shadowColor = color(0xFBBF24, 0.9)
sg.shadowBlurRadius = 30
sg.shadowOffset = .zero
sg.set()
let spark = star(sparkCenter, 118, 22)
color(0xFCD34D).setFill()
spark.fill()
NSGraphicsContext.restoreGraphicsState()
NSGraphicsContext.saveGraphicsState()
spark.addClip()
NSGradient(colors: [color(0xFFFBEB), color(0xFCD34D), color(0xF59E0B)], atLocations: [0, 0.3, 1], colorSpace: .sRGB)!
    .draw(fromCenter: sparkCenter, radius: 0, toCenter: sparkCenter, radius: 118, options: [])
NSGraphicsContext.restoreGraphicsState()
color(0xFFFFFF, 0.95).setFill()
NSBezierPath(ovalIn: NSRect(x: sparkCenter.x - 16, y: sparkCenter.y - 16, width: 32, height: 32)).fill()

// MARK: drifting sparks
for (x, y, r, a) in [(330.0, 690.0, 9.0, 0.9), (300, 600, 6, 0.7), (640, 700, 7, 0.8), (700, 610, 5, 0.6),
    (360, 520, 4, 0.5), (735, 520, 4, 0.45)] {
    let c = p(x, y)
    NSGradient(colors: [color(0xFDE68A, a), color(0xF59E0B, 0)], atLocations: [0, 1], colorSpace: .sRGB)!
        .draw(fromCenter: c, radius: 0, toCenter: c, radius: r * 3.2, options: [])
    color(0xFFFBEB, a).setFill()
    NSBezierPath(ovalIn: NSRect(x: c.x - r / 2, y: c.y - r / 2, width: r, height: r)).fill()
}
NSGraphicsContext.restoreGraphicsState()  // squircle clip
_ = ctx

NSGraphicsContext.restoreGraphicsState()
try! FileManager.default.createDirectory(
    at: URL(fileURLWithPath: out).deletingLastPathComponent(), withIntermediateDirectories: true)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
