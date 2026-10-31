import AppKit
import MossForgeCore
import SwiftUI
import Tweakbench

/// Generate → Edit → Mask…: paint where Picture 1 may change. Outside the mask the result is pinned
/// to the original, step by step (Edit in Layout's pin), so the edit LoRA can't spill onto the rest.
/// Painted on the output's token grid (16 px cells); the paste-back feathers about one cell.
struct EditMaskSheet: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let source: URL
    @State private var mask: EditMask
    @State private var radius = 3.0
    @State private var erasing = false
    private let image: NSImage?

    init(source: URL, mask: EditMask) {
        self.source = source
        _mask = State(initialValue: mask)
        image = NSImage(contentsOf: source)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Mask the edit").font(.title2.weight(.semibold))
            Text("Paint where the image may change. Everything outside stays exactly as it is, whatever the LoRA does.")
                .font(.callout).foregroundStyle(.secondary)
            canvas.tweakable("editMask.canvas")
            HStack {
                Picker("", selection: $erasing) {
                    Label("Paint", systemImage: "paintbrush").tag(false)
                    Label("Erase", systemImage: "eraser").tag(true)
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 180)
                .tweakable("editMask.tool")
                Text("Brush").foregroundStyle(.secondary)
                Slider(value: $radius, in: 1...10).frame(width: 140).tweakable("editMask.brush")
                Spacer()
                Text(String(format: "%.0f%% changes", mask.coverage * 100)).font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            HStack {
                Button("Invert") { mask = mask.inverted() }.help("Paint what to protect, then invert")
                Button("Clear") { mask = .empty(width: mask.width, height: mask.height) }
                Spacer()
                Button("Remove Mask", role: .destructive) {
                    store.draft.editMask = nil
                    dismiss()
                }
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Use Mask") {
                    store.draft.editMask = mask.coverage > 0 ? .init(source: source, mask: mask) : nil
                    dismiss()
                }
                .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                .tweakable("editMask.done")
            }
        }
        .padding(20)
        .frame(width: 640)
    }

    private var aspect: CGFloat { CGFloat(mask.width) / CGFloat(max(mask.height, 1)) }

    private var canvas: some View {
        GeometryReader { geo in
            let size = geo.size
            ZStack {
                if let image { Image(nsImage: image).resizable().frame(width: size.width, height: size.height) }
                Canvas { ctx, s in
                    let (cw, ch) = (s.width / CGFloat(mask.width), s.height / CGFloat(mask.height))
                    for r in 0..<mask.height {
                        for c in 0..<mask.width {
                            let v = mask.values[r * mask.width + c]
                            guard v > 0 else { continue }
                            ctx.fill(Path(CGRect(x: CGFloat(c) * cw, y: CGFloat(r) * ch, width: cw + 0.5, height: ch + 0.5)),
                                     with: .color(.red.opacity(0.45 * Double(v))))
                        }
                    }
                }
                .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                mask.paint(
                    atX: g.location.x / size.width * Double(mask.width), y: g.location.y / size.height * Double(mask.height),
                    radius: radius, value: erasing ? 0 : 1)
            })
        }
        .aspectRatio(aspect, contentMode: .fit)
        .frame(maxHeight: 520)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
