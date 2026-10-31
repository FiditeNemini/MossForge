import AppKit
import MossForgeCore
import SwiftUI
import Tweakbench

// Gestures: drag on empty space to draw, drag a box to move, drag the selected box's
// bottom-right handle to resize, tap to select — over MossForge's `LayoutDraft` (normalised frames →
// abs_centralized yxyx pixels). Geometry/clamping lives in `LayoutDraft` (unit-tested).

/// The interactive layout canvas, at the image's aspect ratio.
struct LayoutCanvas: View {
    @Environment(\.theme) private var theme
    @Binding var draft: LayoutDraft
    var aspectRatio: Double
    /// Label for boxes drawn on empty space.
    var newLabel: LayoutPrompt.Label = .object
    /// Edit in Layout: the image being edited, drawn under the boxes so they can be placed over what should change.
    var background: NSImage?

    @State private var mode: Mode?
    @State private var pending: LayoutDraft.Frame?

    private let handle: CGFloat = 16
    private let minDraw = 0.025

    private enum Mode {
        case create(start: CGPoint)
        case move(id: UUID, last: CGPoint)
        case resize(id: UUID)
    }

    var body: some View {
        GeometryReader { geo in
            let canvas = fitted(in: geo.size)
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.background.opacity(0.5))
                    .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(.quaternary))
                    .overlay(alignment: .bottomTrailing) {
                        Text("\(draft.canvas.width)×\(draft.canvas.height)").font(.system(size: 9)).foregroundStyle(.tertiary).padding(4)
                    }
                    .frame(width: canvas.width, height: canvas.height)
                    .position(x: canvas.midX, y: canvas.midY)
                if let background {
                    Image(nsImage: background).resizable()
                        .frame(width: canvas.width, height: canvas.height)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .opacity(0.85)
                        .position(x: canvas.midX, y: canvas.midY)
                        .allowsHitTesting(false)
                }
                ForEach(draft.items) { item in box(item, canvas: canvas) }
                if let pending {
                    let r = rect(pending, canvas)
                    RoundedRectangle(cornerRadius: 3)
                        .strokeBorder(.secondary, style: StrokeStyle(lineWidth: 1.5, dash: [4]))
                        .frame(width: r.width, height: r.height)
                        .position(x: r.midX, y: r.midY)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .contentShape(Rectangle())
            .gesture(drag(canvas: canvas))
        }
    }

    private func box(_ item: LayoutDraft.Item, canvas: CGRect) -> some View {
        let r = rect(item.frame, canvas)
        let selected = draft.selection == item.id
        let color: Color = item.label == .text ? .blue : .orange
        let name = item.label == .text && !item.text.isEmpty ? item.text : item.caption
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 3)
                .fill(color.opacity(0.14))
                .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(
                    selected ? theme.accent : color,
                    style: StrokeStyle(lineWidth: selected ? 2 : 1.5, dash: item.label == .text ? [5] : [])))
            Text("\(item.label.rawValue) · \(String(name.replacingOccurrences(of: "\n", with: " ").prefix(22)))")
                .font(.system(size: 9)).lineLimit(1)
                .padding(.horizontal, 3).padding(.vertical, 1)
                .background(color.opacity(0.2)).padding(2)
            if selected {
                Rectangle().fill(theme.accent).frame(width: handle * 0.6, height: handle * 0.6).position(x: r.width, y: r.height)
            }
        }
        .frame(width: r.width, height: r.height, alignment: .topLeading)
        .position(x: r.midX, y: r.midY)
    }

    private func drag(canvas: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if mode == nil { mode = begin(at: value.startLocation, canvas: canvas) }
                let cur = fraction(value.location, canvas)
                switch mode {
                case .create(let start):
                    let s = fraction(start, canvas)
                    pending = LayoutDraft.Frame(top: s.y, left: s.x, bottom: cur.y, right: cur.x).normalised
                case .move(let id, let last):
                    let l = fraction(last, canvas)
                    draft.move(id, dx: cur.x - l.x, dy: cur.y - l.y)
                    mode = .move(id: id, last: value.location)
                case .resize(let id):
                    draft.resize(id, toBottom: cur.y, right: cur.x)
                case nil: break
                }
            }
            .onEnded { value in
                if case .create(let start)? = mode {
                    let s = fraction(start, canvas), e = fraction(value.location, canvas)
                    if abs(e.x - s.x) > minDraw && abs(e.y - s.y) > minDraw {
                        draft.add(label: newLabel, frame: .init(top: s.y, left: s.x, bottom: e.y, right: e.x))
                    } else {
                        draft.selection = nil
                    }
                }
                mode = nil
                pending = nil
            }
    }

    /// What a drag starting at `point` does; selects as a side effect.
    private func begin(at point: CGPoint, canvas: CGRect) -> Mode {
        if let id = draft.selection, let item = draft.item(id) {
            let r = rect(item.frame, canvas)
            if CGRect(x: r.maxX - handle / 2, y: r.maxY - handle / 2, width: handle, height: handle).contains(point) {
                return .resize(id: id)
            }
        }
        let f = fraction(point, canvas)
        if let id = draft.hit(x: f.x, y: f.y) {
            draft.selection = id
            return .move(id: id, last: point)
        }
        return .create(start: point)
    }

    private func fitted(in size: CGSize) -> CGRect {
        let pad: CGFloat = 4
        var w = size.width - pad * 2
        var h = w / aspectRatio
        if h > size.height - pad * 2 {
            h = size.height - pad * 2
            w = h * aspectRatio
        }
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func rect(_ f: LayoutDraft.Frame, _ c: CGRect) -> CGRect {
        CGRect(x: c.minX + f.left * c.width, y: c.minY + f.top * c.height, width: (f.right - f.left) * c.width, height: (f.bottom - f.top) * c.height)
    }

    private func fraction(_ p: CGPoint, _ c: CGRect) -> (x: Double, y: Double) {
        (Double((p.x - c.minX) / c.width), Double((p.y - c.minY) / c.height))
    }
}

/// Editor for the selected layout item (and the global caption).
struct LayoutInspector: View {
    @Binding var draft: LayoutDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Overall caption (scene, style)", text: $draft.caption, axis: .vertical).lineLimit(2...4)
                .tweakable("generate.layout.caption")
            if let id = draft.selection, draft.item(id) != nil {
                let item = Binding(
                    get: { draft.item(id)! },
                    set: { new in draft.update(id) { $0 = new } })
                HStack {
                    Picker("Kind", selection: item.label) {
                        ForEach(LayoutPrompt.Label.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }.frame(maxWidth: 180)
                    Spacer()
                    Button(role: .destructive) { draft.remove(id) } label: { Label("Delete box", systemImage: "trash") }
                }
                TextField("What is in this box", text: item.caption, axis: .vertical).lineLimit(1...3)
                    .tweakable("generate.layout.boxCaption")
                if item.wrappedValue.label == .text {
                    TextField("Literal text (use new lines for line breaks)", text: item.text, axis: .vertical).lineLimit(1...4)
                        .tweakable("generate.layout.boxText")
                }
            } else {
                Text("Drag on the canvas to draw a box; drag a box to move it, its corner to resize.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
