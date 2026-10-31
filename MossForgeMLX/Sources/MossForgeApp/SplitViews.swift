import AppKit
import SwiftUI

// One splitter for the whole app, behaving like Finder / Mail / Xcode: a hairline separator with a
// wider hit area, the resize cursor on hover, the position remembered per pane (UserDefaults) and
// restored when you come back, and double-click to reset. SwiftUI's HSplitView can't remember a
// position, so each screen's split felt different.

/// The draggable separator. `axis` is the direction it moves along.
struct SplitHandle: View {
    let axis: Axis
    let onDrag: (CGFloat) -> Void
    let onEnd: () -> Void
    let onReset: () -> Void
    @State private var hovering = false

    var body: some View {
        ZStack {
            Rectangle().fill(Color(nsColor: .separatorColor))
                .frame(width: axis == .horizontal ? 1 : nil, height: axis == .vertical ? 1 : nil)
        }
        .frame(width: axis == .horizontal ? 7 : nil, height: axis == .vertical ? 7 : nil)
        .frame(maxWidth: axis == .vertical ? .infinity : nil, maxHeight: axis == .horizontal ? .infinity : nil)
        .contentShape(Rectangle())
        .onHover { inside in
            guard inside != hovering else { return }
            hovering = inside
            if inside { (axis == .horizontal ? NSCursor.resizeLeftRight : NSCursor.resizeUpDown).push() } else { NSCursor.pop() }
        }
        .onDisappear { if hovering { NSCursor.pop() } }
        .gesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in onDrag(axis == .horizontal ? v.translation.width : v.translation.height) }
                .onEnded { _ in onEnd() }
        )
        .simultaneousGesture(TapGesture(count: 2).onEnded(onReset))
        .accessibilityElement()
        .accessibilityLabel("Splitter")
        .accessibilityHint("Drag to resize; double-click to reset")
    }
}

/// Two panes side by side; one keeps its remembered width, the other takes the rest.
struct PersistentHSplit<Leading: View, Trailing: View>: View {
    enum Sized { case leading, trailing }

    let sized: Sized
    let defaultWidth: CGFloat
    let range: ClosedRange<CGFloat>
    /// The other pane never gets narrower than this.
    let otherMin: CGFloat
    let leading: Leading
    let trailing: Trailing
    @AppStorage private var stored: Double
    @State private var dragStart: CGFloat?

    init(
        key: String, sized: Sized, defaultWidth: CGFloat, range: ClosedRange<CGFloat>, otherMin: CGFloat,
        @ViewBuilder leading: () -> Leading, @ViewBuilder trailing: () -> Trailing
    ) {
        self.sized = sized
        self.defaultWidth = defaultWidth
        self.range = range
        self.otherMin = otherMin
        self.leading = leading()
        self.trailing = trailing()
        _stored = AppStorage(wrappedValue: Double(defaultWidth), "split." + key)
    }

    var body: some View {
        GeometryReader { geo in
            let w = clamp(CGFloat(stored), total: geo.size.width)
            HStack(spacing: 0) {
                if sized == .leading {
                    leading.frame(width: w)
                    handle(w, total: geo.size.width)
                    trailing.frame(maxWidth: .infinity)
                } else {
                    leading.frame(maxWidth: .infinity)
                    handle(w, total: geo.size.width)
                    trailing.frame(width: w)
                }
            }
        }
    }

    private func handle(_ w: CGFloat, total: CGFloat) -> some View {
        SplitHandle(axis: .horizontal) { dx in
            let start = dragStart ?? w
            if dragStart == nil { dragStart = w }
            stored = Double(clamp(start + (sized == .leading ? dx : -dx), total: total))
        } onEnd: {
            dragStart = nil
        } onReset: {
            stored = Double(defaultWidth)
        }
    }

    private func clamp(_ w: CGFloat, total: CGFloat) -> CGFloat {
        min(max(w, range.lowerBound), max(range.lowerBound, min(range.upperBound, total - otherMin - 7)))
    }
}

/// Two panes stacked; the top keeps its remembered height, the bottom takes the rest.
struct PersistentVSplit<Top: View, Bottom: View>: View {
    let defaultHeight: CGFloat
    let minTop: CGFloat
    let minBottom: CGFloat
    let top: Top
    let bottom: Bottom
    @AppStorage private var stored: Double
    @State private var dragStart: CGFloat?

    init(
        key: String, defaultHeight: CGFloat, minTop: CGFloat, minBottom: CGFloat,
        @ViewBuilder top: () -> Top, @ViewBuilder bottom: () -> Bottom
    ) {
        self.defaultHeight = defaultHeight
        self.minTop = minTop
        self.minBottom = minBottom
        self.top = top()
        self.bottom = bottom()
        _stored = AppStorage(wrappedValue: Double(defaultHeight), "split." + key)
    }

    var body: some View {
        GeometryReader { geo in
            let h = clamp(CGFloat(stored), total: geo.size.height)
            VStack(spacing: 0) {
                top.frame(height: h)
                SplitHandle(axis: .vertical) { dy in
                    let start = dragStart ?? h
                    if dragStart == nil { dragStart = h }
                    stored = Double(clamp(start + dy, total: geo.size.height))
                } onEnd: {
                    dragStart = nil
                } onReset: {
                    stored = Double(defaultHeight)
                }
                bottom.frame(maxHeight: .infinity)
            }
        }
    }

    private func clamp(_ h: CGFloat, total: CGFloat) -> CGFloat {
        min(max(h, minTop), max(minTop, total - minBottom - 7))
    }
}

/// A view whose height is set by a grip under it (remembered), for editors inside a scrolling form.
struct ResizableHeight: ViewModifier {
    let range: ClosedRange<CGFloat>
    let defaultHeight: CGFloat
    @AppStorage private var stored: Double
    @State private var dragStart: CGFloat?

    init(key: String, defaultHeight: CGFloat, range: ClosedRange<CGFloat>) {
        self.range = range
        self.defaultHeight = defaultHeight
        _stored = AppStorage(wrappedValue: Double(defaultHeight), "height." + key)
    }

    func body(content: Content) -> some View {
        let h = min(max(CGFloat(stored), range.lowerBound), range.upperBound)
        VStack(spacing: 0) {
            content.frame(height: h)
            SplitHandle(axis: .vertical) { dy in
                let start = dragStart ?? h
                if dragStart == nil { dragStart = h }
                stored = Double(min(max(start + dy, range.lowerBound), range.upperBound))
            } onEnd: {
                dragStart = nil
            } onReset: {
                stored = Double(defaultHeight)
            }
            .overlay(Capsule().fill(Color.secondary.opacity(0.5)).frame(width: 28, height: 3))  // grip
        }
    }
}

extension View {
    /// Let the user drag this view's height (remembered under `key`); double-click the grip resets it.
    func resizableHeight(_ key: String, default h: CGFloat, range: ClosedRange<CGFloat>) -> some View {
        modifier(ResizableHeight(key: key, defaultHeight: h, range: range))
    }
}
