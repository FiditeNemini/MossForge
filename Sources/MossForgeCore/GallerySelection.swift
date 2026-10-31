import Foundation

/// Finder-style selection over an ordered grid: click selects one, ⌘-click toggles, ⇧-click selects
/// the range from the anchor (the last plain or ⌘-click), ⌘A selects all.
public struct GallerySelection<ID: Hashable>: Equatable {
    public enum Modifier: Sendable { case none, command, shift }

    public private(set) var selected: Set<ID> = []
    /// The item the detail pane shows: the most recently clicked one still selected.
    public private(set) var primary: ID?
    private var anchor: ID?

    public init() {}

    public mutating func click(_ id: ID, in order: [ID], modifier: Modifier = .none) {
        switch modifier {
        case .none:
            selected = [id]
            anchor = id
            primary = id
        case .command:
            if selected.contains(id) {
                selected.remove(id)
                if primary == id { primary = order.last { selected.contains($0) } }
            } else {
                selected.insert(id)
                primary = id
            }
            anchor = id
        case .shift:
            guard let a = anchor, let from = order.firstIndex(of: a), let to = order.firstIndex(of: id) else {
                return click(id, in: order)
            }
            selected = Set(order[min(from, to)...max(from, to)])
            primary = id
        }
    }

    public mutating func selectAll(_ order: [ID]) {
        selected = Set(order)
        if primary == nil { primary = order.first }
        if anchor == nil { anchor = order.first }
    }

    public mutating func clear() {
        selected = []
        primary = nil
        anchor = nil
    }

    /// Drop items that no longer exist (deleted, or gone from the folder).
    public mutating func prune(keeping existing: Set<ID>) {
        selected.formIntersection(existing)
        if let p = primary, !existing.contains(p) { primary = nil }
        if let a = anchor, !existing.contains(a) { anchor = nil }
    }

    /// The selection in grid order.
    public func ordered(in order: [ID]) -> [ID] { order.filter(selected.contains) }
}
