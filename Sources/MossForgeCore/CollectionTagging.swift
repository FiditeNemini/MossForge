
/// Pure set logic for tagging many Gallery images at once. Collections are tags, so a bulk
/// action is an explicit add or remove, never a per-image toggle (a toggle drops images that
/// were already in the collection).
public enum CollectionTagging {
    public enum Operation: Sendable { case add, remove }

    /// How much of a selection already carries a tag (drives the bulk menu).
    public enum Membership: Sendable, Equatable { case none, some, all }

    /// One image's tags after the operation. Idempotent either way.
    public static func apply(_ op: Operation, tag: String, to tags: Set<String>) -> Set<String> {
        var out = tags
        switch op {
        case .add: out.insert(tag)
        case .remove: out.remove(tag)
        }
        return out
    }

    /// Whether none, some or all of the selected images' tag sets contain `tag`.
    public static func membership(of tag: String, in selection: [Set<String>]) -> Membership {
        let count = selection.filter { $0.contains(tag) }.count
        if count == 0 { return .none }
        return count == selection.count ? .all : .some
    }
}
