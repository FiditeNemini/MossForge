import MossForgeCore
import Testing

/// Bulk collection tagging from the Gallery's multi-select. *Add* must be idempotent: the
/// old path toggled each image, so images already in the collection were removed.
@Suite struct CollectionTaggingTests {
    @Test func addIsIdempotent() {
        let tags: [Set<String>] = [["fox"], [], ["fox", "winter"]]
        let out = tags.map { CollectionTagging.apply(.add, tag: "fox", to: $0) }
        #expect(out == [["fox"], ["fox"], ["fox", "winter"]])
        // Applying again changes nothing.
        #expect(out.map { CollectionTagging.apply(.add, tag: "fox", to: $0) } == out)
    }

    @Test func removeOnlyStripsThatTag() {
        let out = CollectionTagging.apply(.remove, tag: "fox", to: ["fox", "winter"])
        #expect(out == ["winter"])
        #expect(CollectionTagging.apply(.remove, tag: "fox", to: ["winter"]) == ["winter"])
    }

    @Test func membershipAcrossSelection() {
        #expect(CollectionTagging.membership(of: "fox", in: []) == .none)
        #expect(CollectionTagging.membership(of: "fox", in: [["winter"], []]) == .none)
        #expect(CollectionTagging.membership(of: "fox", in: [["fox"], ["winter"]]) == .some)
        #expect(CollectionTagging.membership(of: "fox", in: [["fox"], ["fox", "winter"]]) == .all)
    }
}
