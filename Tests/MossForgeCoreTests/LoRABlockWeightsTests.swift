import Foundation
import Testing

@testable import MossForgeCore

/// Per-block LoRA strength — e.g. Kijai's SFT banding fix: the later blocks of an SFT-minus-Base LoRA
/// at −0.5 (`file@-0.5:21-41`).
@Suite("LoRABlockWeights — per-block LoRA strength spec")
struct LoRABlockWeightsTests {
    @Test func bareRangesSelectOnlyThoseBlocks() throws {
        let w = try #require(LoRABlockWeights(parsing: "21-41"))
        #expect(w.weight(block: 21) == 1 && w.weight(block: 41) == 1)
        #expect(w.weight(block: 20) == 0 && w.weight(block: nil) == 0)  // nil = non-block module
        let two = try #require(LoRABlockWeights(parsing: "3,4,21-41"))
        #expect(two.weight(block: 4) == 1 && two.weight(block: 5) == 0)
    }

    @Test func explicitWeightsLeaveTheRestAtOneUnlessOverridden() throws {
        let w = try #require(LoRABlockWeights(parsing: "0-20=0, 21-41=0.5"))
        #expect(w.weight(block: 3) == 0 && w.weight(block: 30) == 0.5 && w.weight(block: nil) == 1)
        let x = try #require(LoRABlockWeights(parsing: "30-41=1,*=0,other=0.25"))
        #expect(x.weight(block: 29) == 0 && x.weight(block: 30) == 1 && x.weight(block: nil) == 0.25)
    }

    @Test func rejectsGarbage() {
        #expect(LoRABlockWeights(parsing: "") == nil)
        #expect(LoRABlockWeights(parsing: "41-21") == nil)
        #expect(LoRABlockWeights(parsing: "a-b") == nil)
        #expect(LoRABlockWeights(parsing: "3=x") == nil)
    }

    @Test func specCombinesGlobalStrengthWithTheBlockWeight() throws {
        let spec = LoRASpec(url: URL(fileURLWithPath: "/k.safetensors"), strength: -0.5, blocks: LoRABlockWeights(parsing: "21-41"))
        #expect(spec.strength(forBlock: 30) == -0.5 && spec.strength(forBlock: 2) == 0 && spec.strength(forBlock: nil) == 0)
        #expect(LoRASpec(url: URL(fileURLWithPath: "/a"), strength: 0.7).strength(forBlock: 12) == 0.7)
    }

    @Test func cliSyntaxParses() throws {
        let spec = try #require(LoRASpec(parsing: "/m/kijai.safetensors@-0.5:21-41"))
        #expect(spec.url.path == "/m/kijai.safetensors" && spec.strength == -0.5)
        #expect(spec.blocks == LoRABlockWeights(parsing: "21-41"))
        #expect(LoRASpec(parsing: "/m/a.safetensors")?.strength == 1)
        #expect(LoRASpec(parsing: "/m/a.safetensors@0.7")?.blocks == nil)
        #expect(LoRASpec(parsing: "/m/a.safetensors@zz") == nil)
    }
}
