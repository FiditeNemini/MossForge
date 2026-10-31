import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// What the trainer writes must load back through the same key map inference uses.
@Suite("MossLoRAOutput — trained LoRA files load in MossForge")
struct LoRAOutputTests {
    @Test func trainedKeysMapEveryPartWithAlphaOverRank() throws {
        let config = MossDiTConfig.release
        var descriptors: [TensorDescriptor] = []
        var alphas: [String: Float] = [:]
        let shapes = MossLoRAOutput.targetShapes(config: config, targets: .attentionAndFFN)
        #expect(shapes.count == 42 * 4)
        for (module, shape) in shapes {
            let k = MossLoRAOutput.keys(module: module)
            descriptors.append(TensorDescriptor(name: k.down, dtype: "F32", shape: [8, shape[1]], byteCount: 0))
            descriptors.append(TensorDescriptor(name: k.up, dtype: "F32", shape: [shape[0], 8], byteCount: 0))
            alphas[k.alpha] = 4
        }
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: alphas, config: config)
        #expect(plan.unmapped.isEmpty && plan.parts.count == 42 * 4)
        #expect(plan.parts.allSatisfy { $0.scale == 0.5 && $0.rows == nil })  // alpha / rank, whole fused matrices
        #expect(Set(plan.parts.map(\.target)).contains("layers.41.proj.ffn_down.weight"))
    }

    @Test func targetShapesFollowTheConfig() {
        let s = MossLoRAOutput.targetShapes(config: .release, targets: .attention)
        #expect(s["layers.0.proj.qkv"] == [3 * 3072, 3072] && s["layers.0.proj.o"] == [3072, 3072])
        #expect(s.count == 84)
    }
}
