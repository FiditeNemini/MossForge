import Foundation
import Testing

@testable import MossForgeCore

@Suite struct ResidencyPolicyTests {
    let gib = 1 << 30

    @Test func autoOnA64GBMacKeepsTodaysRules() {
        // A 64 GB Mac reports ~48 GB recommended working set.
        let r = MossResidencyPolicy.resolve(.auto, recommendedWorkingSet: 48 * gib)
        #expect(r == .saveMemory)
        #expect(!r.keepEncoderResident)
        #expect(!r.modelsCoexist)
    }

    @Test func autoOnA96GBMacKeepsTheEncoderOnly() {
        let r = MossResidencyPolicy.resolve(.auto, recommendedWorkingSet: 72 * gib)
        #expect(r.keepEncoderResident)
        #expect(!r.modelsCoexist)
    }

    @Test func autoOnABigMacKeepsEverything() {
        // An M2 Ultra (192 GB) reports 161.3 GB.
        let r = MossResidencyPolicy.resolve(.auto, recommendedWorkingSet: Int(161.3 * Double(gib)))
        #expect(r == .keepLoaded)
        #expect(r.keepEncoderResident)
        #expect(r.modelsCoexist)
    }

    @Test func autoOnA128GBMacKeepsEverything() {
        // With one shared Qwen3.5, everything is ~36 GB + activations; a 128 GB Mac reports ~96–107 GB.
        let r = MossResidencyPolicy.resolve(.auto, recommendedWorkingSet: 96 * gib)
        #expect(r == .keepLoaded)
    }

    @Test func explicitChoicesIgnoreTheMachine() {
        #expect(MossResidencyPolicy.resolve(.keepLoaded, recommendedWorkingSet: 16 * gib) == .keepLoaded)
        #expect(MossResidencyPolicy.resolve(.saveMemory, recommendedWorkingSet: 512 * gib) == .saveMemory)
    }

    @Test func thresholdsAreInclusive() {
        let p = MossResidencyPolicy.self
        #expect(p.resolve(.auto, recommendedWorkingSet: p.encoderThreshold).keepEncoderResident)
        #expect(!p.resolve(.auto, recommendedWorkingSet: p.encoderThreshold - 1).keepEncoderResident)
        #expect(p.resolve(.auto, recommendedWorkingSet: p.coexistThreshold).modelsCoexist)
        #expect(!p.resolve(.auto, recommendedWorkingSet: p.coexistThreshold - 1).modelsCoexist)
    }

    @Test func unknownWorkingSetSavesMemory() {
        #expect(MossResidencyPolicy.resolve(.auto, recommendedWorkingSet: 0) == .saveMemory)
    }

    @Test func modeCodableAndTitles() throws {
        for m in MossResidencyMode.allCases {
            #expect(try JSONDecoder().decode(MossResidencyMode.self, from: JSONEncoder().encode(m)) == m)
            #expect(!m.title.isEmpty)
        }
        #expect(MossResidencyMode.allCases.first == .auto)
    }

    @Test func summaryNamesWhatStaysLoaded() {
        #expect(MossResidencyPolicy.keepLoaded.summary.contains("text encoder"))
        #expect(MossResidencyPolicy.saveMemory.summary.contains("at a time"))
    }
}
