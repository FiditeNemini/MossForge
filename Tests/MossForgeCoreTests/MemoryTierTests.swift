import Foundation
import Testing

@testable import MossForgeCore

@Suite struct MemoryTierTests {
    let gib: UInt64 = 1 << 30

    @Test func sixtyFourGBAndUpIsRecommended() {
        // A 64 GB Mac reports exactly 64 GiB of physical memory.
        #expect(MossMemoryTier(physicalMemory: 64 * gib) == .recommended)
        #expect(MossMemoryTier(physicalMemory: 192 * gib) == .recommended)
        #expect(MossMemoryTier(physicalMemory: 64 * gib).notice == nil)
        #expect(MossMemoryTier(physicalMemory: 64 * gib).defaultResidencyMode == .auto)
    }

    @Test func thirtyTwoToSixtyFourGBIsLimitedAndSavesMemory() {
        for size: UInt64 in [32, 36, 48] {
            let tier = MossMemoryTier(physicalMemory: size * gib)
            #expect(tier == .limited)
            #expect(tier.defaultResidencyMode == .saveMemory)
            #expect(tier.notice?.contains("64 GB") == true)
            #expect(tier.notice?.contains("Save memory") == true)
        }
    }

    @Test func underThirtyTwoGBIsBelowTheMinimum() {
        for size: UInt64 in [8, 16, 24] {
            let tier = MossMemoryTier(physicalMemory: size * gib)
            #expect(tier == .belowMinimum)
            #expect(tier.defaultResidencyMode == .saveMemory)
            #expect(tier.notice?.contains("32 GB") == true)
        }
    }

    @Test func noticeNamesThisMacsMemory() {
        #expect(MossMemoryTier.notice(physicalMemory: 36 * gib)?.contains("36 GB") == true)
        #expect(MossMemoryTier.notice(physicalMemory: 16 * gib)?.contains("16 GB") == true)
        #expect(MossMemoryTier.notice(physicalMemory: 128 * gib) == nil)
    }
}
