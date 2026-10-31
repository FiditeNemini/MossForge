import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Exact decode/encode of the F8_E5M2 (`torch.float8_e5m2`) format — the second fp8 export
/// the community tooling offers next to E4M3. Unlike E4M3FN it is IEEE-like (has ±∞ and NaN),
/// so the largest finite value is 57344. Values are checked against the format definition.
@Suite struct Float8E5M2Tests {
    @Test func encodeRoundTripsEveryFiniteAndInfCode() {
        // Every representable byte decodes then re-encodes to itself (NaN codes excepted, since
        // the three NaN bit patterns all normalize to one canonical NaN code on encode).
        for b in 0...255 {
            let byte = UInt8(b)
            let v = Float8E5M2.decode(byte)
            if v.isNaN { continue }  // 0x7D / 0x7E / 0x7F (and 0xFD…) are NaN codes
            #expect(Float8E5M2.encode(v) == byte, "round-trip failed for 0x\(String(byte, radix: 16))")
        }
    }

    @Test func decodesCanonicalValues() {
        #expect(Float8E5M2.decode(0x00) == 0)  // +0
        #expect(Float8E5M2.decode(0x80) == 0)  // -0
        #expect(Float8E5M2.decode(0x38) == 0.5)  // exp=14 → 1·2^-1
        #expect(Float8E5M2.decode(0x3C) == 1.0)  // exp=15,mant=0 → 1·2^0
        #expect(Float8E5M2.decode(0x40) == 2.0)  // exp=16 → 1·2^1
        #expect(Float8E5M2.decode(0x42) == 3.0)  // exp=16,mant=2 → 1.5·2^1
        #expect(Float8E5M2.decode(0xBC) == -1.0)  // sign + 1.0
        #expect(Float8E5M2.decode(0x7B) == 57344)  // max finite: 1.75·2^15
    }

    @Test func decodesInfinityAndNaNAtSaturatedExponent() {
        #expect(Float8E5M2.decode(0x7C) == .infinity)  // S.11111.00 = +∞
        #expect(Float8E5M2.decode(0xFC) == -.infinity)  // -∞
        #expect(Float8E5M2.decode(0x7D).isNaN)
        #expect(Float8E5M2.decode(0x7F).isNaN)
    }

    @Test func decodesSubnormalsAndSmallestNormal() {
        #expect(Float8E5M2.decode(0x04) == 0.00006103515625)  // smallest normal 2^-14
        #expect(Float8E5M2.decode(0x01) == 0.0000152587890625)  // smallest subnormal (1/4)·2^-14
    }

    @Test func encodeRoundsSaturatesAndKeepsInfinity() {
        #expect(Float8E5M2.encode(0) == 0x00)
        #expect(Float8E5M2.encode(1.0) == 0x3C)
        #expect(Float8E5M2.encode(2.0) == 0x40)
        #expect(Float8E5M2.encode(-1.0) == 0xBC)
        #expect(Float8E5M2.encode(57344) == 0x7B)  // exactly max finite
        #expect(Float8E5M2.encode(100_000) == 0x7B)  // finite overflow saturates, not → ∞
        #expect(Float8E5M2.encode(-100_000) == 0xFB)
        #expect(Float8E5M2.encode(.infinity) == 0x7C)  // real ∞ is preserved
        // A value between two codes rounds to the nearer; decode is then within a step.
        let back = Float8E5M2.decode(Float8E5M2.encode(0.3))
        #expect(abs(back - 0.3) <= 0.06)
    }

    @Test func tableMatchesDecode() {
        #expect(Float8E5M2.table.count == 256)
        for b in 0..<256 {
            let t = Float8E5M2.table[b]
            let d = Float8E5M2.decode(UInt8(b))
            #expect(t == d || (t.isNaN && d.isNaN))
        }
    }
}
