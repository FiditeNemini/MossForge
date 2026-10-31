/// Exact decoder for the **E4M3FN** 8-bit float format — the encoding ComfyUI and
/// PyTorch (`torch.float8_e4m3fn`) use for fp8-scaled checkpoints. 1 sign bit, 4
/// exponent bits (bias 7), 3 mantissa bits; no infinities; the saturated code
/// `S.1111.111` is the only NaN, which lets the format reach ±448.
///
/// Decoding is a 256-entry lookup, so a whole tensor dequantizes with one map. The
/// values come straight from the format definition (powers of two built by shift,
/// no `pow`), so they are bit-exact. This is the read side of consuming community
/// fp8 weights that have no bf16 original (dequant → merge → re-export).
public enum Float8E4M3 {
    /// All 256 decoded values, indexed by the raw byte. NaN codes hold `Float.nan`.
    public static let table: [Float] = (0..<256).map { decode(UInt8($0)) }

    /// Decode one E4M3FN byte to its `Float` value.
    public static func decode(_ b: UInt8) -> Float {
        let sign: Float = (b & 0x80) != 0 ? -1 : 1
        let exp = Int((b >> 3) & 0x0F)
        let mant = Int(b & 0x07)

        if exp == 0 {
            // Subnormal (and zero): (mant / 8) · 2^(1 - bias) = (mant / 8) · 2^-6.
            return sign * (Float(mant) / 8.0) * pow2(-6)
        }
        if exp == 0x0F && mant == 0x07 {
            return .nan  // S.1111.111 — the sole NaN in E4M3FN.
        }
        // Normal: (1 + mant / 8) · 2^(exp - bias).
        return sign * (1.0 + Float(mant) / 8.0) * pow2(exp - 7)
    }

    /// Exact power of two for the small exponent range E4M3 needs (-6 … +8).
    private static func pow2(_ e: Int) -> Float {
        e >= 0 ? Float(1 << e) : 1.0 / Float(1 << (-e))
    }

    /// Encode a `Float` to the nearest E4M3FN byte (round-to-nearest-even) — the inverse
    /// of `decode`, the write side for fp8 export. Saturates to ±448; `NaN` → the sole
    /// NaN code. `encode(decode(b)) == b` for every finite code (verified exhaustively).
    public static func encode(_ x: Float) -> UInt8 {
        if x.isNaN { return 0x7F }
        let sign: UInt8 = (x.sign == .minus) ? 0x80 : 0
        let a = abs(x)
        if a == 0 { return sign }
        if a >= 448 { return sign | 0x7E }  // max finite = S.1111.110

        // Subnormal range: value = mant · 2^-9, mant ∈ 1…7 (exp field 0).
        if a < pow2(-6) {
            let mant = roundHalfEven(Double(a) / Double(pow2(-9)))
            if mant <= 0 { return sign }
            if mant >= 8 { return sign | 0x08 }  // rounded up into the smallest normal
            return sign | UInt8(mant)
        }

        // Normal: value = (1 + mant/8) · 2^(exp-7), exp field 1…15, mant 0…7.
        // floor(log2(a)) for a normal f32 is just its unbiased exponent (no Foundation).
        var exp = Int((a.bitPattern >> 23) & 0xFF) - 127
        var mant = roundHalfEven((Double(a) / Double(pow2(exp)) - 1.0) * 8.0)
        if mant == 8 { mant = 0; exp += 1 }  // rounding pushed the mantissa over
        let field = exp + 7
        if field >= 15 && mant >= 7 { return sign | 0x7E }  // would hit the NaN code → saturate
        return sign | UInt8((field << 3) | mant)
    }

    /// Round to nearest, ties to even (banker's rounding).
    private static func roundHalfEven(_ v: Double) -> Int {
        let r = v.rounded(.toNearestOrEven)
        return Int(r)
    }
}
