/// Exact decoder/encoder for the **E5M2** 8-bit float format (`torch.float8_e5m2`) — the
/// second fp8 encoding the community tooling offers alongside E4M3. 1 sign bit, 5 exponent
/// bits (bias 15), 2 mantissa bits. Unlike E4M3FN it is IEEE-like: `S.11111.00` is ±∞ and
/// `S.11111.{01,10,11}` are NaN, so the largest finite value is `1.75·2¹⁵ = 57344`.
///
/// Decoding is a 256-entry lookup (built by shift, no `pow`, so bit-exact). This is the read
/// side of consuming community fp8-e5m2 weights and the reference the GPU encoder round-trips
/// against for fp8-e5m2 export.
public enum Float8E5M2 {
    /// All 256 decoded values, indexed by the raw byte. Inf codes hold `±Float.infinity`;
    /// NaN codes hold `Float.nan`.
    public static let table: [Float] = (0..<256).map { decode(UInt8($0)) }

    /// Decode one E5M2 byte to its `Float` value.
    public static func decode(_ b: UInt8) -> Float {
        let sign: Float = (b & 0x80) != 0 ? -1 : 1
        let exp = Int((b >> 2) & 0x1F)
        let mant = Int(b & 0x03)

        if exp == 0 {
            // Subnormal (and zero): (mant / 4) · 2^(1 - bias) = (mant / 4) · 2^-14.
            return sign * (Float(mant) / 4.0) * pow2(-14)
        }
        if exp == 0x1F {
            return mant == 0 ? sign * .infinity : .nan  // S.11111.00 = ±∞, else NaN.
        }
        // Normal: (1 + mant / 4) · 2^(exp - bias).
        return sign * (1.0 + Float(mant) / 4.0) * pow2(exp - 15)
    }

    /// Exact power of two for the exponent range E5M2 needs (-16 … +15).
    private static func pow2(_ e: Int) -> Float {
        e >= 0 ? Float(1 << e) : 1.0 / Float(1 << (-e))
    }

    /// Encode a `Float` to the nearest E5M2 byte (round-to-nearest-even) — the inverse of
    /// `decode`, the write side for fp8-e5m2 export. `NaN` → a NaN code; `±∞` → the ∞ code;
    /// finite magnitudes above the largest representable value saturate to `±57344` rather than
    /// overflowing to ∞ (weight tensors should never carry ∞). `encode(decode(b)) == b` for
    /// every finite and ∞ code (verified exhaustively).
    public static func encode(_ x: Float) -> UInt8 {
        if x.isNaN { return 0x7F }  // S.11111.11 — a canonical NaN.
        let sign: UInt8 = (x.sign == .minus) ? 0x80 : 0
        let a = abs(x)
        if a.isInfinite { return sign | 0x7C }  // S.11111.00 = ±∞.
        if a == 0 { return sign }

        // Subnormal range: value = mant · 2^-16, mant ∈ 1…3 (exp field 0).
        if a < pow2(-14) {
            let mant = roundHalfEven(Double(a) / Double(pow2(-16)))
            if mant <= 0 { return sign }
            if mant >= 4 { return sign | 0x04 }  // rounded up into the smallest normal
            return sign | UInt8(mant)
        }

        // Normal: value = (1 + mant/4) · 2^(exp-15), exp field 1…30, mant 0…3.
        // floor(log2(a)) for a normal f32 is just its unbiased exponent (no Foundation).
        var exp = Int((a.bitPattern >> 23) & 0xFF) - 127
        var mant = roundHalfEven((Double(a) / Double(pow2(exp)) - 1.0) * 4.0)
        if mant == 4 { mant = 0; exp += 1 }  // rounding pushed the mantissa over
        let field = exp + 15
        if field >= 31 { return sign | 0x7B }  // beyond max finite → saturate to ±57344
        return sign | UInt8((field << 2) | mant)
    }

    /// Round to nearest, ties to even (banker's rounding).
    private static func roundHalfEven(_ v: Double) -> Int {
        Int(v.rounded(.toNearestOrEven))
    }
}
