import Foundation

/// Pure-logic dequantization of the on-disk quantized formats into plain `Float`
/// values, ready to be wrapped in an `MLXArray` by the MLX backend. Kept here (not
/// in the MLX package) so the decode math is unit-tested under `swift test`. The
/// `DequantOp` cases in `WeightLoadPlan` say *which* of these to call.
public enum QuantizedDecode {
    /// fp8-scaled: each E4M3 weight byte decoded and multiplied by the tensor's
    /// single F32 `weight_scale`. Result is row-major in the tensor's logical shape.
    /// Tight unsafe loop over the 256-entry LUT — runs over billions of params.
    public static func fp8Scaled(weightBytes: Data, scale: Float) -> [Float] {
        fp8Scaled(weightBytes: weightBytes, scale: scale, table: Float8E4M3.table)
    }

    /// E5M2 variant of `fp8Scaled` — same per-tensor scale, decoded through the E5M2 LUT.
    public static func fp8E5M2Scaled(weightBytes: Data, scale: Float) -> [Float] {
        fp8Scaled(weightBytes: weightBytes, scale: scale, table: Float8E5M2.table)
    }

    /// Decode fp8 weight bytes through an arbitrary 256-entry LUT (E4M3 or E5M2), multiplying
    /// by the per-tensor scale. Tight unsafe loop over the LUT — runs over billions of params.
    static func fp8Scaled(weightBytes: Data, scale: Float, table: [Float]) -> [Float] {
        let n = weightBytes.count
        var out = [Float](repeating: 0, count: n)
        table.withUnsafeBufferPointer { table in
            weightBytes.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                let bytes = src.bindMemory(to: UInt8.self)
                out.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<n { dst[i] = table[Int(bytes[i])] * scale }
                }
            }
        }
        return out
    }
}
