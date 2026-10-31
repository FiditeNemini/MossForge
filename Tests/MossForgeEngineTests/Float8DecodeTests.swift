import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Exact decode of the F8_E4M3 (E4M3FN) format the ComfyUI / torch `float8_e4m3fn`
/// checkpoints store — the read side of consuming community fp8 weights that ship
/// with no bf16 original (so they can be dequantized → merged → re-exported). Pure
/// logic; the values are checked against the format definition, not guessed.
@Suite struct Float8E4M3Tests {
    @Test func encodeRoundTripsEveryFiniteCode() {
        // The strongest check: every representable byte decodes then re-encodes to itself.
        for b in 0...255 {
            let byte = UInt8(b)
            let v = Float8E4M3.decode(byte)
            if v.isNaN { continue }  // 0x7F / 0xFF are the NaN codes
            #expect(Float8E4M3.encode(v) == byte, "round-trip failed for 0x\(String(byte, radix: 16))")
        }
    }

    @Test func encodeRoundsAndSaturates() {
        #expect(Float8E4M3.encode(0) == 0x00)
        #expect(Float8E4M3.encode(1.0) == 0x38)
        #expect(Float8E4M3.encode(2.0) == 0x40)
        #expect(Float8E4M3.encode(-1.0) == 0xB8)
        #expect(Float8E4M3.encode(1000) == 0x7E)  // saturate to +448
        #expect(Float8E4M3.encode(-1000) == 0xFE)  // saturate to -448
        // A value between two codes rounds to the nearer; decode is then within a step.
        let back = Float8E4M3.decode(Float8E4M3.encode(0.3))
        #expect(abs(back - 0.3) <= 0.03)
    }

    @Test func decodesCanonicalValues() {
        #expect(Float8E4M3.decode(0x00) == 0)  // +0
        #expect(Float8E4M3.decode(0x80) == 0)  // -0
        #expect(Float8E4M3.decode(0x38) == 1.0)  // exp=7,mant=0 → 1·2^0
        #expect(Float8E4M3.decode(0x3C) == 1.5)  // exp=7,mant=4 → 1.5·2^0
        #expect(Float8E4M3.decode(0x40) == 2.0)  // exp=8 → 1·2^1
        #expect(Float8E4M3.decode(0xB8) == -1.0)  // sign + 1.0
        #expect(Float8E4M3.decode(0x7E) == 448)  // max normal: 1.75·2^8
    }

    @Test func decodesSubnormalsAndSmallestNormal() {
        #expect(Float8E4M3.decode(0x08) == 0.015625)  // smallest normal 2^-6
        #expect(Float8E4M3.decode(0x01) == 0.001953125)  // smallest subnormal (1/8)·2^-6
    }

    @Test func encodesNaNAtSaturatedCode() {
        #expect(Float8E4M3.decode(0x7F).isNaN)  // S.1111.111 = NaN (E4M3FN)
        #expect(Float8E4M3.decode(0xFF).isNaN)
    }

    @Test func tableMatchesDecode() {
        #expect(Float8E4M3.table.count == 256)
        for b in 0..<256 {
            let t = Float8E4M3.table[b]
            let d = Float8E4M3.decode(UInt8(b))
            #expect(t == d || (t.isNaN && d.isNaN))
        }
    }
}

@Suite struct QuantizedDecodeTests {
    @Test func fp8ScaledAppliesPerTensorScale() {
        let bytes = Data([0x38, 0x40, 0xB8])  // 1.0, 2.0, -1.0
        let out = QuantizedDecode.fp8Scaled(weightBytes: bytes, scale: 2.0)
        #expect(out == [2.0, 4.0, -2.0])
    }

    @Test func fp8ScaledEmptyIsEmpty() {
        #expect(QuantizedDecode.fp8Scaled(weightBytes: Data(), scale: 1.0).isEmpty)
    }
}

/// Raw per-tensor byte reader — pulls one tensor's bytes by header offset, so fp8
/// tensors can be decoded without `MLX.loadArrays` (which can't open an fp8 file:
/// mlx-swift has no fp8 DType).
@Suite struct SafetensorsRawReadTests {
    /// Minimal writer: one tensor with explicit raw bytes.
    private func writeOne(name: String, dtype: String, shape: [Int], bytes: [UInt8]) throws -> URL {
        let header: [String: Any] = [
            name: ["dtype": dtype, "shape": shape, "data_offsets": [0, bytes.count]]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var out = Data()
        var len = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(contentsOf: bytes)
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kf-raw-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    @Test func readsExactTensorBytes() throws {
        let payload: [UInt8] = [0x38, 0x40, 0xB8, 0x00, 0x7E, 0x7F]
        let url = try writeOne(name: "w", dtype: "F8_E4M3", shape: [6], bytes: payload)
        defer { try? FileManager.default.removeItem(at: url) }
        let (descriptor, bytes) = try SafetensorsHeader.readTensor(name: "w", from: url)
        #expect(descriptor.dtype == "F8_E4M3")
        #expect(descriptor.shape == [6])
        #expect([UInt8](bytes) == payload)
    }

    @Test func readsScalarF32Scale() throws {
        var f: Float = 0.05
        let bytes = withUnsafeBytes(of: &f) { Array($0) }
        let url = try writeOne(name: "w.weight_scale", dtype: "F32", shape: [], bytes: bytes)
        defer { try? FileManager.default.removeItem(at: url) }
        let (_, raw) = try SafetensorsHeader.readTensor(name: "w.weight_scale", from: url)
        let value = raw.withUnsafeBytes { $0.load(as: Float.self) }
        #expect(value == 0.05)
    }

    @Test func missingTensorThrows() throws {
        let url = try writeOne(name: "w", dtype: "F32", shape: [1], bytes: [0, 0, 0, 0])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            _ = try SafetensorsHeader.readTensor(name: "nope", from: url)
        }
    }
}
