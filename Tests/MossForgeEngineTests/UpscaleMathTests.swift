import Foundation
import Testing

@testable import MossForgeEngine

/// Pure image math for the VOSR upscaler, checked against PIL / torch / OpenCV outputs
/// (vectors generated with Forge's env — see Scripts/vosr_ref.py for the full-model fixtures).
@Suite struct UpscaleMathTests {
    // 6×5 RGB, row-major HWC.
    static let img: [UInt8] = [
        139, 74, 229, 241, 169, 65, 6, 160, 149, 106, 38, 175, 188, 205, 175, 229, 98, 249, 10, 148, 95, 86, 147,
        198, 66, 39, 106, 213, 171, 45, 167, 57, 69, 81, 55, 14, 153, 178, 215, 76, 52, 39, 250, 72, 215, 50, 161,
        223, 112, 172, 161, 233, 38, 17, 89, 1, 221, 6, 242, 127, 143, 6, 60, 210, 94, 25, 166, 33, 249, 189, 12,
        204, 33, 57, 124, 30, 199, 149, 202, 119, 72, 220, 3, 209, 122, 136, 147, 77, 133, 82,
    ]

    @Test func pilBicubicUpscaleIsExact() {
        let expected: [UInt8] = [
            139, 63, 249, 210, 117, 147, 225, 178, 57, 28, 176, 136, 30, 87, 173, 115, 45, 184, 177, 197, 177, 221,
            164, 224, 241, 93, 255, 51, 110, 146, 111, 141, 150, 143, 157, 147, 35, 98, 123, 94, 97, 101, 186, 115,
            91, 184, 117, 107, 157, 93, 115, 133, 73, 118, 35, 168, 111, 52, 148, 141, 81, 103, 167, 110, 40, 141,
            152, 111, 107, 177, 176, 82, 157, 93, 86, 127, 56, 33, 105, 47, 0, 158, 186, 227, 107, 118, 122, 95, 46,
            50, 240, 63, 191, 157, 117, 234, 44, 169, 218, 95, 179, 177, 180, 102, 81, 241, 29, 8, 118, 43, 239, 61,
            124, 169, 34, 183, 92, 152, 32, 100, 178, 58, 71, 163, 117, 84, 153, 73, 220, 185, 31, 211, 216, 6, 163,
            63, 9, 181, 28, 129, 167, 26, 227, 130, 140, 76, 63, 208, 38, 61, 220, 52, 109, 161, 67, 199, 139, 67,
            192, 136, 64, 160, 29, 51, 116, 20, 129, 138, 52, 197, 142, 183, 143, 74, 230, 49, 147, 210, 7, 219,
            135, 123, 156, 87, 149, 101, 67, 140, 71,
        ]
        let out = PILResample.bicubic(Self.img, width: 6, height: 5, channels: 3, toWidth: 9, toHeight: 7)
        #expect(out == expected)
    }

    @Test func pilBicubicDownscaleIsExact() {
        let expected: [UInt8] = [
            123, 128, 153, 87, 125, 127, 148, 113, 123, 172, 105, 139, 84, 130, 154, 139, 72, 131, 132, 137, 150,
            172, 56, 90, 36, 103, 160, 129, 124, 89, 196, 53, 149, 128, 88, 162,
        ]
        #expect(PILResample.bicubic(Self.img, width: 6, height: 5, channels: 3, toWidth: 4, toHeight: 3) == expected)
    }

    @Test func pilSameSizeIsIdentity() {
        #expect(PILResample.bicubic(Self.img, width: 6, height: 5, channels: 3, toWidth: 6, toHeight: 5) == Self.img)
    }

    private func close(_ a: [Float], _ b: [Float], _ tol: Float = 2e-6) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tol }
    }

    @Test func torchBicubicMatrixBySize() {
        let m = TorchBicubic.matrix(inSize: 4, outSize: 7)
        let expected: [Float] = [
            1.0992168, -0.0992165, 0, 0, 0.6592567, 0.4022414, -0.0614982, 0, 0.057216, 0.9889759, -0.0461918, 0,
            -0.0937499, 0.5937498, 0.5937502, -0.0937501, 0, -0.046192, 0.9889759, 0.0572162, 0, -0.0614977,
            0.4022410, 0.6592565, 0, 0, -0.0992165, 1.0992165,
        ]
        #expect(close(m, expected))
    }

    @Test func torchBicubicMatrixByScaleFactorUsesTheGivenScale() {
        // DINOv2 hub interpolation: 37 -> 32 with scale_factor (32 + 0.1) / 37.
        let m = TorchBicubic.matrix(inSize: 37, outSize: 32, scaleFactor: 32.1 / 37)
        #expect(close(Array(m[0..<3]), [0.9386104, 0.0654252, -0.004036]))
        #expect(close(Array(m[(31 * 37 + 34)..<(32 * 37)]), [-0.0222554, 0.189961, 0.8322943]))
        let row100 = Array(TorchBicubic.matrix(inSize: 512, outSize: 448)[(100 * 512 + 113)..<(100 * 512 + 117)])
        #expect(close(row100, [-0.110696, 0.7699477, 0.4022471, -0.0614989]))
    }

    static let noisy: [UInt8] = [
        162, 230, 70, 71, 240, 62, 47, 184, 31, 34, 63, 65, 146, 197, 142, 253, 81, 133, 240, 113, 30, 247, 103,
        122, 31, 19, 42, 129, 141, 136, 33, 149, 161, 0, 178, 141, 107, 35, 107, 130, 72, 27, 217, 254, 99, 13,
        195, 206, 59, 228, 235, 202, 183, 47, 69, 179, 82, 35, 71, 159, 110, 121, 77, 87, 3, 126, 44, 253, 233,
        225, 90, 119, 137, 120, 30, 55, 33, 165, 87, 216, 229, 167, 3, 41, 109, 203, 129, 219, 63, 101, 207, 156,
        27, 220, 93, 29, 143, 200, 63, 11, 11, 180, 219, 113, 190, 87, 34, 9, 133, 29, 61, 36, 244, 192, 207, 131,
        254, 67, 99, 248, 47, 71, 89, 119, 137, 85,
    ]

    @Test func bilateralMatchesOpenCV() {
        // 7×6 noise: only pixels with a near-coloured neighbour move (σcolour 20 vs 12).
        var expected20 = Self.noisy
        expected20[48] = 58; expected20[49] = 230
        expected20[66] = 45; expected20[67] = 252; expected20[70] = 89
        let out20 = ImageFilters.bilateral(Self.noisy, width: 7, height: 6, sigmaColor: 20, sigmaSpace: 1.5)
        #expect(out20 == expected20)
        #expect(ImageFilters.bilateral(Self.noisy, width: 7, height: 6, sigmaColor: 12, sigmaSpace: 1.5) == Self.noisy)
    }

    @Test func gaussianMatchesOpenCVWithinOneLevel() {
        let expected: [UInt8] = [
            115, 149, 85, 96, 147, 88, 75, 146, 99, 71, 144, 113, 92, 131, 119, 132, 102, 104, 152, 87, 89, 113,
            148, 104, 97, 148, 108, 83, 151, 115, 80, 149, 117, 86, 133, 118, 112, 102, 108, 128, 88, 96, 107, 157,
            139, 101, 159, 142, 106, 158, 133, 106, 149, 114, 89, 131, 118, 91, 107, 122, 101, 95, 116, 113, 147,
            144, 120, 150, 141, 139, 147, 121, 135, 134, 103, 104, 120, 120, 93, 105, 139, 101, 97, 139, 130, 117,
            113, 141, 124, 112, 158, 134, 106, 152, 129, 108, 115, 115, 128, 96, 98, 144, 102, 91, 145, 136, 101,
            93, 144, 112, 98, 157, 132, 108, 152, 131, 123, 117, 114, 138, 94, 95, 143, 98, 88, 141,
        ]
        let out = ImageFilters.gaussian(Self.noisy, width: 7, height: 6, sigma: 1.0)
        #expect(zip(out, expected).allSatisfy { abs(Int($0) - Int($1)) <= 1 })
    }

    @Test func gaussianKernelSizeFollowsOpenCV() {
        #expect(ImageFilters.gaussianKernel(sigma: 1.0, eightBit: true).count == 7)
        #expect(ImageFilters.gaussianKernel(sigma: 5.0, eightBit: false).count == 41)
        #expect(abs(ImageFilters.gaussianKernel(sigma: 5.0, eightBit: false).reduce(0, +) - 1) < 1e-9)
    }

    @Test func reflect101Indices() {
        #expect((-3...8).map { ImageFilters.reflect101($0, 6) } == [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2])
    }
}

/// Tile layout + feather for the VOSR VAE/DiT passes (Forge's `_make_tile_grid` / `_blend_weights`).
@Suite struct UpscaleTilingTests {
    @Test func gridCoversTheLengthAndEndsFlush() {
        #expect(UpscaleTiling.grid(length: 64, tile: 64, overlap: 32) == [0])
        #expect(UpscaleTiling.grid(length: 40, tile: 64, overlap: 32) == [0])
        #expect(UpscaleTiling.grid(length: 448, tile: 64, overlap: 32) == Array(stride(from: 0, through: 384, by: 32)))
        #expect(UpscaleTiling.grid(length: 100, tile: 64, overlap: 8) == [0, 36])
    }

    @Test func featherIsRaisedCosineFlooredAt1e3() {
        let w = UpscaleTiling.feather(tile: 8, overlap: 4)
        let expected: [Float] = (0..<8).map { i in
            let x = Float(i) + 0.5
            let ramp = min(min(x, 8 - x) / 4, 1)
            return max(pow(sin(ramp * .pi / 2), 2), 1e-3)
        }
        #expect(zip(w, expected).allSatisfy { abs($0 - $1) < 1e-7 })
        // Interior beyond the overlap is fully weighted.
        let wide = UpscaleTiling.feather(tile: 16, overlap: 4)
        #expect(wide[4...11].allSatisfy { $0 == 1 })
    }

    @Test func latentTileParamsMatchForge() {
        // restore_latent: tile 512px, overlap 256px, patch 2, latent 448×448 (3584 px).
        let p = UpscaleTiling.ditTile(tilePixels: 512, overlapPixels: 256, latentH: 448, latentW: 448, patch: 2)
        #expect(p.tile == 64 && p.overlap == 32)
        // A latent smaller than the tile shrinks the tile to the patch-aligned short side.
        let q = UpscaleTiling.ditTile(tilePixels: 512, overlapPixels: 256, latentH: 40, latentW: 61, patch: 2)
        #expect(q.tile == 40 && q.overlap == 32)
        // VAE: tile 1024px, overlap 128px.
        let v = UpscaleTiling.vaeTile(tilePixels: 1024, overlapPixels: 128, latentH: 448, latentW: 300)
        #expect(v.tile == 128 && v.overlap == 16 && v.tiled)
        #expect(!UpscaleTiling.vaeTile(tilePixels: 1024, overlapPixels: 128, latentH: 64, latentW: 64).tiled)
    }

    @Test func outputSizeIsMultipleOf8AndAtLeastOnePatch() {
        #expect(UpscaleTiling.targetSize(width: 896, height: 600, scale: 4) == (3584, 2400))
        #expect(UpscaleTiling.targetSize(width: 101, height: 77, scale: 2) == (200, 152))
        #expect(UpscaleTiling.targetSize(width: 3, height: 3, scale: 1) == (16, 16))
    }
}
