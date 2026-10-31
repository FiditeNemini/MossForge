import Foundation
import Testing

@testable import MossForgeCore

@Suite struct ChunkedDownloadTests {
    @Test func chunksCoverTheFileExactly() {
        let d = ChunkedDownload(size: 10, sha256: "abc", chunkSize: 4)
        #expect(d.chunkCount == 3)
        #expect(d.range(0) == 0...3 && d.range(1) == 4...7 && d.range(2) == 8...9)
        #expect(d.pending == [0, 1, 2] && d.bytesDone == 0)
    }

    @Test func doneChunksCountTowardProgressAndLeaveThePending() {
        var d = ChunkedDownload(size: 10, sha256: "abc", chunkSize: 4)
        d.done = [0, 2]
        #expect(d.pending == [1])
        #expect(d.bytesDone == 6)
        #expect(!d.isComplete)
        d.done.insert(1)
        #expect(d.isComplete && d.bytesDone == 10)
    }

    @Test func savedStateOnlyResumesTheSameFile() {
        let d = ChunkedDownload(size: 10, sha256: "ABC", chunkSize: 4, done: [1])
        #expect(d.resumes(size: 10, sha256: "abc"))
        #expect(!d.resumes(size: 11, sha256: "abc"))
        #expect(!d.resumes(size: 10, sha256: "def"))
    }

    @Test func emptyFileHasNoChunks() {
        let d = ChunkedDownload(size: 0, sha256: "e3b0")
        #expect(d.chunkCount == 0 && d.isComplete)
    }

    @Test func roundTripsAsJSON() throws {
        let d = ChunkedDownload(size: 100, sha256: "abc", chunkSize: 8, done: [3, 1])
        #expect(try JSONDecoder().decode(ChunkedDownload.self, from: JSONEncoder().encode(d)) == d)
    }

    @Test func defaultChunkIs32MiB() {
        #expect(ChunkedDownload.defaultChunkSize == 32 << 20)
    }
}
