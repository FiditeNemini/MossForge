import Foundation
import MossForgeCore

/// One edit-training example: the source (control) image, the instruction, the edited target.
public struct EditTrainingPair: Sendable, Equatable {
    public var control: URL
    public var target: URL
    public var instruction: String
}

/// Edit datasets in AI Toolkit's layout: the **targets** (edited results) with `name.txt` instructions
/// in the folder, and the **sources** with the same file stem in its `control/` subfolder.
public enum EditDatasetScanner {
    public struct Scan: Sendable, Equatable {
        public var pairs: [EditTrainingPair]
        /// Targets with no matching control image (skipped).
        public var missingControl: [URL]
    }

    public static func controlFolder(for dataset: URL) -> URL { dataset.appendingPathComponent("control") }

    public static func scan(directory: URL, control: URL? = nil, fileManager: FileManager = .default) -> Scan {
        let controlDir = control ?? controlFolder(for: directory)
        let controls = (try? fileManager.contentsOfDirectory(at: controlDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        let byStem = Dictionary(
            controls.filter { DatasetScanner.imageExtensions.contains($0.pathExtension.lowercased()) }
                .map { ($0.deletingPathExtension().lastPathComponent, $0) },
            uniquingKeysWith: { a, _ in a })
        var pairs: [EditTrainingPair] = []
        var missing: [URL] = []
        for sample in DatasetScanner.scan(directory: directory, fileManager: fileManager) {
            if let c = byStem[sample.image.deletingPathExtension().lastPathComponent] {
                pairs.append(EditTrainingPair(control: c, target: sample.image, instruction: sample.caption))
            } else {
                missing.append(sample.image)
            }
        }
        return Scan(pairs: pairs, missingControl: missing)
    }
}
