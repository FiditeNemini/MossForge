import Foundation

/// How well this Mac's unified memory fits MossForge. The Mac App Store can't require a memory size, so the
/// app says so itself (Get Models, Settings → Memory) and picks safer defaults below the recommended size.
///
/// 64 GB is the recommended minimum: the heaviest measured run (enhance → generate → caption at 1024²) peaked
/// at ~41 GB with Save memory, ~60 GB with everything loaded. 32 GB may work with Save memory, one model at a
/// time (text encoder ~16 GB + int8 DiT ~5.4 GB + VAE + activations); it hasn't been run on a 32 GB Mac.
public enum MossMemoryTier: String, Sendable, Equatable {
    /// 64 GB or more: the defaults are fine.
    case recommended
    /// 32 GB up to 64 GB: may work with Save memory and images up to 1024².
    case limited
    /// Under 32 GB: below the minimum; generation is likely to fail.
    case belowMinimum

    public static let recommendedBytes: UInt64 = 64 << 30
    public static let minimumBytes: UInt64 = 32 << 30

    public init(physicalMemory: UInt64) {
        if physicalMemory >= Self.recommendedBytes {
            self = .recommended
        } else if physicalMemory >= Self.minimumBytes {
            self = .limited
        } else {
            self = .belowMinimum
        }
    }

    /// This Mac's memory; `MOSSFORGE_MEMORY_GB` overrides it to check the small-Mac notices on a big one.
    public static var physicalMemory: UInt64 {
        if let gb = ProcessInfo.processInfo.environment["MOSSFORGE_MEMORY_GB"].flatMap(UInt64.init) { return gb << 30 }
        return ProcessInfo.processInfo.physicalMemory
    }

    /// This Mac's tier.
    public static var current: MossMemoryTier { MossMemoryTier(physicalMemory: physicalMemory) }

    /// The Settings → Memory default for a fresh install.
    public var defaultResidencyMode: MossResidencyMode { self == .recommended ? .auto : .saveMemory }

    /// What to tell the user, or nil on a recommended Mac.
    public var notice: String? { Self.notice(tier: self, gb: nil) }

    /// The notice naming this Mac's memory, e.g. "This Mac has 36 GB of memory."
    public static func notice(physicalMemory: UInt64) -> String? {
        notice(tier: MossMemoryTier(physicalMemory: physicalMemory), gb: Int((physicalMemory + (1 << 29)) >> 30))
    }

    private static func notice(tier: MossMemoryTier, gb: Int?) -> String? {
        let mac = gb.map { "This Mac has \($0) GB of memory. " } ?? ""
        switch tier {
        case .recommended:
            return nil
        case .limited:
            return mac + "MossForge recommends 64 GB or more for optimal operation. 32 GB may work with the "
                + "right settings: on this Mac MossForge saves memory by loading one model at a time (keep Settings "
                + "→ Memory on Save memory or Automatic). Keep images at 1024×1024 or smaller, and close other "
                + "large apps while generating."
        case .belowMinimum:
            return mac + "MossForge needs at least 32 GB of memory, and 64 GB or more is recommended. Image "
                + "generation is likely to fail or be very slow on this Mac, so check before downloading the "
                + "models (about 16 GB)."
        }
    }
}
