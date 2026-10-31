import Foundation

/// What the app hands the collector: its own state (never prompts, images or file names the user
/// chose, unless in an opt-in section).
public struct DiagnosticsContext: Sendable {
    public var appVersion: String
    /// "mlx", "stub", … — "stub" is flagged as a warning.
    public var backendName: String
    public var capabilities: [String]
    /// Folders to check for existence and free space, e.g. ("Models folder", url, required: true).
    public var folders: [(label: String, url: URL, required: Bool)]
    /// Non-personal settings as label/value pairs (theme, port, network access, idle unload).
    public var settings: [(String, String)]
    public var log: DiagnosticsLog
    /// App-specific sections (models present and readable, GPU state, …), placed after Storage.
    public var extraSections: [DiagnosticSection]
    /// Below this much installed memory, the System section warns (0 = never).
    public var minimumMemoryGB: Double
    /// Below this much free space on a folder's volume, Storage warns.
    public var minimumFreeGB: Double

    public init(
        appVersion: String, backendName: String, capabilities: [String] = [],
        folders: [(label: String, url: URL, required: Bool)] = [], settings: [(String, String)] = [],
        log: DiagnosticsLog = DiagnosticsLog(), extraSections: [DiagnosticSection] = [],
        minimumMemoryGB: Double = 0, minimumFreeGB: Double = 20
    ) {
        self.appVersion = appVersion
        self.backendName = backendName
        self.capabilities = capabilities
        self.folders = folders
        self.settings = settings
        self.log = log
        self.extraSections = extraSections
        self.minimumMemoryGB = minimumMemoryGB
        self.minimumFreeGB = minimumFreeGB
    }
}

/// Help → Feedback → Run diagnostics: local checks only (no network), each a ✓ / ⚠︎ / ✗ line.
/// Run it off the main actor (`Task.detached`); file-system checks can be slow.
public enum DiagnosticsCollector {
    public static func collect(_ c: DiagnosticsContext) -> [DiagnosticSection] {
        [app(c), system(c), storage(c)] + c.extraSections + [settings(c), activity(c.log)]
    }

    static func app(_ c: DiagnosticsContext) -> DiagnosticSection {
        let stub = c.backendName == "stub"
        var items: [DiagnosticSection.Item] = [
            .init("Version", c.appVersion),
            .init("Backend", stub ? "stub — the real backend didn't load (models missing?)" : c.backendName,
                status: stub ? .warn : .ok),
        ]
        if !c.capabilities.isEmpty { items.append(.init("Capabilities", c.capabilities.joined(separator: ", "))) }
        return DiagnosticSection(id: "app", title: "App", items: items)
    }

    static func system(_ c: DiagnosticsContext) -> DiagnosticSection {
        let p = ProcessInfo.processInfo
        let gb = Double(p.physicalMemory) / 1_073_741_824
        let low = gb < c.minimumMemoryGB
        var items: [DiagnosticSection.Item] = [
            .init("macOS", p.operatingSystemVersionString),
            .init("Chip", sysctlString("machdep.cpu.brand_string") ?? "unknown"),
            .init("Memory", String(format: "%.0f GB", gb) + (low ? " — below the recommended \(Int(c.minimumMemoryGB)) GB" : ""),
                status: low ? .warn : .ok),
            .init("CPU cores", "\(p.activeProcessorCount)"),
        ]
        let thermal: (String, DiagnosticSection.Status) =
            switch p.thermalState {
            case .nominal: ("nominal", .ok)
            case .fair: ("fair", .ok)
            case .serious: ("serious — the Mac is throttling", .warn)
            case .critical: ("critical — the Mac is throttling hard", .fail)
            @unknown default: ("unknown", .info)
            }
        items.append(.init("Thermal state", thermal.0, status: thermal.1))
        if p.isLowPowerModeEnabled { items.append(.init("Low Power Mode", "on — slows the GPU", status: .warn)) }
        return DiagnosticSection(id: "system", title: "System", items: items)
    }

    static func storage(_ c: DiagnosticsContext) -> DiagnosticSection {
        var items: [DiagnosticSection.Item] = []
        for folder in c.folders {
            let exists = FileManager.default.fileExists(atPath: folder.url.path)
            let probe = exists ? folder.url : folder.url.deletingLastPathComponent()
            let free = (try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage
            var value = exists ? "exists" : "doesn't exist yet"
            var status: DiagnosticSection.Status = exists ? .ok : (folder.required ? .fail : .info)
            if let free {
                let gb = Double(free) / 1e9
                value += String(format: " · %.0f GB free", gb)
                if gb < c.minimumFreeGB {
                    status = .warn
                    value += " — low"
                }
            }
            items.append(.init(folder.label, value, status: status))
        }
        return DiagnosticSection(id: "storage", title: "Storage",
            items: items.isEmpty ? [.init("Folders", "none checked")] : items)
    }

    static func settings(_ c: DiagnosticsContext) -> DiagnosticSection {
        DiagnosticSection(id: "settings", title: "Settings", items: c.settings.map { DiagnosticSection.Item($0.0, $0.1) })
    }

    static func activity(_ log: DiagnosticsLog) -> DiagnosticSection {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss"
        let items = log.entries.map { e -> DiagnosticSection.Item in
            let status: DiagnosticSection.Status =
                switch e.level {
                case .info: .info
                case .warning: .warn
                case .error: .fail
                }
            return .init("\(f.string(from: e.date)) \(e.area)", e.message, status: status)
        }
        let note = "Job outcomes and errors since diagnostics were turned on (memory only, last \(log.capacity))"
            + (log.dropped > 0 ? "; \(log.dropped) older events dropped" : "")
        return DiagnosticSection(id: "activity", title: "Recent activity",
            items: items.isEmpty ? [.init("Events", "none recorded")] : items, included: false, note: note)
    }

    /// File sizes as "1.2 GB" / "340.0 MB", for app-specific model sections.
    public static func sizeText(_ url: URL) -> String {
        ((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize).map {
            $0 >= 100_000_000 ? String(format: "%.1f GB", Double($0) / 1e9) : String(format: "%.1f MB", Double($0) / 1e6)
        } ?? "?"
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
