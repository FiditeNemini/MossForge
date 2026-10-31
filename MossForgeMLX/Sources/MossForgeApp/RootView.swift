import AppKit
import SwiftUI
import Tweakbench

// Sidebar shell + themed sidebar.

/// Sidebar shell: the screens on the left, the selected screen on the right.
struct RootView: View {
    @Environment(AppStore.self) private var store

    private var selection: Binding<Screen?> {
        Binding(get: { store.selection }, set: { store.selection = $0 ?? store.selection })
    }

    var body: some View {
        NavigationSplitView {
            Group {
                if theme.isNative {
                    List(Screen.allCases, id: \.self, selection: selection) { screen in
                        Label(screen.title, systemImage: screen.symbol)
                    }
                } else {
                    ThemedSidebar(selection: Binding(get: { store.selection }, set: { store.selection = $0 }))
                }
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190, max: 230).tweakable("sidebar")
            .navigationTitle("MossForge")
        } detail: {
            detail
                .navigationTitle(store.selection.title)
                .toolbar {
                    // One toolbar for every screen, in a fixed order — the screen's actions, then the
                    // backend status pill on its own — so the pill keeps its size when switching screens.
                    ToolbarItemGroup(placement: .automatic) { screenActions }
                    if #available(macOS 26.0, *) { ToolbarSpacer(.fixed) }
                    ToolbarItem(placement: .automatic) {
                        Label("MLX · Apple silicon · backend: \(store.backendName)", systemImage: "cpu")
                            .labelStyle(.titleAndIcon).font(.caption).foregroundStyle(.secondary).fixedSize()
                            .padding(.horizontal, 10)
                    }
                }
        }
        .appTheme(theme)
    }

    private var theme: Theme { Theme(store.settings.theme) }

    /// Toolbar buttons for the selected screen (left of the status pill).
    @ViewBuilder private var screenActions: some View {
        switch store.selection {
        case .gallery:
            Button { store.selectAllGallery() } label: { Label("Select All", systemImage: "checkmark.circle") }
                .keyboardShortcut("a", modifiers: .command).help("Select all (⌘A)")
            Button { store.confirmGalleryDelete = true } label: { Label("Move to Trash", systemImage: "trash") }
                .keyboardShortcut(.delete, modifiers: .command).help("Move the selection to the Trash (⌘⌫)")
                .disabled(store.gallerySelection.selected.isEmpty)
            Button { NSWorkspace.shared.open(store.settings.outputDirectory) } label: { Label("Open folder", systemImage: "folder") }
            Button { store.refreshGallery() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
        default:
            EmptyView()
        }
    }

    @ViewBuilder private var detail: some View {
        switch store.selection {
        case .generate: GenerateView()
        case .characterStudio: CharacterStudioView()
        case .gallery: GalleryView()
        case .models: ModelsView()
        case .train: TrainView()
        case .server: ServerView()
        case .help: HelpView().environment(store.help)
        case .settings: SettingsView()
        }
    }
}
