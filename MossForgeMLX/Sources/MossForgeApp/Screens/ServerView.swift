import AppKit
import SwiftUI
import Tweakbench

/// Resident-server control panel — driven by the **real** `ServerController`
/// (Network.framework listener over the live engine). Start/stop binds an actual
/// socket; the log shows real requests; the bind host follows the egress policy.
struct ServerView: View {
    @Environment(AppStore.self) private var store

    private var server: ServerController { store.server }

    private let endpoints = [
        "GET  /health",
        "GET  /v1/models",
        "POST /v1/images/generations",
        "POST /v1/moss/generate",
    ]

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Circle().fill(server.running ? .green : .secondary).frame(width: 9, height: 9)
                    Text(server.running
                        ? "running · \(server.bindHost):\(store.settings.serverPort)"
                        : "stopped").font(.callout).monospacedDigit()
                }
                .tweakable("server.status")
                GroupBox {
                    row("Backend", store.backendName)
                    row("Bind host", server.bindHost)
                    row("Reachable", store.settings.egressEnabled ? "this Mac + LAN" : "this Mac only")
                }
                .tweakable("server.info")
                GroupBox("Endpoints") {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(endpoints, id: \.self) {
                            Text($0).font(.system(.caption, design: .monospaced))
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
                .tweakable("server.endpoints")
                HStack {
                    Button { server.toggle() } label: {
                        Label(server.running ? "Stop" : "Start",
                            systemImage: server.running ? "pause.fill" : "play.fill")
                    }.buttonStyle(.borderedProminent)
                    if server.running {
                        Button { copyCurl() } label: { Label("Copy test curl", systemImage: "doc.on.doc") }
                    }
                }
                .tweakable("server.buttons")
                if let err = server.lastError {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
                Label(
                    "Loopback-only until you enable network access in Settings → Privacy.",
                    systemImage: "lock.shield"
                ).font(.caption).foregroundStyle(.secondary)
                .tweakable("server.note")
            }.frame(width: 340)
            .tweakable("server.controls")

            VStack(alignment: .leading, spacing: 6) {
                Text("Request log").font(.caption).foregroundStyle(.secondary)
                GroupBox {
                    if server.log.isEmpty {
                        Text("No requests yet.").font(.caption2).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(server.log) { entry in
                                Text(entry.line)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .tweakable("server.log.entries")
                Spacer()
            }.frame(maxWidth: .infinity)
            .tweakable("server.log")
        }.padding(16)
    }

    private func copyCurl() {
        let port = store.settings.serverPort
        let cmd = """
            curl -s http://127.0.0.1:\(port)/v1/moss/generate \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"a red fox in snow","variant":"turbo"}' --output fox.png
            """
        #if canImport(AppKit)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(cmd, forType: .string)
        #endif
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(v).font(.callout)
        }
    }
}
