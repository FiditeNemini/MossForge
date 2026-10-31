import SwiftUI

extension View {
    /// Tags a view for live layout tweaking. A no-op in this stand-in.
    public func tweakable(_ id: String) -> some View { self }

    /// Installs the tweaking panel on a window's root view. A no-op in this stand-in.
    public func tweakbench() -> some View { self }
}
