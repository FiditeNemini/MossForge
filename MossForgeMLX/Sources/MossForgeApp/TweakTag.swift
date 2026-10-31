import SwiftUI

#if canImport(Tweakbench)
import Tweakbench
#endif

extension View {
    /// A Tweakbench tag (`page.region.element`) when the app links Tweakbench;
    /// a no-op otherwise, so the template builds either way.
    func helpTag(_ id: String) -> some View {
        #if canImport(Tweakbench)
        tweakable(id)
        #else
        self
        #endif
    }
}
