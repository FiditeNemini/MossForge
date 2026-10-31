import Foundation
import Testing

@testable import MossForgeCore

@Suite struct NetworkPolicyTests {
    @Test func defaultIsLoopbackOnly() {
        let p = NetworkPolicy()
        #expect(!p.egressAllowed)
        #expect(p.serverBindHost == "127.0.0.1")
        #expect(p.allowsOutbound(to: "localhost"))
        #expect(p.allowsOutbound(to: "127.0.0.1"))
        #expect(!p.allowsOutbound(to: "huggingface.co"))
    }

    @Test func egressOnBindsLANAndAllowsOutbound() {
        let p = NetworkPolicy(egressAllowed: true)
        #expect(p.serverBindHost == "0.0.0.0")
        #expect(p.allowsOutbound(to: "huggingface.co"))
        #expect(p.allowsOutbound(to: "127.0.0.1"))
    }

    @Test func codableRoundTrips() throws {
        for p in [NetworkPolicy(), NetworkPolicy(egressAllowed: true)] {
            let data = try JSONEncoder().encode(p)
            #expect(try JSONDecoder().decode(NetworkPolicy.self, from: data) == p)
        }
    }
}
