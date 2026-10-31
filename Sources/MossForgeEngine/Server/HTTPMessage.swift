import Foundation

/// One parsed HTTP/1.1 request — just what the resident server's routes need.
public struct HTTPRequest: Sendable, Equatable {
    public var method: String
    /// The path without its query string.
    public var path: String
    public var query: [String: String]
    /// Header names lowercased.
    public var headers: [String: String]
    public var body: Data

    public init(
        method: String, path: String, query: [String: String] = [:],
        headers: [String: String] = [:], body: Data = Data()
    ) {
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body
    }
}

/// A complete HTTP/1.1 response; the connection is closed after each reply.
public struct HTTPResponse: Sendable {
    public var status: Int
    public var contentType: String
    public var headers: [String: String]
    public var body: Data

    public init(status: Int, contentType: String, headers: [String: String] = [:], body: Data) {
        self.status = status
        self.contentType = contentType
        self.headers = headers
        self.body = body
    }

    /// A JSON body from a `JSONSerialization`-compatible object.
    public static func json(_ object: Any, status: Int = 200) -> HTTPResponse {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
        return HTTPResponse(status: status, contentType: "application/json", body: data)
    }

    public static func png(_ data: Data, headers: [String: String] = [:]) -> HTTPResponse {
        HTTPResponse(status: 200, contentType: "image/png", headers: headers, body: data)
    }

    /// An OpenAI-shaped error (`{"error":{"message","type"}}`), serialized properly so any
    /// message — quotes, backslashes, paths — stays valid JSON.
    public static func error(_ status: Int, message: String, type: String? = nil) -> HTTPResponse {
        let kind = type ?? (status >= 500 ? "server_error" : "invalid_request_error")
        return .json(["error": ["message": message, "type": kind]], status: status)
    }

    public func serialized() -> Data {
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        for (name, value) in headers.sorted(by: { $0.key < $1.key }) { head += "\(name): \(value)\r\n" }
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(body)
        return out
    }

    static func reason(_ status: Int) -> String {
        switch status {
        case 200: "OK"
        case 400: "Bad Request"
        case 404: "Not Found"
        case 202: "Accepted"
        case 405: "Method Not Allowed"
        case 409: "Conflict"
        case 413: "Payload Too Large"
        case 500: "Internal Server Error"
        case 503: "Service Unavailable"
        default: "Status \(status)"
        }
    }
}

/// Incremental HTTP/1.1 request parsing over a growing receive buffer.
public enum HTTPRequestParser {
    public enum Result: Sendable, Equatable {
        /// Keep reading — the header or body hasn't fully arrived.
        case incomplete
        /// Malformed, or over the header/body limits: answer 400 and close.
        case invalid
        case complete(HTTPRequest)
    }

    /// Headers beyond this are refused (no route needs more).
    public static let maxHeaderBytes = 64 * 1024
    /// Bodies beyond this are refused — room for base64 init images, not unbounded buffering.
    public static let maxBodyBytes = 64 * 1024 * 1024

    public static func parse(_ data: Data) -> Result {
        guard let end = data.range(of: Data("\r\n\r\n".utf8)) else {
            return data.count > maxHeaderBytes ? .invalid : .incomplete
        }
        guard end.lowerBound - data.startIndex <= maxHeaderBytes else { return .invalid }
        let head = String(decoding: data[data.startIndex..<end.lowerBound], as: UTF8.self)
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1."),
            requestLine[0].allSatisfy({ $0.isUppercase }), requestLine[1].hasPrefix("/")
        else { return .invalid }

        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { return .invalid }
            headers[line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()] =
                line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        var length = 0
        if let raw = headers["content-length"] {
            guard let n = Int(raw), n >= 0, n <= maxBodyBytes else { return .invalid }
            length = n
        }
        let bodyStart = end.upperBound
        guard data.endIndex - bodyStart >= length else { return .incomplete }

        let target = String(requestLine[1])
        let (path, query) = splitTarget(target)
        return .complete(HTTPRequest(
            method: String(requestLine[0]), path: path, query: query, headers: headers,
            body: Data(data[bodyStart..<(bodyStart + length)])))
    }

    static func splitTarget(_ target: String) -> (String, [String: String]) {
        guard let q = target.firstIndex(of: "?") else { return (target, [:]) }
        var query: [String: String] = [:]
        for pair in target[target.index(after: q)...].split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1).map {
                String($0).replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? String($0)
            }
            if let key = kv.first, !key.isEmpty { query[key] = kv.count > 1 ? kv[1] : "" }
        }
        return (String(target[..<q]), query)
    }
}
