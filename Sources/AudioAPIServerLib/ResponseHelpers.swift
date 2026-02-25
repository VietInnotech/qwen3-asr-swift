import Foundation
import Hummingbird
import HTTPTypes

// MARK: - Response Helpers

/// JSON-encode a Codable value into an HTTP Response.
public func jsonResponse<T: Encodable>(_ value: T, status: HTTPResponse.Status = .ok) -> Response {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    do {
        let data = try encoder.encode(value)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data))
        )
    } catch {
        let fallback = "{\"error\":{\"message\":\"Failed to encode response\",\"type\":\"server_error\",\"code\":null}}"
        return Response(
            status: .internalServerError,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(string: fallback))
        )
    }
}

/// Create a JSON error response matching OpenAI error format.
public func errorResponse(_ status: HTTPResponse.Status, _ message: String) -> Response {
    let error = ErrorResponse(
        error: ErrorDetail(
            message: message,
            type: "invalid_request_error",
            code: nil
        )
    )
    return jsonResponse(error, status: status)
}
