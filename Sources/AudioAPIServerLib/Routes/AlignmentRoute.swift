import Foundation
import Hummingbird
import HTTPTypes

// MARK: - POST /v1/audio/alignments

/// Word-level forced alignment endpoint.
/// Accepts multipart/form-data with `file` (audio) and `text` (transcript).
public enum AlignmentRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        router.post("v1/audio/alignments") { request, context -> Response in
            // Check aligner is available
            let status = await models.status
            guard status.aligner else {
                return errorResponse(.serviceUnavailable, "Aligner not loaded. Start server with --enable-aligner")
            }

            // Require multipart
            guard let contentType = request.headers[.contentType],
                  contentType.contains("multipart/form-data"),
                  let boundary = MultipartParser.extractBoundary(from: contentType) else {
                return errorResponse(.badRequest, "Expected multipart/form-data with 'file' and 'text' fields")
            }

            let body = try await request.body.collect(upTo: 100_000_000)
            let bodyData = Data(buffer: body)
            let parts = MultipartParser.parse(data: bodyData, boundary: boundary)

            guard let filePart = parts.first(where: { $0.name == "file" }) else {
                return errorResponse(.badRequest, "Missing 'file' field")
            }
            guard let textPart = parts.first(where: { $0.name == "text" }),
                  let text = String(data: textPart.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else {
                return errorResponse(.badRequest, "Missing or empty 'text' field")
            }

            let language = parts.first(where: { $0.name == "language" })
                .flatMap { String(data: $0.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) }

            // Decode audio
            let samples: [Float]
            do {
                samples = try AudioDecoder.decode(data: filePart.data, targetSampleRate: 16000)
            } catch {
                return errorResponse(.badRequest, "Failed to decode audio: \(error.localizedDescription)")
            }

            // Align
            let aligned = await models.align(
                audio: samples, text: text, sampleRate: 16000, language: language
            )

            let words = aligned.map { WordTimestamp(word: $0.text, start: $0.startTime, end: $0.endTime) }
            return jsonResponse(AlignmentResponse(words: words))
        }
    }
}
