import Foundation
import Hummingbird
import HTTPTypes

// MARK: - POST /v1/audio/transcriptions

/// OpenAI-compatible transcription endpoint.
/// Accepts multipart/form-data with `file` (audio), optional `language`, optional `response_format`.
public enum TranscriptionRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        router.post("v1/audio/transcriptions") { request, context -> Response in
            // Require multipart content type
            guard let contentType = request.headers[.contentType],
                  contentType.contains("multipart/form-data"),
                  let boundary = MultipartParser.extractBoundary(from: contentType) else {
                return errorResponse(.badRequest, "Expected multipart/form-data with audio file")
            }

            // Collect body
            let body = try await request.body.collect(upTo: 100_000_000)  // 100 MB max
            let bodyData = Data(buffer: body)

            // Parse multipart
            let parts = MultipartParser.parse(data: bodyData, boundary: boundary)

            guard let filePart = parts.first(where: { $0.name == "file" }) else {
                return errorResponse(.badRequest, "Missing 'file' field in multipart body")
            }

            let language = parts.first(where: { $0.name == "language" })
                .flatMap { String(data: $0.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) }
            let responseFormat = parts.first(where: { $0.name == "response_format" })
                .flatMap { String(data: $0.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) }
                ?? "json"

            // Decode audio
            let samples: [Float]
            do {
                samples = try AudioDecoder.decode(data: filePart.data, targetSampleRate: 16000)
            } catch {
                return errorResponse(.badRequest, "Failed to decode audio: \(error.localizedDescription)")
            }

            // Transcribe
            let text = await models.transcribe(audio: samples, sampleRate: 16000, language: language)

            // Format response
            switch responseFormat {
            case "text":
                return Response(
                    status: .ok,
                    headers: [.contentType: "text/plain; charset=utf-8"],
                    body: .init(byteBuffer: .init(string: text))
                )

            case "verbose_json":
                // Include word timestamps if aligner is available
                let words: [WordTimestamp]?
                let hasAligner = await models.status.aligner
                if hasAligner {
                    let aligned = await models.align(
                        audio: samples, text: text, sampleRate: 16000, language: language
                    )
                    words = aligned.map { WordTimestamp(word: $0.text, start: $0.startTime, end: $0.endTime) }
                } else {
                    words = nil
                }
                let response = VerboseTranscriptionResponse(text: text, words: words)
                return jsonResponse(response)

            default:  // "json"
                let response = TranscriptionResponse(text: text)
                return jsonResponse(response)
            }
        }
    }
}
