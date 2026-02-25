import Foundation
import Hummingbird
import HTTPTypes

// MARK: - POST /v1/audio/respond

/// PersonaPlex speech-to-speech endpoint.
/// Accepts multipart/form-data with `file` (audio), optional `voice`.
public enum RespondRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        router.post("v1/audio/respond") { request, context -> Response in
            // Check PersonaPlex is available
            let status = await models.status
            guard status.personaPlex else {
                return errorResponse(.serviceUnavailable, "PersonaPlex not loaded. Start server with --enable-persona-plex")
            }

            // Require multipart
            guard let contentType = request.headers[.contentType],
                  contentType.contains("multipart/form-data"),
                  let boundary = MultipartParser.extractBoundary(from: contentType) else {
                return errorResponse(.badRequest, "Expected multipart/form-data with 'file' field")
            }

            let body = try await request.body.collect(upTo: 100_000_000)
            let bodyData = Data(buffer: body)
            let parts = MultipartParser.parse(data: bodyData, boundary: boundary)

            guard let filePart = parts.first(where: { $0.name == "file" }) else {
                return errorResponse(.badRequest, "Missing 'file' field")
            }

            // Decode audio at 24kHz (PersonaPlex input rate)
            let samples: [Float]
            do {
                samples = try AudioDecoder.decode(data: filePart.data, targetSampleRate: 24000)
            } catch {
                return errorResponse(.badRequest, "Failed to decode audio: \(error.localizedDescription)")
            }

            // Generate spoken response
            let responseSamples = await models.respond(audio: samples)

            // Encode to WAV at 24kHz
            let wavData = AudioDecoder.encodeWAV(samples: responseSamples, sampleRate: 24000)

            return Response(
                status: .ok,
                headers: [
                    .contentType: "audio/wav",
                    .init("Content-Disposition")!: "attachment; filename=\"response.wav\""
                ],
                body: .init(byteBuffer: .init(data: wavData))
            )
        }
    }
}
