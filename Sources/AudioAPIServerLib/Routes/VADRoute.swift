import Foundation
import Hummingbird
import HTTPTypes
import AudioCommon

// MARK: - POST /v1/audio/vad

/// Voice activity detection endpoint.
/// Accepts multipart/form-data with a `file` field containing an audio file.
/// Returns JSON with detected speech segments, their start/end times, and overall statistics.
public enum VADRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        router.post("v1/audio/vad") { request, context -> Response in
            // Require VAD to be loaded
            let status = await models.status
            guard status.vad else {
                return errorResponse(.serviceUnavailable, "VAD not loaded. Start server with --enable-vad")
            }

            // Require multipart
            guard let contentType = request.headers[.contentType],
                  contentType.contains("multipart/form-data"),
                  let boundary = MultipartParser.extractBoundary(from: contentType) else {
                return errorResponse(.badRequest, "Expected multipart/form-data with 'file' field")
            }

            let body = try await request.body.collect(upTo: 100_000_000)  // 100 MB max
            let bodyData = Data(buffer: body)
            let parts = MultipartParser.parse(data: bodyData, boundary: boundary)

            guard let filePart = parts.first(where: { $0.name == "file" }) else {
                return errorResponse(.badRequest, "Missing 'file' field")
            }

            // Decode audio to 16 kHz (Silero VAD input rate)
            let samples: [Float]
            do {
                samples = try AudioDecoder.decode(data: filePart.data, targetSampleRate: 16000)
            } catch {
                return errorResponse(.badRequest, "Failed to decode audio: \(error.localizedDescription)")
            }

            let totalDuration = Float(samples.count) / 16000.0

            // Run VAD
            let segments: [SpeechSegment]
            do {
                segments = try await models.detectSpeech(audio: samples, sampleRate: 16000) ?? []
            } catch {
                return errorResponse(.internalServerError, "VAD inference failed: \(error.localizedDescription)")
            }

            let segmentResponses = segments.map {
                VADSegmentResponse(start: $0.startTime, end: $0.endTime, duration: $0.duration)
            }
            let speechDuration = segmentResponses.reduce(0.0) { $0 + $1.duration }

            let response = VADResponse(
                segments: segmentResponses,
                speech_duration: speechDuration,
                total_duration: totalDuration,
                segment_count: segmentResponses.count
            )
            return jsonResponse(response)
        }
    }
}
