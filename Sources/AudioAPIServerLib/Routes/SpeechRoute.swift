import Foundation
import Hummingbird
import HTTPTypes

// MARK: - POST /v1/audio/speech

/// OpenAI-compatible text-to-speech endpoint.
/// Accepts JSON body with `input` (text), optional `voice`, `language`, `response_format`.
public enum SpeechRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        router.post("v1/audio/speech") { request, context -> Response in
            // Check TTS is available
            let status = await models.status
            guard status.tts else {
                return errorResponse(.serviceUnavailable, "TTS model not loaded. Start server with --tts-engine qwen3|cosyvoice")
            }

            // Decode JSON body
            let body = try await request.body.collect(upTo: 1_000_000)  // 1 MB max for text
            let bodyData = Data(buffer: body)

            let speechRequest: SpeechRequest
            do {
                speechRequest = try JSONDecoder().decode(SpeechRequest.self, from: bodyData)
            } catch {
                return errorResponse(.badRequest, "Invalid JSON body: \(error.localizedDescription)")
            }

            guard !speechRequest.input.isEmpty else {
                return errorResponse(.badRequest, "Missing or empty 'input' field")
            }

            // Generate speech
            let samples: [Float]
            do {
                samples = try await models.generateSpeech(
                    text: speechRequest.input,
                    language: speechRequest.language
                )
            } catch {
                return errorResponse(.internalServerError, "Speech generation failed: \(error.localizedDescription)")
            }

            // Encode to WAV
            let sampleRate = models.ttsSampleRate ?? 24000
            let wavData = AudioDecoder.encodeWAV(samples: samples, sampleRate: sampleRate)

            return Response(
                status: .ok,
                headers: [
                    .contentType: "audio/wav",
                    .init("Content-Disposition")!: "attachment; filename=\"speech.wav\""
                ],
                body: .init(byteBuffer: .init(data: wavData))
            )
        }
    }
}
