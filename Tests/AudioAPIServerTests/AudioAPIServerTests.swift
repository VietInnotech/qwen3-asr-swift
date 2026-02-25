import XCTest
import Foundation
import Hummingbird
import HummingbirdTesting
import AudioCommon
@testable import AudioAPIServerLib

// MARK: - Mock Models

/// Mock ASR model that returns a fixed transcript.
final class MockASRModel: SpeechRecognitionModel {
    var inputSampleRate: Int { 16000 }
    var returnText: String
    init(returning text: String = "Hello world") { self.returnText = text }
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String { returnText }
}

/// Mock TTS model that returns silence.
final class MockTTSModel: SpeechGenerationModel {
    var sampleRate: Int { 24000 }
    func generate(text: String, language: String?) async throws -> [Float] { [Float](repeating: 0, count: 24000) }
    func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(AudioChunk(samples: [Float](repeating: 0, count: 24000),
                                          sampleRate: 24000, frameIndex: 0, isFinal: true))
            continuation.finish()
        }
    }
}

/// Mock ForcedAligner that returns fixed word timestamps.
final class MockAligner: ForcedAlignmentModel {
    func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord] {
        [AlignedWord(text: "Hello", startTime: 0.0, endTime: 0.5),
         AlignedWord(text: "world", startTime: 0.5, endTime: 1.0)]
    }
}

/// Mock speech-to-speech model that echoes input as silence.
final class MockPersonaPlexModel: SpeechToSpeechModel {
    var sampleRate: Int { 24000 }
    func respond(userAudio: [Float]) -> [Float] { [Float](repeating: 0, count: 24000) }
    func respondStream(userAudio: [Float]) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(AudioChunk(samples: [Float](repeating: 0, count: 24000),
                                          sampleRate: 24000, frameIndex: 0, isFinal: true))
            continuation.finish()
        }
    }
}

/// Mock VAD model that returns configurable speech segments.
final class MockVADModel: VoiceActivityDetector {
    var inputSampleRate: Int { 16000 }
    /// Segments to return from detectSpeech. Defaults to one segment covering all audio.
    var stubbedSegments: ((Int) -> [SpeechSegment]) = { total in
        [SpeechSegment(startSample: 0, endSample: total, sampleRate: 16000)]
    }
    func detectSpeech(audio: [Float], sampleRate: Int) throws -> [SpeechSegment] {
        stubbedSegments(audio.count)
    }
    func detectSpeechProbability(chunk: [Float]) throws -> Float { 0.9 }
    func resetState() {}
}

// MARK: - Helpers

/// Build a `ModelContext` with the given mocks (pass nil to leave that model unloaded).
func makeContext(
    asr: (any SpeechRecognitionModel)? = nil,
    tts: (any SpeechGenerationModel)? = nil,
    aligner: (any ForcedAlignmentModel)? = nil,
    personaPlex: (any SpeechToSpeechModel)? = nil,
    vad: (any VoiceActivityDetector)? = nil,
    ttsEngineName: String? = nil,
    ttsSampleRate: Int? = nil
) -> ModelContext {
    ModelContext(asr: asr, tts: tts, aligner: aligner, personaPlex: personaPlex,
                 vad: vad, ttsEngineName: ttsEngineName, ttsSampleRate: ttsSampleRate)
}

/// Build a Hummingbird `Application` with all routes registered against the given context.
func makeApp(context: ModelContext) -> some ApplicationProtocol {
    let router = Router()
    HealthRoute.register(on: router, models: context)
    TranscriptionRoute.register(on: router, models: context)
    SpeechRoute.register(on: router, models: context)
    AlignmentRoute.register(on: router, models: context)
    RespondRoute.register(on: router, models: context)
    VADRoute.register(on: router, models: context)
    return Application(router: router)
}

/// Build a minimal multipart body with a single file field.
func multipartBody(fileData: Data, filename: String = "audio.wav", boundary: String = "TestBoundary") -> (Data, String) {
    var body = Data()
    let header = "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\nContent-Type: audio/wav\r\n\r\n"
    body.append(header.data(using: .utf8)!)
    body.append(fileData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    let contentType = "multipart/form-data; boundary=\(boundary)"
    return (body, contentType)
}

/// Build a minimal WAV file (44-byte header + 16-bit PCM data) for testing.
func silentWAVData(samples: Int = 16000, sampleRate: Int = 16000) -> Data {
    AudioDecoder.encodeWAV(samples: [Float](repeating: 0, count: samples), sampleRate: sampleRate)
}

// MARK: - Health Tests

final class HealthRouteTests: XCTestCase {

    func testHealthEndpoint() async throws {
        let ctx = makeContext(asr: MockASRModel())
        let app = makeApp(context: ctx)
        try await app.test(.router) { client in
            try await client.execute(uri: "/health", method: .get) { response in
                XCTAssertEqual(response.status, .ok)
                let body = Data(buffer: response.body)
                let json = try JSONDecoder().decode(HealthResponse.self, from: body)
                XCTAssertEqual(json.status, "ok")
                XCTAssertTrue(json.models.asr)
                XCTAssertFalse(json.models.tts)
                XCTAssertFalse(json.models.aligner)
                XCTAssertFalse(json.models.personaPlex)
            }
        }
    }

    func testModelsEndpointListsLoadedModels() async throws {
        let ctx = makeContext(asr: MockASRModel(), tts: MockTTSModel(), ttsEngineName: "qwen3-tts")
        let app = makeApp(context: ctx)
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                XCTAssertEqual(response.status, .ok)
                let body = Data(buffer: response.body)
                let json = try JSONDecoder().decode(ModelListResponse.self, from: body)
                XCTAssertEqual(json.object, "list")
                XCTAssertEqual(json.data.count, 2)
                XCTAssertTrue(json.data.contains { $0.id == "qwen3-asr" })
                XCTAssertTrue(json.data.contains { $0.id == "qwen3-tts" })
            }
        }
    }
}

// MARK: - Transcription Tests

final class TranscriptionRouteTests: XCTestCase {

    func testTranscribeJSONFormat() async throws {
        let asr = MockASRModel(returning: "Hello world")
        let ctx = makeContext(asr: asr)
        let app = makeApp(context: ctx)
        let (body, contentType) = multipartBody(fileData: silentWAVData())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/transcriptions",
                method: .post,
                headers: [.contentType: contentType],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .ok)
                let responseData = Data(buffer: response.body)
                let json = try JSONDecoder().decode(TranscriptionResponse.self, from: responseData)
                XCTAssertEqual(json.text, "Hello world")
            }
        }
    }

    func testTranscribeTextFormat() async throws {
        let asr = MockASRModel(returning: "Hello world")
        let ctx = makeContext(asr: asr)
        let app = makeApp(context: ctx)

        var bodyData = Data()
        let boundary = "TestBoundary123"
        let fileData = silentWAVData()
        bodyData.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        bodyData.append(fileData)
        bodyData.append("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        let body = bodyData  // freeze for async capture

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/transcriptions",
                method: .post,
                headers: [.contentType: "multipart/form-data; boundary=\(boundary)"],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .ok)
                XCTAssertEqual(String(buffer: response.body), "Hello world")
            }
        }
    }

    func testTranscribeMissingFileFails() async throws {
        let ctx = makeContext(asr: MockASRModel())
        let app = makeApp(context: ctx)
        let boundary = "TestBoundary"
        let body = "--\(boundary)\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3-asr\r\n--\(boundary)--\r\n".data(using: .utf8)!

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/transcriptions",
                method: .post,
                headers: [.contentType: "multipart/form-data; boundary=\(boundary)"],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .badRequest)
            }
        }
    }

    func testTranscribeWrongContentTypeFails() async throws {
        let ctx = makeContext(asr: MockASRModel())
        let app = makeApp(context: ctx)

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/transcriptions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: "{}")
            ) { response in
                XCTAssertEqual(response.status, .badRequest)
            }
        }
    }
}

// MARK: - Speech (TTS) Tests

final class SpeechRouteTests: XCTestCase {

    func testSpeechReturnWAV() async throws {
        let ctx = makeContext(tts: MockTTSModel(), ttsEngineName: "qwen3-tts", ttsSampleRate: 24000)
        let app = makeApp(context: ctx)
        let reqBody = try JSONEncoder().encode(SpeechRequest(model: nil, input: "Hello", voice: nil, language: nil, response_format: nil))

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(data: reqBody)
            ) { response in
                XCTAssertEqual(response.status, .ok)
                XCTAssertEqual(response.headers[.contentType], "audio/wav")
                let wavData = Data(buffer: response.body)
                XCTAssertGreaterThan(wavData.count, 44)  // at least a WAV header
                // Verify RIFF header
                XCTAssertEqual(wavData.prefix(4), "RIFF".data(using: .utf8)!)
            }
        }
    }

    func testSpeechUnavailableReturns503() async throws {
        let ctx = makeContext()  // no TTS loaded
        let app = makeApp(context: ctx)
        let reqBody = try JSONEncoder().encode(SpeechRequest(model: nil, input: "Hello", voice: nil, language: nil, response_format: nil))

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(data: reqBody)
            ) { response in
                XCTAssertEqual(response.status, .serviceUnavailable)
            }
        }
    }

    func testSpeechEmptyInputFails() async throws {
        let ctx = makeContext(tts: MockTTSModel(), ttsSampleRate: 24000)
        let app = makeApp(context: ctx)
        let reqBody = try JSONEncoder().encode(SpeechRequest(model: nil, input: "", voice: nil, language: nil, response_format: nil))

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(data: reqBody)
            ) { response in
                XCTAssertEqual(response.status, .badRequest)
            }
        }
    }
}

// MARK: - Alignment Tests

final class AlignmentRouteTests: XCTestCase {

    func testAlignmentReturnsWords() async throws {
        let ctx = makeContext(aligner: MockAligner())
        let app = makeApp(context: ctx)

        var bodyData = Data()
        let boundary = "TestBoundary"
        let fileData = silentWAVData()
        bodyData.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        bodyData.append(fileData)
        bodyData.append("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\nHello world\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        let body = bodyData  // freeze for async capture

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/alignments",
                method: .post,
                headers: [.contentType: "multipart/form-data; boundary=\(boundary)"],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .ok)
                let data = Data(buffer: response.body)
                let json = try JSONDecoder().decode(AlignmentResponse.self, from: data)
                XCTAssertEqual(json.words.count, 2)
                XCTAssertEqual(json.words[0].word, "Hello")
                XCTAssertEqual(json.words[1].word, "world")
            }
        }
    }

    func testAlignmentWithoutAlignerReturns503() async throws {
        let ctx = makeContext()
        let app = makeApp(context: ctx)
        let (body, contentType) = multipartBody(fileData: silentWAVData())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/alignments",
                method: .post,
                headers: [.contentType: contentType],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .serviceUnavailable)
            }
        }
    }
}

// MARK: - Respond (PersonaPlex) Tests

final class RespondRouteTests: XCTestCase {

    func testRespondReturnsWAV() async throws {
        let ctx = makeContext(personaPlex: MockPersonaPlexModel())
        let app = makeApp(context: ctx)
        let (body, contentType) = multipartBody(fileData: silentWAVData(sampleRate: 24000))

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/respond",
                method: .post,
                headers: [.contentType: contentType],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .ok)
                XCTAssertEqual(response.headers[.contentType], "audio/wav")
                let wavData = Data(buffer: response.body)
                XCTAssertEqual(wavData.prefix(4), "RIFF".data(using: .utf8)!)
            }
        }
    }

    func testRespondWithoutPersonaPlexReturns503() async throws {
        let ctx = makeContext()
        let app = makeApp(context: ctx)
        let (body, contentType) = multipartBody(fileData: silentWAVData())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/audio/respond",
                method: .post,
                headers: [.contentType: contentType],
                body: ByteBuffer(data: body)
            ) { response in
                XCTAssertEqual(response.status, .serviceUnavailable)
            }
        }
    }
}

// MARK: - AudioDecoder Tests

final class AudioDecoderTests: XCTestCase {

    func testEncodeDecodeRoundTrip() {
        let original: [Float] = (0..<100).map { sin(Float($0) * 0.1) }
        let wav = AudioDecoder.encodeWAV(samples: original, sampleRate: 16000)

        // Verify RIFF header
        XCTAssertEqual(wav.prefix(4), "RIFF".data(using: .utf8)!)
        XCTAssertEqual(wav[8..<12], "WAVE".data(using: .utf8)!)

        // Verify size: 44 header + 100 samples * 2 bytes = 244 bytes
        XCTAssertEqual(wav.count, 44 + 100 * 2)
    }

    func testEncodeWAVHeaderSampleRate() {
        let wav = AudioDecoder.encodeWAV(samples: [Float](repeating: 0, count: 10), sampleRate: 44100)
        // Sample rate is at bytes 24-27 (little-endian UInt32)
        let sampleRateBytes = wav[24..<28]
        let sampleRate = sampleRateBytes.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        XCTAssertEqual(Int(sampleRate), 44100)
    }
}

// MARK: - MultipartParser Tests

final class MultipartParserTests: XCTestCase {

    func testExtractBoundary() {
        XCTAssertEqual(
            MultipartParser.extractBoundary(from: "multipart/form-data; boundary=TestBoundary"),
            "TestBoundary"
        )
        XCTAssertEqual(
            MultipartParser.extractBoundary(from: "multipart/form-data; boundary=\"WebKitBound\""),
            "WebKitBound"
        )
        XCTAssertNil(MultipartParser.extractBoundary(from: "application/json"))
    }

    func testParsesSingleFilePart() {
        let boundary = "Boundary123"
        let fileContent = "FAKE AUDIO DATA".data(using: .utf8)!
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(fileContent)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        let parts = MultipartParser.parse(data: body, boundary: boundary)
        XCTAssertEqual(parts.count, 1)
        XCTAssertEqual(parts[0].name, "file")
        XCTAssertEqual(parts[0].filename, "audio.wav")
        XCTAssertEqual(parts[0].data, fileContent)
    }

    func testParsesMultipleFields() {
        let boundary = "Bound"
        var body = Data()
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nBINARY\r\n".data(using: .utf8)!)
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        let parts = MultipartParser.parse(data: body, boundary: boundary)
        XCTAssertEqual(parts.count, 2)
        XCTAssertEqual(parts[0].name, "file")
        XCTAssertEqual(String(data: parts[1].data, encoding: .utf8), "en")
    }
}

// MARK: - VAD Route Tests

final class VADRouteTests: XCTestCase {

    func testVADNotLoaded_ReturnsServiceUnavailable() async throws {
        let ctx = makeContext()  // No VAD
        let app = makeApp(context: ctx)
        let wav = silentWAVData(samples: 16000)
        let (body, ct) = multipartBody(fileData: wav)

        try await app.test(.router) { client in
            let response = try await client.execute(
                uri: "/v1/audio/vad", method: .post,
                headers: [.contentType: ct], body: ByteBuffer(data: body)
            )
            XCTAssertEqual(response.status, .serviceUnavailable)
        }
    }

    func testVADRoute_ReturnsSegments() async throws {
        let mockVAD = MockVADModel()
        // Stub: return one segment from 0s to 1s
        mockVAD.stubbedSegments = { _ in
            [SpeechSegment(startSample: 0, endSample: 16000, sampleRate: 16000)]
        }
        let ctx = makeContext(asr: MockASRModel(), vad: mockVAD)
        let app = makeApp(context: ctx)
        let wav = silentWAVData(samples: 16000)
        let (body, ct) = multipartBody(fileData: wav)

        try await app.test(.router) { client in
            let response = try await client.execute(
                uri: "/v1/audio/vad", method: .post,
                headers: [.contentType: ct], body: ByteBuffer(data: body)
            )
            XCTAssertEqual(response.status, .ok)
            let json = try JSONDecoder().decode(VADResponse.self, from: Data(buffer: response.body))
            XCTAssertEqual(json.segment_count, 1)
            XCTAssertEqual(json.segments[0].start, 0.0, accuracy: 0.001)
            XCTAssertEqual(json.segments[0].end, 1.0, accuracy: 0.001)
        }
    }

    func testVADRoute_NoSpeech_ReturnsEmptySegments() async throws {
        let mockVAD = MockVADModel()
        mockVAD.stubbedSegments = { _ in [] }  // No speech detected
        let ctx = makeContext(asr: MockASRModel(), vad: mockVAD)
        let app = makeApp(context: ctx)
        let wav = silentWAVData(samples: 16000)
        let (body, ct) = multipartBody(fileData: wav)

        try await app.test(.router) { client in
            let response = try await client.execute(
                uri: "/v1/audio/vad", method: .post,
                headers: [.contentType: ct], body: ByteBuffer(data: body)
            )
            XCTAssertEqual(response.status, .ok)
            let json = try JSONDecoder().decode(VADResponse.self, from: Data(buffer: response.body))
            XCTAssertEqual(json.segment_count, 0)
            XCTAssertEqual(json.speech_duration, 0.0, accuracy: 0.001)
        }
    }

    func testVADInHealthModels_WhenLoaded() async throws {
        let ctx = makeContext(vad: MockVADModel())
        let app = makeApp(context: ctx)

        try await app.test(.router) { client in
            let response = try await client.execute(uri: "/v1/models", method: .get)
            XCTAssertEqual(response.status, .ok)
            let json = try JSONDecoder().decode(ModelListResponse.self, from: Data(buffer: response.body))
            XCTAssertTrue(json.data.contains(where: { $0.id == "silero-vad" }))
        }
    }
}