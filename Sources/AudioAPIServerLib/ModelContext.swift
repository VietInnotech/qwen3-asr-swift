import Foundation
import Logging
import Qwen3ASR
import Qwen3TTS
import CosyVoiceTTS
import PersonaPlex
import AudioCommon

// MARK: - Model Context

/// Thread-safe holder for loaded ML models.
/// Models are `AnyObject` (not `Sendable`), so we use an actor for safe concurrent access.
/// Protocol types are used for stored properties to allow injection of test doubles.
public actor ModelContext {
    public let asr: (any SpeechRecognitionModel)?
    public let tts: (any SpeechGenerationModel)?
    public let aligner: (any ForcedAlignmentModel)?
    public let personaPlex: (any SpeechToSpeechModel)?
    public let vad: (any VoiceActivityDetector)?

    /// TTS engine name for /v1/models listing
    public let ttsEngineName: String?
    /// TTS sample rate
    public let ttsSampleRate: Int?

    /// Designated init — accepts protocol types to support test doubles.
    public init(
        asr: (any SpeechRecognitionModel)?,
        tts: (any SpeechGenerationModel)?,
        aligner: (any ForcedAlignmentModel)?,
        personaPlex: (any SpeechToSpeechModel)?,
        vad: (any VoiceActivityDetector)? = nil,
        ttsEngineName: String?,
        ttsSampleRate: Int?
    ) {
        self.asr = asr
        self.tts = tts
        self.aligner = aligner
        self.personaPlex = personaPlex
        self.vad = vad
        self.ttsEngineName = ttsEngineName
        self.ttsSampleRate = ttsSampleRate
    }

    // MARK: - Inference Methods

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        guard let asr else { fatalError("ASR model not loaded") }
        return asr.transcribe(audio: audio, sampleRate: sampleRate, language: language)
    }

    public func generateSpeech(text: String, language: String?) async throws -> [Float] {
        guard let tts else { throw ModelError.modelNotLoaded("TTS") }
        return try await tts.generate(text: text, language: language)
    }

    public func generateSpeechStream(text: String, language: String?) throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard let tts else { throw ModelError.modelNotLoaded("TTS") }
        return tts.generateStream(text: text, language: language)
    }

    public func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord] {
        guard let aligner else { fatalError("Aligner model not loaded") }
        return aligner.align(audio: audio, text: text, sampleRate: sampleRate, language: language)
    }

    public func respond(audio: [Float]) -> [Float] {
        guard let personaPlex else { fatalError("PersonaPlex model not loaded") }
        return personaPlex.respond(userAudio: audio)
    }

    public func respondStream(audio: [Float]) throws -> AsyncThrowingStream<AudioChunk, Error> {
        guard let personaPlex else { throw ModelError.modelNotLoaded("PersonaPlex") }
        return personaPlex.respondStream(userAudio: audio)
    }

    // MARK: - VAD Methods

    /// Detect all speech segments in an audio buffer. Returns nil if VAD is not loaded.
    public func detectSpeech(audio: [Float], sampleRate: Int) throws -> [SpeechSegment]? {
        guard let vad else { return nil }
        return try vad.detectSpeech(audio: audio, sampleRate: sampleRate)
    }

    /// Concatenate all detected speech segments, removing all silence (inter-segment and leading/trailing).
    /// Use this for transcription where no timestamp mapping is needed.
    /// Returns the original audio unchanged if VAD is not loaded or no speech is found.
    public func concatenateSpeech(audio: [Float], sampleRate: Int) throws -> [Float] {
        guard let segments = try detectSpeech(audio: audio, sampleRate: sampleRate),
              !segments.isEmpty else {
            return audio
        }
        var result: [Float] = []
        for segment in segments {
            let end = min(segment.endSample, audio.count)
            if segment.startSample < end {
                result.append(contentsOf: audio[segment.startSample ..< end])
            }
        }
        return result
    }

    /// Trim leading and trailing silence from audio using VAD, returning the start offset.
    /// Use this when timestamps must map back to the original audio (e.g. forced alignment).
    /// Returns the original audio unchanged if VAD is not loaded or no speech is found.
    public func trimSilence(audio: [Float], sampleRate: Int) throws -> (samples: [Float], offsetSamples: Int) {
        guard let segments = try detectSpeech(audio: audio, sampleRate: sampleRate),
              !segments.isEmpty else {
            return (audio, 0)
        }
        let start = segments.first!.startSample
        let end = segments.last!.endSample
        let trimmed = Array(audio[start ..< min(end, audio.count)])
        return (trimmed, start)
    }

    /// Check whether the tail of an audio buffer contains speech (for end-of-turn detection).
    /// Returns `true` if the last `windowSamples` samples are above `threshold` on average.
    /// Returns `nil` if VAD is not loaded.
    public func isSpeaking(
        audio: [Float],
        sampleRate: Int,
        windowSeconds: Float = 0.3,
        threshold: Float = 0.4
    ) throws -> Bool? {
        guard let vad else { return nil }
        let windowSamples = Int(windowSeconds * Float(vad.inputSampleRate))
        guard audio.count >= windowSamples else { return nil }
        let tail = Array(audio.suffix(windowSamples))
        vad.resetState()
        var probabilities: [Float] = []
        var offset = 0
        let chunkSize = 512
        while offset + chunkSize <= tail.count {
            let chunk = Array(tail[offset ..< offset + chunkSize])
            probabilities.append(try vad.detectSpeechProbability(chunk: chunk))
            offset += chunkSize
        }
        guard !probabilities.isEmpty else { return nil }
        let avgProb = probabilities.reduce(0, +) / Float(probabilities.count)
        return avgProb >= threshold
    }

    // MARK: - Model Status

    public var status: ModelStatus {
        ModelStatus(
            asr: asr != nil,
            tts: tts != nil,
            aligner: aligner != nil,
            personaPlex: personaPlex != nil,
            vad: vad != nil,
            ttsEngine: ttsEngineName
        )
    }

    // MARK: - Factory

    public static func load(
        asrModelSpec: String,
        ttsEngine: String,
        ttsModelSpec: String,
        enableAligner: Bool,
        enablePersonaPlex: Bool,
        enableVAD: Bool = true,
        logger: Logger
    ) async throws -> ModelContext {
        // ASR
        let asrModelId = resolveASRModelId(asrModelSpec)
        logger.info("Loading ASR model: \(asrModelId)")
        let asrConcrete = try await Qwen3ASRModel.fromPretrained(
            modelId: asrModelId,
            progressHandler: { progress, status in
                logger.info("ASR: \(status) (\(Int(progress * 100))%)")
            }
        )
        let asr: any SpeechRecognitionModel = asrConcrete

        // TTS
        var tts: (any SpeechGenerationModel)? = nil
        var ttsEngineName: String? = nil
        var ttsSampleRate: Int? = nil

        switch ttsEngine.lowercased() {
        case "qwen3":
            let modelId = resolveTTSModelId(ttsModelSpec)
            logger.info("Loading Qwen3-TTS: \(modelId)")
            let ttsModel = try await Qwen3TTSModel.fromPretrained(
                modelId: modelId,
                progressHandler: { progress, status in
                    logger.info("TTS: \(status) (\(Int(progress * 100))%)")
                }
            )
            tts = ttsModel
            ttsEngineName = "qwen3-tts"
            ttsSampleRate = ttsModel.sampleRate

        case "cosyvoice":
            logger.info("Loading CosyVoice TTS...")
            let cosyModel = try await CosyVoiceTTSModel.fromPretrained(
                progressHandler: { progress, status in
                    logger.info("CosyVoice: \(status) (\(Int(progress * 100))%)")
                }
            )
            tts = cosyModel
            ttsEngineName = "cosyvoice-tts"
            ttsSampleRate = cosyModel.sampleRate

        case "none":
            logger.info("TTS disabled.")

        default:
            logger.warning("Unknown TTS engine '\(ttsEngine)', disabling TTS.")
        }

        // Forced Aligner
        var aligner: (any ForcedAlignmentModel)? = nil
        if enableAligner {
            logger.info("Loading Forced Aligner...")
            aligner = try await Qwen3ForcedAligner.fromPretrained(
                progressHandler: { progress, status in
                    logger.info("Aligner: \(status) (\(Int(progress * 100))%)")
                }
            )
        }

        // PersonaPlex
        var personaPlex: (any SpeechToSpeechModel)? = nil
        if enablePersonaPlex {
            logger.info("Loading PersonaPlex (5.3 GB)...")
            personaPlex = try await PersonaPlexModel.fromPretrained(
                progressHandler: { progress, status in
                    logger.info("PersonaPlex: \(status) (\(Int(progress * 100))%)")
                }
            )
        }

        // VAD
        var vad: (any VoiceActivityDetector)? = nil
        if enableVAD {
            logger.info("Loading Silero VAD...")
            vad = try await SileroVADModel.fromPretrained(
                progressHandler: { progress, status in
                    logger.info("VAD: \(status) (\(Int(progress * 100))%)")
                }
            )
        }

        return ModelContext(
            asr: asr,
            tts: tts,
            aligner: aligner,
            personaPlex: personaPlex,
            vad: vad,
            ttsEngineName: ttsEngineName,
            ttsSampleRate: ttsSampleRate
        )
    }
}

// MARK: - Errors

public enum ModelError: Error, LocalizedError {
    case modelNotLoaded(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded(let name):
            return "\(name) model is not loaded. Enable it with the appropriate flag."
        }
    }
}

// MARK: - Model Status

public struct ModelStatus: Codable, Sendable {
    public let asr: Bool
    public let tts: Bool
    public let aligner: Bool
    public let personaPlex: Bool
    public let vad: Bool
    public let ttsEngine: String?
}

// MARK: - Model ID Resolution

private func resolveASRModelId(_ spec: String) -> String {
    switch spec.lowercased() {
    case "0.6b", "small":
        return "mlx-community/Qwen3-ASR-0.6B-4bit"
    case "1.7b", "large":
        return "mlx-community/Qwen3-ASR-1.7B-8bit"
    default:
        return spec  // Full HuggingFace model ID
    }
}

private func resolveTTSModelId(_ spec: String) -> String {
    switch spec.lowercased() {
    case "base":
        return "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
    case "custom", "customvoice":
        return "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    default:
        return spec
    }
}
