import Foundation

// MARK: - Unified Audio Chunk

/// A chunk of audio produced during streaming synthesis or generation.
public struct AudioChunk: Sendable {
    /// PCM audio samples (Float32)
    public let samples: [Float]
    /// Sample rate in Hz (e.g. 24000)
    public let sampleRate: Int
    /// Index of the first frame in this chunk
    public let frameIndex: Int
    /// True if this is the last chunk
    public let isFinal: Bool
    /// Wall-clock seconds since generation started (nil if not tracked)
    public let elapsedTime: Double?

    public init(
        samples: [Float],
        sampleRate: Int,
        frameIndex: Int,
        isFinal: Bool,
        elapsedTime: Double? = nil
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.frameIndex = frameIndex
        self.isFinal = isFinal
        self.elapsedTime = elapsedTime
    }
}

// MARK: - Aligned Word

/// A word with its aligned start and end timestamps (in seconds).
public struct AlignedWord: Sendable {
    public let text: String
    public let startTime: Float
    public let endTime: Float

    public init(text: String, startTime: Float, endTime: Float) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

// MARK: - Speech Generation (TTS)

/// A text-to-speech model that generates audio from text.
public protocol SpeechGenerationModel: AnyObject {
    /// Output sample rate in Hz
    var sampleRate: Int { get }
    /// Synthesize audio from text (blocking, returns full waveform)
    func generate(text: String, language: String?) async throws -> [Float]
    /// Synthesize audio from text with streaming output
    func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error>
}

// MARK: - Speech Recognition (STT)

/// A speech-to-text model that transcribes audio.
public protocol SpeechRecognitionModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Transcribe audio to text
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String
}

// MARK: - Forced Alignment

/// A model that aligns text to audio at the word level.
public protocol ForcedAlignmentModel: AnyObject {
    /// Align text to audio, returning word-level timestamps
    func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord]
}

// MARK: - Speech-to-Speech

/// A speech-to-speech model that generates a spoken response to spoken input.
public protocol SpeechToSpeechModel: AnyObject {
    /// Output sample rate in Hz
    var sampleRate: Int { get }
    /// Generate response audio from input audio (blocking)
    func respond(userAudio: [Float]) -> [Float]
    /// Generate response audio from input audio with streaming output
    func respondStream(userAudio: [Float]) -> AsyncThrowingStream<AudioChunk, Error>
}

// MARK: - Speech Segment

/// A contiguous span of audio that contains detected speech.
public struct SpeechSegment: Sendable {
    /// Start position in the audio buffer (sample index)
    public let startSample: Int
    /// End position in the audio buffer (sample index, exclusive)
    public let endSample: Int
    /// Start time in seconds
    public let startTime: Float
    /// End time in seconds
    public let endTime: Float

    public init(startSample: Int, endSample: Int, sampleRate: Int) {
        self.startSample = startSample
        self.endSample = endSample
        self.startTime = Float(startSample) / Float(sampleRate)
        self.endTime = Float(endSample) / Float(sampleRate)
    }

    public var duration: Float { endTime - startTime }
}

// MARK: - Voice Activity Detection

/// A model that detects speech segments in audio.
public protocol VoiceActivityDetector: AnyObject {
    /// Expected input sample rate in Hz (Silero VAD uses 16000)
    var inputSampleRate: Int { get }

    /// Detect all speech segments in a complete audio buffer.
    /// Auto-resamples if `sampleRate` differs from `inputSampleRate`.
    func detectSpeech(audio: [Float], sampleRate: Int) throws -> [SpeechSegment]

    /// Process one pre-resampled chunk at `inputSampleRate`, returning speech probability in [0, 1].
    /// Maintains internal LSTM state across calls — suitable for streaming.
    func detectSpeechProbability(chunk: [Float]) throws -> Float

    /// Reset LSTM hidden state and context buffer.
    /// Must be called between independent audio streams.
    func resetState()
}


