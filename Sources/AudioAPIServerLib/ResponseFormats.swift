import Foundation

// MARK: - OpenAI-Compatible Request/Response Types

/// POST /v1/audio/transcriptions response (json format)
public struct TranscriptionResponse: Codable, Sendable {
    public let text: String
}

/// POST /v1/audio/transcriptions response (verbose_json format)
public struct VerboseTranscriptionResponse: Codable, Sendable {
    public let text: String
    public let words: [WordTimestamp]?
}

/// Word-level timestamp in verbose transcription response
public struct WordTimestamp: Codable, Sendable {
    public let word: String
    public let start: Float
    public let end: Float
}

/// POST /v1/audio/speech request body
public struct SpeechRequest: Codable, Sendable {
    public let model: String?
    public let input: String
    public let voice: String?
    public let language: String?
    public let response_format: String?  // "wav" (only wav supported)
}

/// POST /v1/audio/respond request (custom endpoint)
public struct RespondRequest: Codable, Sendable {
    public let voice: String?
}

/// GET /health response
public struct HealthResponse: Codable, Sendable {
    public let status: String
    public let models: ModelStatus
}

/// GET /v1/models response
public struct ModelListResponse: Codable, Sendable {
    public let object: String
    public let data: [ModelInfo]
}

/// Individual model entry in /v1/models
public struct ModelInfo: Codable, Sendable {
    public let id: String
    public let object: String
    public let owned_by: String
}

/// POST /v1/audio/alignments response
public struct AlignmentResponse: Codable, Sendable {
    public let words: [WordTimestamp]
}

// MARK: - VAD Response

/// A single speech segment in a VAD response.
public struct VADSegmentResponse: Codable, Sendable {
    /// Start time of the speech segment in seconds
    public let start: Float
    /// End time of the speech segment in seconds
    public let end: Float
    /// Duration of this segment in seconds
    public let duration: Float
}

/// POST /v1/audio/vad response
public struct VADResponse: Codable, Sendable {
    /// Detected speech segments
    public let segments: [VADSegmentResponse]
    /// Total speech duration in seconds
    public let speech_duration: Float
    /// Total audio duration in seconds
    public let total_duration: Float
    /// Number of detected speech segments
    public let segment_count: Int
}

/// Generic error response
public struct ErrorResponse: Codable, Sendable {
    public let error: ErrorDetail
}

public struct ErrorDetail: Codable, Sendable {
    public let message: String
    public let type: String
    public let code: String?
}
