import Foundation
import CoreML

// MARK: - Silero VAD (CoreML unified model, single stateful pass)

/// Voice Activity Detector using Silero VAD v6.0.0 via CoreML.
///
/// Wraps `silero-vad-unified-v6.0.0.mlmodelc` — a unified, single-pass model
/// that processes 512-sample audio chunks at 16 kHz (32 ms per call) and
/// maintains LSTM hidden + cell state between consecutive chunks for temporal
/// coherence.
///
/// **Model I/O (verified from metadata.json):**
///
/// Inputs:
///   - `audio_input`   Float32 [1, 576]  — 64-sample context window + 512-sample chunk
///   - `hidden_state`  Float32 [1, 128]  — LSTM hidden state (h)
///   - `cell_state`    Float32 [1, 128]  — LSTM cell state  (c)
///
/// Outputs:
///   - `vad_output`        Float32 [1, 1, 1] — speech probability in [0, 1]
///   - `new_hidden_state`  Float32 [1, 128]
///   - `new_cell_state`    Float32 [1, 128]
///
/// Source: `FluidInference/silero-vad-coreml` on HuggingFace (MIT license).
///
/// **Thread safety:** one instance per audio stream. Call `resetState()` between
/// independent streams.
public final class SileroVADModel: VoiceActivityDetector {

    // MARK: - Constants

    public static let defaultModelId = "FluidInference/silero-vad-coreml"
    public static let modelFolderName = "silero-vad-unified-v6.0.0.mlmodelc"

    /// Samples per inference chunk at 16 kHz (32 ms)
    public static let chunkSamples = 512

    /// Context samples prepended to each chunk (64 samples = 4 ms)
    public static let contextSamples = 64

    /// Total audio_input length: context + chunk
    public static let inputLength = contextSamples + chunkSamples  // 576

    /// LSTM hidden size
    public static let hiddenSize = 128

    public let inputSampleRate: Int = 16000

    // MARK: - CoreML model

    private let model: MLModel

    // MARK: - LSTM state buffers

    private var hiddenState: MLMultiArray   // shape [1, 128]
    private var cellState: MLMultiArray     // shape [1, 128]

    /// Ring buffer: last `contextSamples` processed samples, prepended each chunk
    private var contextBuffer: [Float]

    // MARK: - Init

    /// Load the unified CoreML model from a directory that contains
    /// `silero-vad-unified-v6.0.0.mlmodelc`.
    public init(modelDirectory: URL) throws {
        let modelURL = modelDirectory.appendingPathComponent(Self.modelFolderName)
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine / GPU / CPU — CoreML picks best
        self.model = try MLModel(contentsOf: modelURL, configuration: config)

        self.hiddenState = try MLMultiArray(shape: [1, Self.hiddenSize as NSNumber], dataType: .float32)
        self.cellState   = try MLMultiArray(shape: [1, Self.hiddenSize as NSNumber], dataType: .float32)
        Self.zeroMultiArray(&self.hiddenState)
        Self.zeroMultiArray(&self.cellState)

        self.contextBuffer = [Float](repeating: 0, count: Self.contextSamples)
    }

    // MARK: - fromPretrained

    /// Load Silero VAD from bundled repo resources, or download if not found.
    /// Checks in order:
    ///   1. Bundled in repo: `Models/silero-vad/` (relative to executable or current working directory)
    ///   2. HuggingFace cache: `~/Library/Caches/qwen3-speech/FluidInference_silero-vad-coreml/`
    ///   3. Download from HuggingFace if not in cache
    public static func fromPretrained(
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SileroVADModel {
        // 1. Try bundled model (for deployed binaries or development checkout)
        let bundledCandidates = [
            URL(fileURLWithPath: "Models/silero-vad"),  // relative to cwd
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models/silero-vad"),
            URL(fileURLWithPath: "/Users/vit/qwen3-asr-swift/Models/silero-vad")  // dev path
        ]
        
        for candidate in bundledCandidates {
            let modelPath = candidate.appendingPathComponent(modelFolderName)
            let sentinelFile = modelPath.appendingPathComponent("coremldata.bin")
            if FileManager.default.fileExists(atPath: sentinelFile.path) {
                progressHandler?(1.0, "Loading bundled Silero VAD...")
                return try SileroVADModel(modelDirectory: candidate)
            }
        }
        
        // 2. Try HuggingFace cache
        let directory = try HuggingFaceDownloader.getCacheDirectory(for: defaultModelId)
        let sentinelFile = directory
            .appendingPathComponent(modelFolderName)
            .appendingPathComponent("coremldata.bin")

        if !FileManager.default.fileExists(atPath: sentinelFile.path) {
            // 3. Download from HuggingFace
            progressHandler?(0.0, "Downloading Silero VAD CoreML model...")
            try await HuggingFaceDownloader.downloadRepoFiles(
                modelId: defaultModelId,
                to: directory,
                filterPaths: { path in
                    // Only download files belonging to the unified v6 model
                    path.hasPrefix(modelFolderName)
                },
                progressHandler: { progress in
                    progressHandler?(progress, "Downloading Silero VAD CoreML model...")
                }
            )
        }

        progressHandler?(1.0, "Loading Silero VAD...")
        return try SileroVADModel(modelDirectory: directory)
    }

    // MARK: - VoiceActivityDetector

    public func resetState() {
        Self.zeroMultiArray(&hiddenState)
        Self.zeroMultiArray(&cellState)
        contextBuffer = [Float](repeating: 0, count: Self.contextSamples)
    }

    /// Process a single 512-sample chunk (pre-resampled to 16 kHz).
    /// Updates LSTM state in place. Returns speech probability in [0, 1].
    public func detectSpeechProbability(chunk: [Float]) throws -> Float {
        guard chunk.count == Self.chunkSamples else {
            throw VADError.invalidChunkSize(expected: Self.chunkSamples, got: chunk.count)
        }

        // Concatenate context + chunk → audio_input [1, 576]
        let inputSamples = contextBuffer + chunk
        let audioInput = try MLMultiArray(
            shape: [1, Self.inputLength as NSNumber], dataType: .float32
        )
        for (i, v) in inputSamples.enumerated() { audioInput[i] = NSNumber(value: v) }

        // Update context buffer with the tail of this chunk
        contextBuffer = Array(chunk.suffix(Self.contextSamples))

        // Run inference
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "audio_input":  MLFeatureValue(multiArray: audioInput),
            "hidden_state": MLFeatureValue(multiArray: hiddenState),
            "cell_state":   MLFeatureValue(multiArray: cellState)
        ])
        let outputs = try model.prediction(from: inputs)

        // Update LSTM states in-place
        if let nh = outputs.featureValue(for: "new_hidden_state")?.multiArrayValue {
            for i in 0..<hiddenState.count { hiddenState[i] = nh[i] }
        }
        if let nc = outputs.featureValue(for: "new_cell_state")?.multiArrayValue {
            for i in 0..<cellState.count { cellState[i] = nc[i] }
        }

        // Extract scalar probability from vad_output [1, 1, 1]
        guard let probArray = outputs.featureValue(for: "vad_output")?.multiArrayValue else {
            throw VADError.missingModelOutput("vad_output")
        }
        return probArray[0].floatValue
    }

    /// Batch-mode detection over a complete audio buffer.
    /// Auto-resamples to 16 kHz if needed. Resets LSTM state before processing.
    public func detectSpeech(audio: [Float], sampleRate: Int) throws -> [SpeechSegment] {
        var samples = audio
        if sampleRate != inputSampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        resetState()

        var probabilities: [Float] = []
        probabilities.reserveCapacity(samples.count / Self.chunkSamples)

        var offset = 0
        while offset + Self.chunkSamples <= samples.count {
            let chunk = Array(samples[offset ..< offset + Self.chunkSamples])
            probabilities.append(try detectSpeechProbability(chunk: chunk))
            offset += Self.chunkSamples
        }

        return segmentsFromProbabilities(
            probabilities: probabilities,
            chunkSamples: Self.chunkSamples,
            sampleRate: inputSampleRate,
            totalSamples: samples.count
        )
    }

    // MARK: - Private Helpers

    private static func zeroMultiArray(_ array: inout MLMultiArray) {
        for i in 0..<array.count { array[i] = 0.0 }
    }
}

// MARK: - VAD Errors

public enum VADError: Error, LocalizedError {
    case invalidChunkSize(expected: Int, got: Int)
    case missingModelOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidChunkSize(let expected, let got):
            return "VAD expects chunks of exactly \(expected) samples, got \(got)"
        case .missingModelOutput(let name):
            return "CoreML model missing output '\(name)'"
        }
    }
}
