import Foundation
import ArgumentParser
import Qwen3ASR
import AudioCommon

public struct TranscribeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe speech to text using Qwen3-ASR"
    )

    @Argument(help: "Audio file to transcribe (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Model: 0.6B (default), 1.7B, or full HuggingFace model ID")
    public var model: String = "0.6B"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    @Flag(name: .long, help: "Disable VAD-based silence trimming")
    public var noVAD: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)
            let detectedSize = ASRModelSize.detect(from: modelId)
            let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"

            print("Loading audio: \(audioFile)")
            var audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count, sampleRate: 16000))s @ 16kHz)")

            // VAD: concatenate all speech segments, removing all silence
            if !noVAD {
                print("Running VAD...")
                let vad = try await SileroVADModel.fromPretrained(progressHandler: reportProgress)
                audio = vadConcatenatedAudio(audio, sampleRate: 16000, vad: vad)
            }

            print("Loading model (\(sizeLabel)): \(modelId)")
            let asrModel = try await Qwen3ASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Transcribing...")
            let result = asrModel.transcribe(
                audio: audio,
                sampleRate: 16000,
                language: language,
                progressHandler: { chunkIndex, totalChunks, offsetSeconds in
                    if totalChunks > 1 {
                        let minutes = Int(offsetSeconds) / 60
                        let seconds = Int(offsetSeconds) % 60
                        print("  Chunk \(chunkIndex + 1)/\(totalChunks) (\(String(format: "%d:%02d", minutes, seconds))s)...")
                    }
                }
            )
            print("Result: \(result)")
        }
    }
}

/// Resolve shorthand model specifiers to HuggingFace model IDs.
public func resolveASRModelId(_ specifier: String) -> String {
    switch specifier.lowercased() {
    case "0.6b", "small":
        return ASRModelSize.small.defaultModelId
    case "1.7b", "large":
        return ASRModelSize.large.defaultModelId
    default:
        return specifier
    }
}
