import Foundation
import ArgumentParser
import Qwen3ASR
import AudioCommon

public struct AlignCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "align",
        abstract: "Forced alignment: align text to audio with word-level timestamps"
    )

    @Argument(help: "Audio file (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Text to align (if omitted, transcribes first)")
    public var text: String?

    @Option(name: .shortAndLong, help: "ASR model for transcription: 0.6B (default), 1.7B, or full ID")
    public var model: String = "0.6B"

    @Option(name: .long, help: "Forced aligner model ID")
    public var alignerModel: String = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    @Flag(name: .long, help: "Disable VAD-based silence trimming")
    public var noVAD: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            var audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count, sampleRate: 16000))s @ 16kHz)")

            // VAD: trim silence and keep track of offset for timestamp correction
            var trimOffsetSeconds: Float = 0
            if !noVAD {
                print("Running VAD...")
                let vad = try await SileroVADModel.fromPretrained(progressHandler: reportProgress)
                let (trimmed, offsetSamples) = vadTrimAudio(audio, sampleRate: 16000, vad: vad)
                audio = trimmed
                trimOffsetSeconds = Float(offsetSamples) / 16000.0
                if trimOffsetSeconds > 0 {
                    print("  Trimmed \(String(format: "%.2f", trimOffsetSeconds))s of leading silence")
                }
            }

            var textToAlign = text

            // If no text provided, transcribe first
            if textToAlign == nil {
                let modelId = resolveASRModelId(model)
                let detectedSize = ASRModelSize.detect(from: modelId)
                let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"
                print("Loading ASR model (\(sizeLabel)): \(modelId)")

                let asrModel = try await Qwen3ASRModel.fromPretrained(
                    modelId: modelId, progressHandler: reportProgress)

                print("Transcribing...")
                textToAlign = asrModel.transcribe(audio: audio, sampleRate: 16000, language: language)
                print("Transcription: \(textToAlign!)")
            }

            guard let alignText = textToAlign, !alignText.isEmpty else {
                print("Error: no text to align")
                throw ExitCode(1)
            }

            print("Loading aligner model: \(alignerModel)")
            let aligner = try await Qwen3ForcedAligner.fromPretrained(
                modelId: alignerModel, progressHandler: reportProgress)

            print("Aligning...")
            let start = Date()
            let aligned = aligner.align(audio: audio, text: alignText, sampleRate: 16000)
            let elapsed = Date().timeIntervalSince(start)

            // Shift timestamps back to original audio timeline
            for word in aligned {
                let startStr = String(format: "%.2f", word.startTime + trimOffsetSeconds)
                let endStr = String(format: "%.2f", word.endTime + trimOffsetSeconds)
                print("[\(startStr)s - \(endStr)s] \(word.text)")
            }
            print("Alignment took \(String(format: "%.2f", elapsed))s")
        }
    }
}
