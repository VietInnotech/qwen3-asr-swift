import Foundation
import ArgumentParser
import AudioCommon

/// Run an async block from a synchronous ArgumentParser `run()` method.
public func runAsync(_ block: @escaping () async throws -> Void) throws {
    let semaphore = DispatchSemaphore(value: 0)
    var exitCode: Int32 = 0

    Task {
        do {
            try await block()
        } catch {
            print("Error: \(error)")
            exitCode = 1
        }
        semaphore.signal()
    }

    semaphore.wait()
    if exitCode != 0 {
        throw ExitCode(exitCode)
    }
}

/// Print model loading progress in a consistent format.
public func reportProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

/// Format audio duration from sample count.
public func formatDuration(_ samples: Int, sampleRate: Int = 24000) -> String {
    String(format: "%.2f", Double(samples) / Double(sampleRate))
}

// MARK: - VAD CLI Helpers

/// Run VAD and concatenate all detected speech segments into a continuous audio stream,
/// removing all inter-segment silence. Prints segment info. Falls back to original audio on error.
///
/// Use this for transcription where no timestamp mapping is needed.
public func vadConcatenatedAudio(_ audio: [Float], sampleRate: Int, vad: VoiceActivityDetector) -> [Float] {
    do {
        let segments = try vad.detectSpeech(audio: audio, sampleRate: sampleRate)
        guard !segments.isEmpty else {
            print("  VAD: no speech detected, using full audio")
            return audio
        }

        let totalSpeech = segments.reduce(0.0) { $0 + $1.duration }
        let originalDuration = Float(audio.count) / Float(sampleRate)
        print("  VAD: \(segments.count) speech segment(s), \(String(format: "%.2f", totalSpeech))s / \(String(format: "%.2f", originalDuration))s")

        var result: [Float] = []
        result.reserveCapacity(Int(totalSpeech * Float(sampleRate)))
        for segment in segments {
            let end = min(segment.endSample, audio.count)
            if segment.startSample < end {
                result.append(contentsOf: audio[segment.startSample ..< end])
            }
        }
        let silenceRemoved = originalDuration - totalSpeech
        print("  VAD: removed \(String(format: "%.2f", silenceRemoved))s of silence")
        return result
    } catch {
        print("  VAD warning: \(error.localizedDescription) — using full audio")
        return audio
    }
}

/// Run VAD and return trimmed audio plus the start offset in samples.
public func vadTrimAudio(_ audio: [Float], sampleRate: Int, vad: VoiceActivityDetector) -> (samples: [Float], offsetSamples: Int) {
    do {
        let segments = try vad.detectSpeech(audio: audio, sampleRate: sampleRate)
        guard !segments.isEmpty else {
            print("  VAD: no speech detected, using full audio")
            return (audio, 0)
        }

        let totalSpeech = segments.reduce(0.0) { $0 + $1.duration }
        let originalDuration = Float(audio.count) / Float(sampleRate)
        print("  VAD: \(segments.count) speech segment(s), \(String(format: "%.2f", totalSpeech))s / \(String(format: "%.2f", originalDuration))s")

        let start = segments.first!.startSample
        let end   = segments.last!.endSample
        let trimmed = Array(audio[start ..< min(end, audio.count)])
        let savedSeconds = Float(start) / Float(sampleRate)
        if savedSeconds > 0.05 {
            print("  VAD: trimmed \(String(format: "%.2f", savedSeconds))s leading silence")
        }
        return (trimmed, start)
    } catch {
        print("  VAD warning: \(error.localizedDescription) — using full audio")
        return (audio, 0)
    }
}
