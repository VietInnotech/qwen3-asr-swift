import Foundation

// MARK: - VAD Post-Processing

/// Converts a sequence of per-chunk speech probabilities into speech segments
/// with hysteresis, minimum-duration enforcement, and padding.
///
/// This is a standalone function — not a method on any model class — so it can
/// be unit-tested without loading any CoreML or ML model.
///
/// The algorithm matches the reference `get_speech_timestamps` logic from
/// snakers4/silero-vad (Python implementation).
///
/// - Parameters:
///   - probabilities: Per-chunk speech probability in [0, 1], one value per chunk.
///   - chunkSamples: Number of audio samples per chunk (e.g. 512 at 16 kHz = 32 ms).
///   - sampleRate: Sample rate of the original audio (used to compute time offsets).
///   - threshold: Probability above which we enter "speech" state (default 0.5).
///   - negThreshold: Probability below which we start counting silence (default 0.35).
///   - minSpeechSamples: Min speech run length in samples before it's counted as speech.
///     Defaults to 250 ms at `sampleRate`.
///   - minSilenceSamples: Min silence run length in samples before a gap is accepted.
///     Defaults to 100 ms at `sampleRate`.
///   - speechPadSamples: Padding added at both ends of each segment.
///     Defaults to 30 ms at `sampleRate`.
///   - totalSamples: Total length of the original audio buffer in samples,
///     used to clamp the final segment boundary.
/// - Returns: Sorted, non-overlapping `SpeechSegment` array in sample-index space.
public func segmentsFromProbabilities(
    probabilities: [Float],
    chunkSamples: Int,
    sampleRate: Int,
    threshold: Float = 0.5,
    negThreshold: Float = 0.35,
    minSpeechSamples: Int? = nil,
    minSilenceSamples: Int? = nil,
    speechPadSamples: Int? = nil,
    totalSamples: Int
) -> [SpeechSegment] {
    let minSpeech = minSpeechSamples ?? Int(0.25 * Float(sampleRate))
    let minSilence = minSilenceSamples ?? Int(0.10 * Float(sampleRate))
    let speechPad = speechPadSamples ?? Int(0.03 * Float(sampleRate))

    var segments: [SpeechSegment] = []
    var speechStart: Int? = nil
    var silenceStart: Int? = nil
    var inSpeech = false

    for (i, prob) in probabilities.enumerated() {
        let currentSample = i * chunkSamples

        if !inSpeech {
            if prob >= threshold {
                inSpeech = true
                speechStart = currentSample
                silenceStart = nil
            }
        } else {
            if prob < negThreshold {
                if silenceStart == nil {
                    silenceStart = currentSample
                } else if let silBegin = silenceStart,
                          (currentSample - silBegin) >= minSilence {
                    // Silence long enough — close the segment
                    if let start = speechStart {
                        let end = silBegin
                        if (end - start) >= minSpeech {
                            let padStart = max(0, start - speechPad)
                            let padEnd = min(totalSamples, end + speechPad)
                            segments.append(SpeechSegment(
                                startSample: padStart, endSample: padEnd, sampleRate: sampleRate
                            ))
                        }
                    }
                    inSpeech = false
                    speechStart = nil
                    silenceStart = nil
                }
            } else {
                silenceStart = nil  // Reset silence counter when speech resumes
            }
        }
    }

    // Close any still-open segment at end of audio
    if inSpeech, let start = speechStart {
        let end = silenceStart ?? totalSamples
        if (end - start) >= minSpeech {
            let padStart = max(0, start - speechPad)
            let padEnd = min(totalSamples, end + speechPad)
            segments.append(SpeechSegment(
                startSample: padStart, endSample: padEnd, sampleRate: sampleRate
            ))
        }
    }

    return mergeOverlappingSegments(segments, sampleRate: sampleRate)
}

// MARK: - Private Helpers

/// Merge overlapping or adjacent segments into one.
private func mergeOverlappingSegments(_ segments: [SpeechSegment], sampleRate: Int) -> [SpeechSegment] {
    guard !segments.isEmpty else { return [] }
    var merged: [SpeechSegment] = []
    var currentStart = segments[0].startSample
    var currentEnd = segments[0].endSample

    for seg in segments.dropFirst() {
        if seg.startSample <= currentEnd {
            currentEnd = max(currentEnd, seg.endSample)
        } else {
            merged.append(SpeechSegment(startSample: currentStart, endSample: currentEnd, sampleRate: sampleRate))
            currentStart = seg.startSample
            currentEnd = seg.endSample
        }
    }
    merged.append(SpeechSegment(startSample: currentStart, endSample: currentEnd, sampleRate: sampleRate))
    return merged
}
