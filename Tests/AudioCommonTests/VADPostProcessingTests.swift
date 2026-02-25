import XCTest
import Foundation
@testable import AudioCommon

// MARK: - segmentsFromProbabilities Tests

final class VADPostProcessingTests: XCTestCase {

    // MARK: - Basic speech detection

    func testAllSilence() {
        let probs = [Float](repeating: 0.1, count: 50)
        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000, totalSamples: 50 * 512
        )
        XCTAssertTrue(segments.isEmpty)
    }

    func testAllSpeech() {
        let probs = [Float](repeating: 0.9, count: 50)
        let total = 50 * 512
        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000, totalSamples: total
        )
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].startSample, 0)   // Leading pad clamped to 0
        XCTAssertEqual(segments[0].endSample, total) // Trailing pad clamped to total
    }

    func testSingleSpeechSegment() {
        // Build: 10 silence chunks, 20 speech, 10 silence
        var probs = [Float](repeating: 0.1, count: 10)
        probs += [Float](repeating: 0.9, count: 20)
        probs += [Float](repeating: 0.1, count: 10)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: 0,  // Disable padding to get exact boundaries
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].startSample, 10 * 512)  // Starts at chunk 10
        XCTAssertGreaterThanOrEqual(segments[0].duration, 0)
    }

    func testTwoSpeechSegments() {
        // Build: silence, speech, long silence, speech, silence
        var probs = [Float](repeating: 0.1, count: 5)
        probs += [Float](repeating: 0.9, count: 15)
        probs += [Float](repeating: 0.1, count: 20)  // ~640ms silence at 16kHz
        probs += [Float](repeating: 0.9, count: 15)
        probs += [Float](repeating: 0.1, count: 5)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: 0,
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 2, "Long silence (~640ms) should separate into 2 segments")
    }

    func testShortSilence_DoesNotSplit() {
        // 5 silence chunks = 5*512 = 2560 samples = 160ms < 250ms threshold
        var probs = [Float](repeating: 0.9, count: 15)
        probs += [Float](repeating: 0.1, count: 3)  // ~96ms silence
        probs += [Float](repeating: 0.9, count: 15)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: 0,
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 1, "Short silence should not split segment")
    }

    func testMinSpeechDuration_FiltersShortBursts() {
        // 3 speech chunks = 3*512 = 1536 samples = 96ms < 250ms minimum
        var probs = [Float](repeating: 0.1, count: 10)
        probs += [Float](repeating: 0.9, count: 3)  // too short
        probs += [Float](repeating: 0.1, count: 10)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: 0,
            totalSamples: total
        )
        XCTAssertTrue(segments.isEmpty, "Speech burst < 250ms should be discarded")
    }

    func testHysteresis_NegThreshold() {
        // Probability drops to 0.4 (below threshold=0.5, above negThreshold=0.35)
        // Should NOT exit speech state
        var probs = [Float](repeating: 0.9, count: 10)
        probs += [Float](repeating: 0.4, count: 20)  // in-between — stays in speech
        probs += [Float](repeating: 0.9, count: 10)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: 0,
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 1, "Values between negThreshold and threshold keep speech active")
    }

    func testSpeechPadding() {
        let padSamples = 480  // 30ms
        var probs = [Float](repeating: 0.1, count: 10)
        probs += [Float](repeating: 0.9, count: 20)
        probs += [Float](repeating: 0.1, count: 10)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: padSamples,
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].startSample, max(0, 10 * 512 - padSamples))
    }

    func testEmptyInput() {
        let segments = segmentsFromProbabilities(
            probabilities: [], chunkSamples: 512, sampleRate: 16000, totalSamples: 0
        )
        XCTAssertTrue(segments.isEmpty)
    }

    func testMergesOverlappingPaddedSegments() {
        // Two segments whose padding would overlap — they should merge into one
        let padSamples = 1024  // Large padding
        var probs = [Float](repeating: 0.1, count: 5)
        probs += [Float](repeating: 0.9, count: 10)  // segment 1
        probs += [Float](repeating: 0.1, count: 3)   // short gap
        probs += [Float](repeating: 0.9, count: 10)  // segment 2
        probs += [Float](repeating: 0.1, count: 5)
        let total = probs.count * 512

        let segments = segmentsFromProbabilities(
            probabilities: probs, chunkSamples: 512, sampleRate: 16000,
            speechPadSamples: padSamples,
            totalSamples: total
        )
        XCTAssertEqual(segments.count, 1, "Overlapping padded segments should merge")
    }

    // MARK: - SpeechSegment struct

    func testSpeechSegmentTimestamps() {
        let seg = SpeechSegment(startSample: 16000, endSample: 48000, sampleRate: 16000)
        XCTAssertEqual(seg.startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(seg.endTime, 3.0, accuracy: 0.001)
        XCTAssertEqual(seg.duration, 2.0, accuracy: 0.001)
    }
}
