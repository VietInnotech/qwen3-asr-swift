import Foundation
import AudioCommon

// MARK: - Audio Decoding from Request Bytes

/// Converts raw uploaded audio bytes to Float samples.
/// Uses a temp file + AudioFileLoader (AVFoundation) to support WAV and other formats.
public enum AudioDecoder {

    /// Decode audio bytes to mono Float samples at the given sample rate.
    /// - Parameters:
    ///   - data: Raw audio file bytes (WAV, CAF, etc.)
    ///   - targetSampleRate: Desired output sample rate in Hz
    /// - Returns: Mono Float samples
    public static func decode(data: Data, targetSampleRate: Int) throws -> [Float] {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent(UUID().uuidString + ".wav")

        defer {
            try? FileManager.default.removeItem(at: tempFile)
        }

        try data.write(to: tempFile)
        return try AudioFileLoader.load(url: tempFile, targetSampleRate: targetSampleRate)
    }

    /// Encode Float samples to WAV Data (16-bit PCM).
    /// - Parameters:
    ///   - samples: Mono Float samples in [-1.0, 1.0]
    ///   - sampleRate: Sample rate in Hz
    /// - Returns: Complete WAV file as Data
    public static func encodeWAV(samples: [Float], sampleRate: Int) -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample) / 8
        let dataSize = samples.count * bytesPerSample
        let fileSize = 36 + dataSize

        var data = Data(capacity: fileSize + 8)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(&data, UInt32(fileSize))
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(&data, 16)
        appendUInt16(&data, 1)  // PCM
        appendUInt16(&data, numChannels)
        appendUInt32(&data, UInt32(sampleRate))
        appendUInt32(&data, UInt32(sampleRate * Int(numChannels) * bytesPerSample))
        appendUInt16(&data, numChannels * UInt16(bytesPerSample))
        appendUInt16(&data, bitsPerSample)

        // data chunk
        data.append(contentsOf: "data".utf8)
        appendUInt32(&data, UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Value = Int16(clamped * 32767.0)
            var le = int16Value.littleEndian
            data.append(Data(bytes: &le, count: 2))
        }

        return data
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 4))
    }

    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }
}
