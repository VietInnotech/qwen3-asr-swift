import Foundation

// MARK: - Simple Multipart Parser

/// Minimal multipart/form-data parser for audio upload endpoints.
/// Extracts named fields (text) and file parts (binary) from a multipart body.
public enum MultipartParser {

    /// A parsed multipart form part.
    public struct Part {
        public let name: String
        public let filename: String?
        public let contentType: String?
        public let data: Data
    }

    /// Parse multipart/form-data body with the given boundary.
    /// - Parameters:
    ///   - data: Raw body bytes
    ///   - boundary: Boundary string from Content-Type header
    /// - Returns: Array of parsed parts
    public static func parse(data: Data, boundary: String) -> [Part] {
        let delimiter = "--\(boundary)".data(using: .utf8)!
        let endDelimiter = "--\(boundary)--".data(using: .utf8)!
        let crlf = "\r\n".data(using: .utf8)!
        let doubleCRLF = "\r\n\r\n".data(using: .utf8)!

        var parts: [Part] = []
        var searchStart = data.startIndex

        while let delimiterRange = data.range(of: delimiter, in: searchStart..<data.endIndex) {
            let afterDelimiter = delimiterRange.upperBound

            // Skip if this is the end delimiter
            if data[afterDelimiter..<min(afterDelimiter + 2, data.endIndex)] == "--".data(using: .utf8)! {
                break
            }

            // Skip the CRLF after delimiter
            let headerStart: Data.Index
            if afterDelimiter + crlf.count <= data.endIndex,
               data[afterDelimiter..<afterDelimiter + crlf.count] == crlf {
                headerStart = afterDelimiter + crlf.count
            } else {
                searchStart = afterDelimiter
                continue
            }

            // Find end of headers (double CRLF)
            guard let headerEnd = data.range(of: doubleCRLF, in: headerStart..<data.endIndex) else {
                break
            }

            let headerData = data[headerStart..<headerEnd.lowerBound]
            let headerString = String(data: headerData, encoding: .utf8) ?? ""

            let bodyStart = headerEnd.upperBound

            // Find end of this part (next delimiter)
            let bodyEnd: Data.Index
            if let nextDelimiter = data.range(of: delimiter, in: bodyStart..<data.endIndex) {
                // Body ends before the CRLF that precedes the next delimiter
                let candidateEnd = nextDelimiter.lowerBound
                if candidateEnd >= crlf.count,
                   data[candidateEnd - crlf.count..<candidateEnd] == crlf {
                    bodyEnd = candidateEnd - crlf.count
                } else {
                    bodyEnd = candidateEnd
                }
                searchStart = nextDelimiter.lowerBound
            } else {
                // Last part — end before end delimiter
                if let endRange = data.range(of: endDelimiter, in: bodyStart..<data.endIndex) {
                    let candidateEnd = endRange.lowerBound
                    if candidateEnd >= crlf.count,
                       data[candidateEnd - crlf.count..<candidateEnd] == crlf {
                        bodyEnd = candidateEnd - crlf.count
                    } else {
                        bodyEnd = candidateEnd
                    }
                } else {
                    bodyEnd = data.endIndex
                }
                searchStart = data.endIndex
            }

            // Parse headers
            let name = extractHeaderValue(from: headerString, header: "Content-Disposition", param: "name")
            let filename = extractHeaderValue(from: headerString, header: "Content-Disposition", param: "filename")
            let contentType = extractSimpleHeader(from: headerString, header: "Content-Type")

            if let name {
                parts.append(Part(
                    name: name,
                    filename: filename,
                    contentType: contentType,
                    data: Data(data[bodyStart..<bodyEnd])
                ))
            }
        }

        return parts
    }

    /// Extract boundary string from Content-Type header value.
    /// e.g. "multipart/form-data; boundary=----WebKitFormBoundary" → "----WebKitFormBoundary"
    public static func extractBoundary(from contentType: String) -> String? {
        let parts = contentType.components(separatedBy: ";")
        for part in parts {
            let trimmed = part.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("boundary=") {
                var boundary = String(trimmed.dropFirst("boundary=".count))
                // Remove surrounding quotes if present
                if boundary.hasPrefix("\"") && boundary.hasSuffix("\"") {
                    boundary = String(boundary.dropFirst().dropLast())
                }
                return boundary
            }
        }
        return nil
    }

    // MARK: - Private Helpers

    private static func extractHeaderValue(from headers: String, header: String, param: String) -> String? {
        for line in headers.components(separatedBy: "\r\n") {
            let lower = line.lowercased()
            if lower.hasPrefix(header.lowercased() + ":") {
                // Look for param="value"
                let pattern = param + "=\""
                if let range = line.range(of: pattern, options: .caseInsensitive) {
                    let start = range.upperBound
                    if let end = line[start...].firstIndex(of: "\"") {
                        return String(line[start..<end])
                    }
                }
                // Also try without quotes: param=value
                let patternNoQuote = param + "="
                if let range = line.range(of: patternNoQuote, options: .caseInsensitive) {
                    let start = range.upperBound
                    let remaining = line[start...]
                    let value = remaining.prefix(while: { $0 != ";" && $0 != " " && $0 != "\r" })
                    if !value.isEmpty {
                        return String(value)
                    }
                }
            }
        }
        return nil
    }

    private static func extractSimpleHeader(from headers: String, header: String) -> String? {
        for line in headers.components(separatedBy: "\r\n") {
            if line.lowercased().hasPrefix(header.lowercased() + ":") {
                let value = line.dropFirst(header.count + 1).trimmingCharacters(in: .whitespaces)
                return value.isEmpty ? nil : value
            }
        }
        return nil
    }
}
