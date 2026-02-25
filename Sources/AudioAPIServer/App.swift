import Foundation
import ArgumentParser
import AudioAPIServerLib
import Hummingbird
import Logging

@main
struct AudioAPIServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio-api",
        abstract: "Speech API server (OpenAI-compatible)"
    )

    @Option(name: .long, help: "Host to bind to")
    var host: String = "0.0.0.0"

    @Option(name: .long, help: "Port to listen on")
    var port: Int = 8080

    @Option(name: .long, help: "ASR model: 0.6B, 1.7B, or full HuggingFace model ID")
    var asrModel: String = "1.7B"

    @Option(name: .long, help: "TTS engine: qwen3 (default), cosyvoice, or none")
    var ttsEngine: String = "qwen3"

    @Option(name: .long, help: "TTS model: base, custom, or full HuggingFace model ID")
    var ttsModel: String = "base"

    @Flag(name: .long, help: "Enable forced alignment endpoint")
    var enableAligner: Bool = false

    @Flag(name: .long, help: "Enable PersonaPlex speech-to-speech endpoint (5.3 GB download)")
    var enablePersonaPlex: Bool = false

    @Flag(name: .long, help: "Disable VAD-based silence trimming (enabled by default)")
    var disableVAD: Bool = false

    func run() async throws {
        let logger = Logger(label: "audio-api")

        // Load models
        logger.info("Loading models...")
        let models = try await ModelContext.load(
            asrModelSpec: asrModel,
            ttsEngine: ttsEngine,
            ttsModelSpec: ttsModel,
            enableAligner: enableAligner,
            enablePersonaPlex: enablePersonaPlex,
            enableVAD: !disableVAD,
            logger: logger
        )
        logger.info("All models loaded.")

        // Build router
        let router = Router()

        // Register routes
        HealthRoute.register(on: router, models: models)
        TranscriptionRoute.register(on: router, models: models)
        SpeechRoute.register(on: router, models: models)
        AlignmentRoute.register(on: router, models: models)
        RespondRoute.register(on: router, models: models)
        VADRoute.register(on: router, models: models)

        // Start server
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        logger.info("Starting server on \(host):\(port)")
        try await app.runService()
    }
}
