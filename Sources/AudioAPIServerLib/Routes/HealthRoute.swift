import Foundation
import Hummingbird
import HTTPTypes

// MARK: - Health & Model Listing

public enum HealthRoute {

    public static func register(on router: Router<some RequestContext>, models: ModelContext) {
        // GET /health
        router.get("health") { _, _ -> Response in
            let status = await models.status
            let health = HealthResponse(status: "ok", models: status)
            return jsonResponse(health)
        }

        // GET /v1/models — OpenAI-compatible model listing
        router.get("v1/models") { _, _ -> Response in
            let status = await models.status
            var modelList: [ModelInfo] = []

            if status.asr {
                modelList.append(ModelInfo(id: "qwen3-asr", object: "model", owned_by: "local"))
            }
            if status.tts, let engine = status.ttsEngine {
                modelList.append(ModelInfo(id: engine, object: "model", owned_by: "local"))
            }
            if status.aligner {
                modelList.append(ModelInfo(id: "qwen3-forced-aligner", object: "model", owned_by: "local"))
            }
            if status.personaPlex {
                modelList.append(ModelInfo(id: "personaplex", object: "model", owned_by: "local"))
            }
            if status.vad {
                modelList.append(ModelInfo(id: "silero-vad", object: "model", owned_by: "local"))
            }

            return jsonResponse(ModelListResponse(object: "list", data: modelList))
        }
    }
}
