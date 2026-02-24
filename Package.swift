// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Qwen3Speech",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "Qwen3ASR",
            targets: ["Qwen3ASR"]
        ),
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
        .library(
            name: "AudioCommon",
            targets: ["AudioCommon"]
        ),
        .library(
            name: "CosyVoiceTTS",
            targets: ["CosyVoiceTTS"]
        ),
        .executable(
            name: "qwen3-asr-cli",
            targets: ["Qwen3ASRCLI"]
        ),
        .executable(
            name: "qwen3-tts-cli",
            targets: ["Qwen3TTSCLI"]
        ),
        .executable(
            name: "cosyvoice-tts-cli",
            targets: ["CosyVoiceTTSCLI"]
        ),
        .library(
            name: "PersonaPlex",
            targets: ["PersonaPlex"]
        ),
        .executable(
            name: "personaplex-cli",
            targets: ["PersonaPlexCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")
    ],
    targets: [
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3TTS",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "CosyVoiceTTS",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "Qwen3ASRCLI",
            dependencies: [
                "Qwen3ASR",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "Qwen3TTSCLI",
            dependencies: [
                "Qwen3TTS",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "CosyVoiceTTSCLI",
            dependencies: [
                "CosyVoiceTTS",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .target(
            name: "PersonaPlex",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "PersonaPlexCLI",
            dependencies: [
                "PersonaPlex",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "PersonaPlexTests",
            dependencies: ["PersonaPlex", "AudioCommon", "Qwen3ASR"]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: ["Qwen3ASR", "AudioCommon"],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        ),
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS", "Qwen3ASR", "AudioCommon"]
        ),
        .testTarget(
            name: "CosyVoiceTTSTests",
            dependencies: ["CosyVoiceTTS", "AudioCommon"]
        )
    ]
)
