# AGENTS.md — AI Agent Coding Guidelines

## Principles

1. **SOLID** — Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.
2. **KISS** — Keep It Simple, Stupid. Prefer clear, direct code over clever abstractions.
3. **YAGNI** — You Aren't Gonna Need It. Don't add code, abstractions, or config until there's a concrete need.

## Project Conventions

- **One file per concern.** Route handlers, models, utilities each get their own file.
- **Protocol extensions live in `+Protocols.swift`** (e.g. `Qwen3ASR+Protocols.swift`).
- **Weight loading lives in `WeightLoading.swift`** per module.
- **Shared types and protocols live in `AudioCommon`** — all model targets depend on it.
- **No cross-dependencies between model targets** (`Qwen3ASR`, `Qwen3TTS`, `CosyVoiceTTS`, `PersonaPlex` are independent).

## Swift Concurrency

- Models are `AnyObject` (not `Sendable`). Use a Swift `actor` to provide thread-safe access.
- Route handlers are `@Sendable` closures — access models only through the actor.
- Prefer structured concurrency (`async/await`, `AsyncThrowingStream`) over callbacks or Combine.
- Never block the cooperative thread pool with synchronous model inference — dispatch appropriately.

## Error Handling

- Throw typed errors. Let the HTTP framework convert them to status codes.
- Use `guard` for early returns. Avoid deeply nested `if-let` chains.
- Log errors with context (model name, request ID) at the call site.

## API Design

- Follow OpenAI API conventions for compatible endpoints (`/v1/audio/transcriptions`, `/v1/audio/speech`).
- Use multipart/form-data for audio uploads, JSON for text-only requests.
- Return JSON responses with `Codable` structs — no manual string building.
- Custom endpoints (alignment, respond) use the `/v1/audio/` prefix for consistency.

## Code Style

- Use `// MARK: -` sections to organize files.
- Keep functions under ~40 lines. Extract helpers when logic grows.
- Prefer `let` over `var`. Prefer value types unless reference semantics are required.
- Name files after the primary type they contain.
- No force-unwrapping (`!`) outside of tests.

## Testing

- Unit tests go in `Tests/<ModuleName>Tests/`.
- Use protocol-based mocks for model dependencies in API server tests.
- Test route handlers with the framework's test client (e.g. `HummingbirdTesting`).

## What NOT to Do

- Don't add auth, rate limiting, or HTTPS to the local API server — use a reverse proxy.
- Don't abstract over model loading — each model has unique init params.
- Don't create wrapper protocols around existing `AudioCommon` protocols.
- Don't add dependencies without a concrete, immediate need.

## Mistake Prevention

- Treat `AGENTS.md` as a living contract. When a mistake repeats or causes rework, update this file in the same PR.
- Add prevention rules as specific, testable bullets (what happened, what to do instead, and where it applies).
- Prefer narrow rules over broad policy changes; avoid duplicating existing guidance.
- Before finalizing work, run a short self-check against relevant `AGENTS.md` rules and confirm compliance in the summary.
- If a rule is ambiguous or conflicts with existing guidance, clarify and edit `AGENTS.md` first, then implement code changes.

## Recorded Mistakes

These are real mistakes made during API server development. Each rule prevents a specific recurrence.

### Swift Package Manager

- **Executables can't be imported by tests.** Any code you want to test must live in a library target. Always split into `<Name>Lib` (library) + `<Name>` (executable) from the start. The executable contains only the `@main` entry point.

- **All public API in a library target needs explicit `public`.** Types default to `internal`: routes, helpers, response structs. Add `public` to every type and method in a library target that must be used from another module.

- **Binary location on macOS Apple Silicon is `.build/arm64-apple-macosx/<config>/`**, not `.build/<config>/`. Use `find .build -name "audio-api" -type f` to locate it when unsure.

- **`swift build -c release --target X` reports "Build complete!" even when only the compile phase runs.** Run `swift build -c release` without `--target` to guarantee the binary is fully linked.

### Hummingbird 2

- **`Application<Router<BasicRequestContext>>` is a wrong return type.** `Router` is not an `HTTPResponder`; Hummingbird wraps it into `RouterResponder` internally. Use `some ApplicationProtocol` as the function return type when returning `Application(router:)`.

- **Background terminal shows no Hummingbird server logs.** Server startup and request logs are not visible in a background agent terminal. Use `curl /health` to verify the server is running.

### Swift Concurrency

- **`var body = Data()` captured by an `async` closure triggers a Swift 6 warning.** After all `body.append()` mutations, add `let body = body` before the async block to create an immutable capture.

- **Actor `let` properties are nonisolated in Swift 5.9+.** No `await` is needed to read `let` properties on an actor; adding `await` generates a "no async operations performed" warning. Access `let` properties directly.

### API / Codable

- **Swift `Codable` omits `nil` Optional keys (not `null`).** A `nil` optional is completely absent from JSON output. If OpenAI spec compliance requires `null`, use a custom encoder or a non-optional with a sentinel value.

### Dependency Injection / Testability

- **State holders (e.g. `ModelContext`) must store protocol types, not concrete types.** Storing `Qwen3ASRModel` makes the mock-injection impossible. Store `any SpeechRecognitionModel` (and similar). Violating this forces concrete-type tests which require model downloads.

### VAD Pipeline

- **VAD trimming ≠ VAD silence removal.** `audio[first.start ..< last.end]` only removes leading/trailing silence — all inter-segment gaps remain, wasting inference time. For transcription, always **concatenate** every detected segment (`vadConcatenatedAudio` / `ModelContext.concatenateSpeech`). Only use trim+offset when timestamps must map back to original audio (forced alignment).

- **One VAD function, two callers, two different needs.** `TranscribeCommand`/`TranscriptionRoute`/`RespondRoute` need segment concatenation. `AlignCommand`/`AlignmentRoute` need trim+offset. Keep these as separate functions (`concatenateSpeech` and `trimSilence`) — do not merge or alias them.
