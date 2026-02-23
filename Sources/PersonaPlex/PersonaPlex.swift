import Foundation
import MLX
import MLXNN
import Qwen3Common

// MARK: - PersonaPlex Model

public final class PersonaPlexModel: Module {
    public let cfg: PersonaPlexConfig

    @ModuleInfo public var temporal: TemporalTransformer
    @ModuleInfo public var depformer: Depformer
    public let mimi: Mimi

    public init(cfg: PersonaPlexConfig = .default) {
        self.cfg = cfg
        self._temporal = ModuleInfo(wrappedValue: TemporalTransformer(cfg: cfg.temporal))
        self._depformer = ModuleInfo(wrappedValue: Depformer(cfg: cfg.depformer, temporalDim: cfg.temporal.dim))
        self.mimi = Mimi(cfg: cfg.mimi)
    }

    // MARK: - Offline Inference

    /// Process user audio and generate response audio.
    ///
    /// Stream layout (17 streams):
    ///   - Stream 0:    text (agent inner monologue)
    ///   - Streams 1-8: agent audio (8 codebooks, predicted by depformer)
    ///   - Streams 9-16: user audio (8 codebooks from Mimi encoder)
    ///
    /// Prompt sequence before user audio:
    ///   1. Voice prompt (pre-computed embeddings fed through temporal transformer)
    ///   2. 0.5s silence spacer
    ///   3. Text system prompt (SentencePiece tokens, one per frame)
    ///   4. 0.5s silence spacer
    ///   5. User audio frames, then autoregressive generation
    ///
    /// - Parameters:
    ///   - userAudio: [numSamples] float array of 24kHz mono audio
    ///   - voice: voice preset for the agent
    ///   - systemPromptTokens: SentencePiece-tokenized system prompt (nil = default)
    ///   - maxSteps: maximum generation steps (at 12.5 Hz)
    ///   - verbose: print timing info
    /// - Returns: [numSamples] float array of 24kHz response audio
    public func respond(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        systemPromptTokens: [Int32]? = nil,
        maxSteps: Int = 500,
        verbose: Bool = false
    ) -> [Float] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Encode user audio with Mimi
        let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
        let userCodes = mimi.encode(audioArray)  // [1, numCodebooks, T]
        eval(userCodes)

        let userFrameCount = userCodes.shape[2]
        if verbose {
            let encTime = CFAbsoluteTimeGetCurrent() - startTime
            print("  Mimi encode: \(String(format: "%.2f", encTime))s, \(userFrameCount) frames")
        }

        // 2. Load voice prompt embeddings + cache
        let voiceStart = CFAbsoluteTimeGetCurrent()
        let voiceEmbeddings: MLXArray?
        let voiceCache: MLXArray?  // [1, 17, CT] ring buffer with voice prompt tokens
        do {
            let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let voiceDir = modelDir.appendingPathComponent("voices")
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            if FileManager.default.fileExists(atPath: voiceFile.path) {
                let weights = try MLX.loadArrays(url: voiceFile)
                voiceEmbeddings = weights["embeddings"]  // [T, 1, 1, dim]
                voiceCache = weights["cache"]             // [1, 17, maxDelay+3]
            } else {
                voiceEmbeddings = nil
                voiceCache = nil
            }
        } catch {
            voiceEmbeddings = nil
            voiceCache = nil
        }

        let voiceFrameCount = voiceEmbeddings?.shape[0] ?? 0
        let silenceFrameCount = Int(0.5 * cfg.mimi.frameRate)  // 0.5s silence = ~6 frames
        let textPromptTokens = systemPromptTokens ?? TemporalTransformerConfig.defaultSystemPromptTokens
        let textPromptLen = textPromptTokens.count

        if verbose {
            let voiceTime = CFAbsoluteTimeGetCurrent() - voiceStart
            print("  Voice prompt: \(voiceFrameCount) frames, text prompt: \(textPromptLen) tokens (\(String(format: "%.2f", voiceTime))s)")
        }

        // 3. Reset caches
        temporal.resetCache()
        mimi.resetState()

        // Total steps: voice + silence1 + text_prompt + silence2 + user audio + gen
        // The reference model skips offset=0 in prepare_step_input (writes initial tokens,
        // no forward pass). The first real forward pass uses voice[0] at position 0.
        // So RoPE offset = step (matching PyTorch's transformer offset exactly).
        let promptLen = voiceFrameCount + silenceFrameCount + textPromptLen + silenceFrameCount
        let prefillLen = promptLen + userFrameCount
        let delays = cfg.delays
        let maxDelay = cfg.maxDelay
        let numStreams = cfg.numStreams
        let nQ = cfg.temporal.nQ
        let totalLen = prefillLen + maxSteps + maxDelay + 3

        // 4. Initialize token cache
        // Stream 0 = text, streams 1-8 = agent audio, streams 9-16 = user audio
        var tokenCache = [[Int32]](repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // --- Phase 1: Voice prompt tokens ---
        // During voice prompt: text=PAD, agent audio=silence tokens, user audio=sine tokens
        for t in 0..<voiceFrameCount {
            // Text: padding token
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
            // Agent audio: silence tokens (streams 1-8)
            for cb in 0..<nQ {
                let streamIdx = 1 + cb
                tokenCache[streamIdx][t + delays[streamIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            // User audio: sine tokens (streams 9-16)
            for cb in 0..<nQ {
                let streamIdx = 1 + nQ + cb
                tokenCache[streamIdx][t + delays[streamIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
        }

        // --- Apply voice prompt cache ---
        // The voice .safetensors contains a ring buffer snapshot [1, 17, CT] with the actual
        // tokens that were in the delay buffer after voice prompt creation. This includes real
        // voice audio tokens for agent streams (not silence). We overwrite the last few
        // positions of the voice prompt phase so subsequent reads get correct token values.
        if let vc = voiceCache, voiceFrameCount > 0 {
            let CT = maxDelay + 3  // ring buffer size (4 for PersonaPlex)
            eval(vc)
            // Map ring buffer positions to flat array positions.
            // Python state.offset after voice prompt = V+1 (V voice steps + 1 init skip).
            // Python reads ring[(state.offset-1) % CT]; Swift reads tokenCache[step-1].
            // With RoPE offset=step, the transformer position matches Python's internal offset.
            // Mapping: tokenCache[flatPos] = cache[s, (flatPos + 1) % CT].
            for s in 0..<numStreams {
                let d = delays[s]
                for k in 0...d {
                    let flatPos = voiceFrameCount - 1 + k
                    let ringPos = (voiceFrameCount + k) % CT
                    if flatPos >= 0 && flatPos < totalLen {
                        tokenCache[s][flatPos] = Int32(vc[0, s, ringPos].item(Float.self))
                    }
                }
            }
        }

        // --- Phase 2: Silence spacer 1 ---
        var pos = voiceFrameCount
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 3: Text prompt ---
        // Text stream gets the actual system prompt tokens (one per frame)
        // Agent audio = silence tokens, user audio = sine tokens
        for t in 0..<textPromptLen {
            tokenCache[0][pos + delays[0]] = textPromptTokens[t]
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 4: Silence spacer 2 ---
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 5: User audio ---
        // Fill user audio into streams 9-16, agent audio = silence, text = PAD
        let userCodesArr = userCodes.asType(.int32)
        eval(userCodesArr)
        for t in 0..<userFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            // Agent audio: silence during user turn
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            // User audio from Mimi encoder
            for cb in 0..<min(nQ, userCodes.shape[1]) {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = userCodesArr[0, cb, t].item(Int32.self)
            }
            pos += 1
        }

        // 5. Autoregressive generation (full-duplex: agent generates while user speaks)
        //
        // Phase layout:
        //   steps 0..<voiceFrameCount:    voice prompt (embeddings only, no generation)
        //   steps voiceFrameCount..<promptLen: silence + text prompt + silence (forward, no generation)
        //   steps promptLen..<prefillLen:  user audio + simultaneous agent generation
        //   steps prefillLen..<prefillLen+maxSteps: post-user generation (user=sine)
        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let genStart = CFAbsoluteTimeGetCurrent()
        let generationStartStep = promptLen  // Start generating when user audio begins

        for step in 0..<(prefillLen + maxSteps) {
            // --- Voice prompt: use pre-computed embeddings ---
            if step < voiceFrameCount, let voiceEmb = voiceEmbeddings {
                let emb = voiceEmb[step].reshaped([1, 1, cfg.temporal.dim])
                temporal.forwardEmbedding(emb, offset: step)
                continue
            }

            // Build input tokens for this step.
            // Original Moshi reads (offset - 1) % CT — the PREVIOUS step's token.
            // At step 0 there is no previous step, so read -1 (will be mapped to initial token).
            let readIdx = step > 0 ? step - 1 : 0
            let textTok = step > 0 ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = step > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                // Pass -1 through — TemporalTransformer zero-masks negative tokens
                audioTokenArrs.append(MLXArray([tok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            // Forward through temporal transformer
            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr,
                audioTokens: audioTokens,
                offset: step
            )
            eval(hidden, textLogits)

            // During silence/text prompt, skip sampling
            if step < generationStartStep {
                continue
            }

            // Sample text token
            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp,
                topK: cfg.sampling.textTopK
            )
            eval(textToken)

            // Build provided tokens for depformer conditioning during user audio.
            // In Python Moshi, the depformer uses real user audio tokens (from the cache
            // target position) as conditioning for user audio codebook steps (8-15).
            // This ensures the autoregressive chain within the depformer sees real audio
            // rather than its own potentially wrong predictions.
            var providedTokens: [Int32]? = nil
            if step < prefillLen {
                var provided = [Int32](repeating: -1, count: cfg.depformer.numSteps)
                for cb in 0..<nQ {
                    let userStreamIdx = 1 + nQ + cb
                    // Read from position `step` — matches Python's target_position.
                    // For delay-0 streams: current step's user audio.
                    // For delay-1 streams: previous step's user audio (written at pos-1+1=pos).
                    if step >= 0 && step < totalLen {
                        let tok = tokenCache[userStreamIdx][step]
                        if tok >= 0 {
                            provided[nQ + cb] = tok  // depformer step nQ+cb = user cb
                        }
                    }
                }
                providedTokens = provided
            }

            // Generate audio tokens via depformer (with per-codebook repetition penalty)
            let agentCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken,
                providedTokens: providedTokens
            ) { logits, cbIdx in
                let windowSize = cfg.sampling.repetitionWindow
                let history = Array(agentTokens[cbIdx].suffix(windowSize))
                return sampleTopKWithPenalty(
                    logits: logits,
                    temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK,
                    pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty
                )
            }
            eval(agentCodes)

            // Write generated tokens into cache at position `step` (NO delay).
            // Critical: In Python Moshi, process_transformer_output() writes ALL depformer
            // tokens at target_position = offset % CT (same position for all streams).
            // The delay is only used for external input (user audio, prompt tokens).
            // Writing at `step` (not `step + delays[k]`) ensures the next step immediately
            // reads the depformer's output for ALL streams, including delay-1 streams.
            let textVal = textToken[0].item(Int32.self)
            if step < totalLen {
                tokenCache[0][step] = textVal
            }

            // Agent audio tokens → streams 1-8
            let agentArr = agentCodes[0]  // [numSteps]
            for cb in 0..<nQ {
                let tok = agentArr[cb].item(Int32.self)
                if step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }

            // User audio predictions (depformer steps 8-15) → streams 9-16
            // During user audio: don't write (user audio already pre-filled, matches
            //   Python's provided=True preventing depformer overwrites).
            // After user audio: write depformer predictions (matches Python's provided=False
            //   allowing depformer writes).
            for cb in nQ..<cfg.depformer.numSteps {
                let tok = agentArr[cb].item(Int32.self)
                if step >= prefillLen && step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }
        }

        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let totalSteps = prefillLen + maxSteps
            let msPerStep = genTime / Double(totalSteps) * 1000
            print("  Generation: \(String(format: "%.2f", genTime))s, \(String(format: "%.1f", msPerStep))ms/step (\(totalSteps) steps, \(maxSteps) gen)")
        }

        // 6. Decode agent tokens with Mimi
        // Only use first nQ (8) codebooks — the depformer generates dep_q=16 tokens per step,
        // but only the first 8 are agent audio codebooks (streams 1-8). The remaining 8 are
        // predictions for user audio codebooks (streams 9-16, unused for decoding).
        // Original PersonaPlex: mimi.set_num_codebooks(8); pcm = mimi.decode(tokens[:, 1:9])
        let decStart = CFAbsoluteTimeGetCurrent()
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return [] }

        let numDecodeCodebooks = nQ  // 8 (matching set_num_codebooks(8) in original)
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames),
                                   count: numDecodeCodebooks)
        for cb in 0..<numDecodeCodebooks {
            codeMatrix[cb] = agentTokens[cb]
        }

        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, numDecodeCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)  // [1, 1, numSamples]
        eval(decoded)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
        }

        // Extract audio samples
        let numSamples = decoded.shape[2]
        var samples = [Float](repeating: 0, count: numSamples)
        let flatDecoded = decoded.reshaped([numSamples])
        eval(flatDecoded)
        for i in 0..<numSamples {
            samples[i] = flatDecoded[i].item(Float.self)
        }

        if verbose {
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let audioDuration = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", totalTime))s, audio: \(String(format: "%.2f", audioDuration))s, RTF: \(String(format: "%.2f", totalTime / max(audioDuration, 0.001)))")
        }

        return samples
    }

    // MARK: - Diagnostic Info

    public struct DiagnosticInfo {
        public var textTokens: [Int32] = []
        public var agentTokensByCodebook: [[Int32]] = []
        public var hiddenStats: [(mean: Float, std: Float, min: Float, max: Float)] = []
        public var textLogitStats: [(topToken: Int32, topLogit: Float, entropy: Float)] = []
        public var inputTokenSnapshots: [[(stream: Int, token: Int32)]] = []
    }

    /// Same as respond() but captures diagnostic info for debugging.
    public func respondDiagnostic(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        systemPromptTokens: [Int32]? = nil,
        maxSteps: Int = 500
    ) -> (audio: [Float], diag: DiagnosticInfo) {
        var diag = DiagnosticInfo()

        let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
        let userCodes = mimi.encode(audioArray)
        eval(userCodes)
        let userFrameCount = userCodes.shape[2]

        let voiceEmbeddings: MLXArray?
        let voiceCache2: MLXArray?
        do {
            let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let voiceDir = modelDir.appendingPathComponent("voices")
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            if FileManager.default.fileExists(atPath: voiceFile.path) {
                let weights = try MLX.loadArrays(url: voiceFile)
                voiceEmbeddings = weights["embeddings"]
                voiceCache2 = weights["cache"]
            } else { voiceEmbeddings = nil; voiceCache2 = nil }
        } catch { voiceEmbeddings = nil; voiceCache2 = nil }

        let voiceFrameCount = voiceEmbeddings?.shape[0] ?? 0
        let silenceFrameCount = Int(0.5 * cfg.mimi.frameRate)
        let textPromptTokens = systemPromptTokens ?? TemporalTransformerConfig.defaultSystemPromptTokens
        let textPromptLen = textPromptTokens.count

        temporal.resetCache()
        mimi.resetState()

        let promptLen = voiceFrameCount + silenceFrameCount + textPromptLen + silenceFrameCount
        let prefillLen = promptLen + userFrameCount
        let delays = cfg.delays
        let numStreams = cfg.numStreams
        let nQ = cfg.temporal.nQ
        let totalLen = prefillLen + maxSteps + cfg.maxDelay + 3

        var tokenCache = [[Int32]](repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // Pre-fill phases (same as respond)
        for t in 0..<voiceFrameCount {
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let s = 1 + cb; tokenCache[s][t + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let s = 1 + nQ + cb; tokenCache[s][t + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
            }
        }
        // Apply voice prompt cache (same as respond)
        if let vc = voiceCache2, voiceFrameCount > 0 {
            let CT = cfg.maxDelay + 3
            eval(vc)
            for s in 0..<numStreams {
                let d = delays[s]
                for k in 0...d {
                    let flatPos = voiceFrameCount - 1 + k
                    let ringPos = (voiceFrameCount + k) % CT
                    if flatPos >= 0 && flatPos < totalLen {
                        tokenCache[s][flatPos] = Int32(vc[0, s, ringPos].item(Float.self))
                    }
                }
            }
        }
        var pos = voiceFrameCount
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        for t in 0..<textPromptLen {
            tokenCache[0][pos + delays[0]] = textPromptTokens[t]
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        let userCodesArr = userCodes.asType(.int32); eval(userCodesArr)
        for t in 0..<userFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<min(nQ, userCodes.shape[1]) {
                let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = userCodesArr[0, cb, t].item(Int32.self)
            }
            pos += 1
        }

        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let generationStartStep = promptLen

        for step in 0..<(prefillLen + maxSteps) {
            if step < voiceFrameCount, let voiceEmb = voiceEmbeddings {
                let emb = voiceEmb[step].reshaped([1, 1, cfg.temporal.dim])
                temporal.forwardEmbedding(emb, offset: step)
                continue
            }

            let readIdx = step > 0 ? step - 1 : 0
            let textTok = step > 0 ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = step > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                audioTokenArrs.append(MLXArray([tok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr, audioTokens: audioTokens, offset: step)
            eval(hidden, textLogits)

            // Capture hidden state stats for first 20 gen steps
            if step >= generationStartStep && diag.hiddenStats.count < 20 {
                let h = hidden.reshaped([-1])
                let hMean = MLX.mean(h).item(Float.self)
                let hStd = MLX.sqrt(MLX.mean((h - MLXArray(hMean)) * (h - MLXArray(hMean)))).item(Float.self)
                let hMin = MLX.min(h).item(Float.self)
                let hMax = MLX.max(h).item(Float.self)
                diag.hiddenStats.append((mean: hMean, std: hStd, min: hMin, max: hMax))

                // Text logit stats
                let tl = textLogits.squeezed(axes: [0, 1])  // [vocabSize]
                let topIdx = argMax(tl).item(Int32.self)
                let topVal = tl[Int(topIdx)].item(Float.self)
                let probs = softMax(tl, axis: -1)
                let logProbs = log(probs + MLXArray(Float(1e-10)))
                let ent = -(probs * logProbs).sum().item(Float.self)
                diag.textLogitStats.append((topToken: topIdx, topLogit: topVal, entropy: ent))

                // Snapshot input tokens
                var snapshot: [(stream: Int, token: Int32)] = [(0, textTok)]
                for stream in 1..<min(5, numStreams) {
                    let tok = step > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                    snapshot.append((stream, tok))
                }
                diag.inputTokenSnapshots.append(snapshot)
            }

            if step < generationStartStep { continue }

            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp, topK: cfg.sampling.textTopK)
            eval(textToken)
            let textVal = textToken[0].item(Int32.self)
            diag.textTokens.append(textVal)

            // Depformer conditioning (same as respond)
            var providedTokensDiag: [Int32]? = nil
            if step < prefillLen {
                var provided = [Int32](repeating: -1, count: cfg.depformer.numSteps)
                for cb in 0..<nQ {
                    let userStreamIdx = 1 + nQ + cb
                    if step >= 0 && step < totalLen {
                        let tok = tokenCache[userStreamIdx][step]
                        if tok >= 0 { provided[nQ + cb] = tok }
                    }
                }
                providedTokensDiag = provided
            }

            let agentCodes = depformer.generate(
                temporalHidden: hidden, textToken: textToken,
                providedTokens: providedTokensDiag
            ) { logits, cbIdx in
                let history = Array(agentTokens[cbIdx].suffix(cfg.sampling.repetitionWindow))
                return sampleTopKWithPenalty(
                    logits: logits, temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK, pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty)
            }
            eval(agentCodes)

            // Write at position `step` (no delay) — matches Python's target_position
            if step < totalLen { tokenCache[0][step] = textVal }
            let agentArr = agentCodes[0]
            for cb in 0..<nQ {
                let tok = agentArr[cb].item(Int32.self)
                if step < totalLen { tokenCache[1 + cb][step] = tok }
                agentTokens[cb].append(tok)
            }
            for cb in nQ..<cfg.depformer.numSteps {
                let tok = agentArr[cb].item(Int32.self)
                if step >= prefillLen && step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }
        }

        diag.agentTokensByCodebook = agentTokens

        // Decode
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return ([], diag) }
        let numDecodeCodebooks = nQ
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames), count: numDecodeCodebooks)
        for cb in 0..<numDecodeCodebooks { codeMatrix[cb] = agentTokens[cb] }
        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, numDecodeCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)
        eval(decoded)
        let numSamples = decoded.shape[2]
        var samples = [Float](repeating: 0, count: numSamples)
        let flatDecoded = decoded.reshaped([numSamples]); eval(flatDecoded)
        for i in 0..<numSamples { samples[i] = flatDecoded[i].item(Float.self) }

        return (samples, diag)
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PersonaPlexModel {
        // Download weights first to get config
        progressHandler?(0.05, "Downloading PersonaPlex weights...")
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let weightFiles = [
            "temporal.safetensors",
            "depformer.safetensors",
            "embeddings.safetensors",
            "mimi.safetensors",
            "config.json"
        ]

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: modelDir,
            additionalFiles: weightFiles
        ) { progress in
            progressHandler?(0.05 + progress * 0.5, "Downloading...")
        }

        // Read config.json to detect quantization settings
        var cfg = PersonaPlexConfig.default
        let configFile = modelDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configFile.path),
           let data = try? Data(contentsOf: configFile),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let quant = json["quantization"] as? [String: Any] {
            if let bits = quant["bits"] as? Int {
                cfg.temporal.bits = bits
            }
            if let groupSize = quant["group_size"] as? Int {
                cfg.temporal.groupSize = groupSize
            }
        } else {
            // No quantization section → BF16
            cfg.temporal.bits = 16
            cfg.temporal.groupSize = 1
        }
        let model = PersonaPlexModel(cfg: cfg)

        // Load weights
        progressHandler?(0.55, "Loading model weights...")
        try PersonaPlexWeightLoader.loadWeights(
            model: model,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.55 + progress * 0.25, status)
        }

        // Load Mimi
        progressHandler?(0.80, "Loading Mimi codec...")
        try PersonaPlexWeightLoader.loadMimi(
            model: model.mimi,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.80 + progress * 0.15, status)
        }

        model.train(false)
        progressHandler?(1.0, "PersonaPlex ready")
        return model
    }
}
