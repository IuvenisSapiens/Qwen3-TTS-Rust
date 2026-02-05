use crate::assets_manager::Assets;
use crate::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use crate::models::onnx::{AudioDecoder, AudioEncoder, SpeakerEncoder};
use crate::tts::prompt::PromptBuilder;
use crate::utils::cache;
use crate::utils::tokenizer::Tokenizer;
use crate::AudioSample;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Main TTS Engine Struct
pub struct TtsEngine {
    assets: Assets,
    tokenizer: Tokenizer,
    // Models
    encoder: AudioEncoder,
    speaker_encoder: SpeakerEncoder,
    audio_decoder: AudioDecoder, // We need to clone it or use it. But it's not Clone? It is loaded from file.
    // We will keep one instance and maybe clone it if needed or use internal mutability?
    // AudioDecoder::load returns Self.
    // Llama
    talker_model: LlamaModel,
    predictor_model: LlamaModel,
    // We create contexts per generation or keep them?
    // Creating context is cheap(ish) but keeping them is better for cache?
    // The example recreated them? No, example created them once.
    // So we keep contexts.
    // LlamaContext is not thread safe for parallel usage, but we assume single threaded usage for now.
    // We need RefCell or Mutex if we want immutable &self for generate?
    // Let's make generate take &mut self.
    talker_ctx: LlamaContext,
    predictor_ctx: LlamaContext,

    // Config
    model_dir: PathBuf,
}

impl TtsEngine {
    /// Load all models and assets from the specified directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        println!("Loading TtsEngine from: {:?}", model_dir);

        // 1. Assets
        let assets =
            Assets::load(model_dir).map_err(|e| format!("Failed to load assets: {}", e))?;

        // 2. Tokenizer
        let tokenizer =
            Tokenizer::load(model_dir).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // 3. ONNX Models
        let encoder = AudioEncoder::load(
            &model_dir
                .join("qwen3_tts_codec_encoder.onnx")
                .to_string_lossy(),
        )
        .map_err(|e| format!("Failed to load AudioEncoder: {}", e))?;

        let speaker_encoder = SpeakerEncoder::load(
            &model_dir
                .join("qwen3_tts_speaker_encoder.onnx")
                .to_string_lossy(),
        )
        .map_err(|e| format!("Failed to load SpeakerEncoder: {}", e))?;

        let audio_decoder =
            AudioDecoder::load(&model_dir.join("qwen3_tts_decoder.onnx").to_string_lossy())
                .map_err(|e| format!("Failed to load AudioDecoder: {}", e))?;

        // 4. Initialize Llama Backend
        crate::load_backends();
        crate::init_backend();

        // 5. Load GGUF Models
        let talker_path = model_dir.join("qwen3_tts_talker-q4km.gguf");
        let predictor_path = model_dir.join("qwen3_tts_predictor-q4km.gguf");

        let talker_model = LlamaModel::load(&talker_path, 99)
            .map_err(|e| format!("Failed to load Talker: {}", e))?;

        let predictor_model = LlamaModel::load(&predictor_path, 99)
            .map_err(|e| format!("Failed to load Predictor: {}", e))?;

        // 6. Create Contexts
        let talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, -1)
            .map_err(|e| format!("Failed to create Talker context: {}", e))?;

        let predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, 4)
            .map_err(|e| format!("Failed to create Predictor context: {}", e))?;

        println!("TtsEngine loaded successfully.");

        Ok(Self {
            assets,
            tokenizer,
            encoder,
            speaker_encoder,
            audio_decoder,
            talker_model,
            predictor_model,
            talker_ctx,
            predictor_ctx,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Generate speech from text using a reference audio.
    pub fn generate(
        &mut self,
        text: &str,
        ref_audio_path: impl AsRef<Path>,
        ref_text: &str,
    ) -> Result<AudioSample, String> {
        let ref_audio_path = ref_audio_path.as_ref();

        // 1. Process Reference Audio
        let (ref_codes, spk_emb) = self.process_reference(ref_audio_path)?;

        // 2. Build Prompt
        // lang_id = 2055 (Chinese) hardcoded for now or parameterize later
        let ref_text_ids = self.tokenizer.encode(ref_text);
        let ref_codes_i32: Vec<i32> = ref_codes.iter().map(|&c| c as i32).collect();

        let prompt_data = PromptBuilder::build_clone_prompt(
            text,
            &self.tokenizer,
            &self.assets,
            &ref_codes_i32,
            &ref_text_ids,
            &spk_emb,
            2055,
        );

        self.run_inference(prompt_data)
    }

    /// Process reference audio to get codes and speaker embedding, using cache if available.
    fn process_reference(&mut self, audio_path: &Path) -> Result<(Vec<i64>, Vec<f32>), String> {
        let cache_path = audio_path.with_extension("cache");
        if cache_path.exists() {
            if let Ok((c, e)) = cache::load_cache(&cache_path) {
                return Ok((c, e));
            }
        }

        let audio = AudioSample::load_wav(audio_path)
            .map_err(|e| format!("Failed to load audio: {}", e))?;

        let ref_codes = self
            .encoder
            .encode(&audio.samples)
            .map_err(|e| format!("Audio encode failed: {}", e))?;
        let spk_emb = self
            .speaker_encoder
            .encode(&audio.samples)
            .map_err(|e| format!("Speaker extraction failed: {}", e))?;

        let _ = cache::save_cache(&cache_path, &ref_codes, &spk_emb);

        Ok((ref_codes, spk_emb))
    }

    // --- Helpers ---

    fn qwen3_position(start: i32, len: usize) -> Vec<i32> {
        let mut pos = Vec::with_capacity(len * 4);
        let range: Vec<i32> = (start..start + len as i32).collect();
        pos.extend_from_slice(&range); // Temporal
        pos.extend_from_slice(&range); // Height
        pos.extend_from_slice(&range); // Width
        pos.extend(std::iter::repeat_n(0, len)); // Channel
        pos
    }

    fn normal_position(cur_pos: usize, n_tokens: usize) -> Vec<i32> {
        (0..n_tokens).map(|i| (cur_pos + i) as i32).collect()
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
    ) -> Result<AudioSample, String> {
        // 1. Build Prompt
        let ref_text_ids = self.tokenizer.encode(&voice.ref_text);
        let ref_codes_i32: Vec<i32> = voice.audio_codes.iter().map(|&c| c as i32).collect();

        let prompt_data = PromptBuilder::build_clone_prompt(
            text,
            &self.tokenizer,
            &self.assets,
            &ref_codes_i32,
            &ref_text_ids,
            &voice.speaker_embedding,
            2055,
        );

        // Use existing internal logic to generate
        self.run_inference(prompt_data)
    }

    // Refactor generation logic into a private helper to avoid code duplication
    fn run_inference(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
    ) -> Result<AudioSample, String> {
        let n_tokens_prompt = prompt_data.embd.len();
        let prompt_embeds_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
        let talker_embd = self.talker_model.n_embd;
        let predictor_embd = self.predictor_model.n_embd;

        // Talker Prefill
        let mut talker_batch = LlamaBatch::new(4096, talker_embd, 1, 4);
        let pos_arr = Self::qwen3_position(0, n_tokens_prompt);
        talker_batch.set_embd(&prompt_embeds_flat, &pos_arr, 0);

        self.talker_ctx
            .decode(&mut talker_batch)
            .map_err(|e| format!("Talker prefill failed: {}", e))?;

        // Generation Loop
        let n_steps = 50;
        let mut all_codes: Vec<i32> = Vec::new();
        let mut cur_pos = n_tokens_prompt;

        // Hoisted resources
        let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);
        let predictor_sampler = LlamaSampler::greedy(self.predictor_model.n_vocab);
        let talker_sampler = LlamaSampler::new(self.talker_model.n_vocab, 0.5, 50, 1.0, 12345);

        let (tx, rx) = std::sync::mpsc::channel::<(Vec<i64>, bool)>();
        let decoder_model_path = self
            .model_dir
            .join("qwen3_tts_decoder.onnx")
            .to_string_lossy()
            .to_string();

        let decoder_handle = std::thread::spawn(move || {
            let mut local_decoder = match AudioDecoder::load(&decoder_model_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to load decoder in thread: {}", e);
                    return Vec::new();
                }
            };
            let mut full_audio = Vec::new();
            let mut state = AudioDecoder::create_state();
            let mut code_buffer: Vec<i64> = Vec::with_capacity(64);

            while let Ok((codes, is_final)) = rx.recv() {
                code_buffer.extend(codes);
                if code_buffer.len() >= 64 || is_final {
                    let safe_codes: Vec<i64> = code_buffer.iter().map(|&c| c.min(2047)).collect();
                    if let Ok(samples) = local_decoder.decode(&safe_codes, &mut state, is_final) {
                        full_audio.extend(samples);
                    }
                    code_buffer.clear();
                }
                if is_final {
                    break;
                }
            }
            full_audio
        });

        for step in 0..n_steps {
            // Talker
            let sample_idx = if cur_pos == n_tokens_prompt {
                (n_tokens_prompt - 1) as i32
            } else {
                0
            };
            let code_0 = talker_sampler.sample(&self.talker_ctx, sample_idx, None, Some(2160));

            if code_0 == self.talker_model.eos_token
                || code_0 == 2150
                || code_0 == 151673
                || code_0 == 151643
                || code_0 == 151645
            {
                break;
            }
            let code_0_i32 = code_0 as i32;
            all_codes.push(code_0_i32);

            // Predictor
            let emb_idx = if step == 0 { n_tokens_prompt - 1 } else { 0 };
            let m_hidden = self.talker_ctx.get_embedding_at(emb_idx).to_vec();

            let m_h_1024 = self.assets.project(&m_hidden);
            let code_0_1024 = self.assets.get_codec_embedding_1024(0, code_0_i32);

            let mut predictor_input = Vec::with_capacity(2 * predictor_embd);
            predictor_input.extend_from_slice(&m_h_1024);
            predictor_input.extend_from_slice(&code_0_1024);

            self.predictor_ctx.clear_kv_cache();
            predictor_batch.clear();
            let pred_pos = Self::normal_position(0, 2);
            predictor_batch.set_embd(&predictor_input, &pred_pos, 0);

            self.predictor_ctx
                .decode(&mut predictor_batch)
                .map_err(|e| format!("Predictor prefill failed: {}", e))?;

            let mut step_embeds_2048: Vec<Vec<f32>> = Vec::new();
            step_embeds_2048.push(self.assets.get_codec_embedding(0, code_0_i32));

            for q in 1..16 {
                let start_offset = (q - 1) * 2048;
                let end_offset = q * 2048;
                let sampled = predictor_sampler.sample(
                    &self.predictor_ctx,
                    0,
                    Some(start_offset),
                    Some(end_offset),
                );
                let code_q = sampled - start_offset as i32;
                all_codes.push(code_q);

                let emb = self.assets.get_codec_embedding(q, code_q);
                step_embeds_2048.push(emb.to_vec());

                if q < 15 {
                    let next_embed_1024 = self.assets.get_codec_embedding_1024(q, code_q);
                    let next_pos = Self::normal_position(q + 1, 1);
                    predictor_batch.clear();
                    predictor_batch.set_embd(&next_embed_1024, &next_pos, 0);
                    self.predictor_ctx
                        .decode(&mut predictor_batch)
                        .map_err(|e| format!("Predictor decode failed: {}", e))?;
                }
            }

            let frame_codes: Vec<i64> = all_codes
                .iter()
                .rev()
                .take(16)
                .rev()
                .map(|&c| c as i64)
                .collect();
            let _ = tx.send((frame_codes, false));

            let mut feedback = vec![0.0f32; 2048];
            for embed in &step_embeds_2048 {
                for (i, val) in embed.iter().enumerate() {
                    feedback[i] += val;
                }
            }
            for (i, val) in self.assets.tts_pad.iter().enumerate() {
                feedback[i] += val;
            }
            feedback.resize(talker_embd, 0.0);

            let talker_pos = Self::qwen3_position(cur_pos as i32, 1);
            talker_batch.clear();
            talker_batch.set_embd(&feedback, &talker_pos, 0);

            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker step failed: {}", e))?;

            cur_pos += 1;
        }

        let _ = tx.send((Vec::new(), true));
        drop(tx);

        let audio_samples = decoder_handle
            .join()
            .map_err(|_| "Decoder thread panicked".to_string())?;

        Ok(AudioSample {
            samples: audio_samples,
            sample_rate: 24000,
            channels: 1,
        })
    }
}
