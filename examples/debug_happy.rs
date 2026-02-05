use qwen3_tts::assets::Assets;
use qwen3_tts::llama::{LlamaBatch, LlamaContext, LlamaModel};
use qwen3_tts::onnx::{AudioDecoder, AudioEncoder, SpeakerEncoder};
use qwen3_tts::prompt_builder::PromptBuilder;
use qwen3_tts::prompt_builder::{BOS, BOS_TOKEN, EOS_TOKEN, PAD, TEXT_AUDIO_MARKER};
use qwen3_tts::tokenizer::Tokenizer;
use qwen3_tts::AudioSample;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

mod cache {
    // Empty mod for now
}

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
    let mut result = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        result.push((cur_pos + i) as i32);
    }
    result
}

fn main() {
    println!("===========================================================");
    println!("  Debug Happy Instruction");
    println!("===========================================================");

    let model_dir = Path::new(r"c:\work\ai00-tts\thrd\Qwen3-TTS-GGUF\model-base");
    let clone_audio_path = r"c:\work\ai00-tts\clone.wav";

    // 1. Load Assets
    println!("Loading Assets...");
    let assets = Assets::load(model_dir).expect("Failed to load assets");
    let clone_audio = AudioSample::load_wav(clone_audio_path).expect("Failed to load audio");

    // 2. Load ONNX Models
    println!("Loading ONNX...");
    let mut encoder = AudioEncoder::load(
        &model_dir
            .join("qwen3_tts_codec_encoder.onnx")
            .to_string_lossy(),
    )
    .unwrap();
    let mut speaker_encoder = SpeakerEncoder::load(
        &model_dir
            .join("qwen3_tts_speaker_encoder.onnx")
            .to_string_lossy(),
    )
    .unwrap();
    let mut decoder =
        AudioDecoder::load(&model_dir.join("qwen3_tts_decoder.onnx").to_string_lossy()).unwrap();

    // 3. Encoder Reference
    println!("Encoding Reference Audio...");
    let ref_codes = encoder.encode(&clone_audio.samples).unwrap();
    let spk_emb = speaker_encoder.encode(&clone_audio.samples).unwrap();

    // 4. Load Models
    qwen3_tts::load_backends();
    qwen3_tts::init_backend();

    // Force F16 and CUDA for this debug
    let talker_file = "qwen3_tts_talker.gguf";
    let predictor_file = "qwen3_tts_predictor.gguf";
    let gpu_layers = 99;

    println!("Loading GGUF Models...");
    let talker_model = LlamaModel::load(&model_dir.join(talker_file), gpu_layers).unwrap();
    let predictor_model = LlamaModel::load(&model_dir.join(predictor_file), gpu_layers).unwrap();

    let mut talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, -1).unwrap();
    let mut predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, 4).unwrap();

    let tokenizer = Tokenizer::load(model_dir).unwrap();

    // 5. Test Case: Happy Instruction
    let text = "今天天气真不错，我们出去散步吧！";
    let instruct = "开心的说";

    // Prepare Reference for Clone Prompt
    let ref_text = "这是克隆音色参考音频，很高兴在这再次与你相见";
    let ref_text_ids = tokenizer.encode(ref_text);
    println!(
        "[Debug] Ref Text IDs ({}): {:?}",
        ref_text_ids.len(),
        ref_text_ids
    );

    let ref_codes_i32: Vec<i32> = ref_codes.iter().map(|&c| c as i32).collect();
    if !ref_codes_i32.is_empty() {
        let min_c = ref_codes_i32.iter().min().unwrap();
        let max_c = ref_codes_i32.iter().max().unwrap();
        println!(
            "[Debug] Ref Codes: len={}, min={}, max={}",
            ref_codes_i32.len(),
            min_c,
            max_c
        );
        println!("[Debug] Ref Codes[0..16]: {:?}", &ref_codes_i32[0..16]);
    }

    let mid_embeds = build_mid_embeds(&assets, &tokenizer, &ref_codes_i32, &ref_text_ids);

    let prompt_data = PromptBuilder::build_core(
        text,
        &tokenizer,
        &assets,
        Some(2055),     // Chinese
        None,           // No Spk ID (Clone mode)
        Some(&spk_emb), // Use cloned embedding
        Some(instruct),
        Some(mid_embeds),
    );

    // print Prompt Data stats
    println!("[Debug] Prompt Embeds: {} tokens", prompt_data.embd.len());
    let prompt_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
    // print some checks
    println!("[Debug] Prompt Flattened Head: {:?}", &prompt_flat[0..10]);

    let start = Instant::now();

    // Run Inference
    let current_seed = 123; // Fixed seed for debug
    println!("Running Inference with Seed={}...", current_seed);

    let (audio_samples, infer_dur) = run_inference(
        &prompt_data,
        &assets,
        &mut talker_ctx,
        &mut predictor_ctx,
        &talker_model,
        &predictor_model,
        &mut decoder,
        0.0, // Temp (Greedy)
        current_seed,
    );

    println!(
        "    -> Audio: {:.2}s, Time: {:.2}s",
        audio_samples.len() as f32 / 24000.0,
        infer_dur
    );

    // Save Audio
    let output_path = "debug_happy.wav";
    let audio_obj = AudioSample {
        samples: audio_samples,
        sample_rate: 24000,
        channels: 1,
    };
    audio_obj.save_wav(output_path).unwrap();
    println!("Saved to {}", output_path);

    let _ = std::io::stdout().flush();
    std::process::exit(0);
}

// Helper to construct "Mid Embeds" (Clone references)
fn build_mid_embeds(
    assets: &Assets,
    _tokenizer: &Tokenizer,
    ref_codes: &[i32],
    ref_text_ids: &[u32],
) -> Vec<Vec<f32>> {
    let mut mid_embeds = Vec::new();

    // 1. Inject Identity Overlay (Text): BOS_TOKEN -> ID -> EOS_TOKEN
    // Python Ref:
    // ref_ids = [p["BOS_TOKEN"]] + list(anchor.text_ids) + [p["EOS_TOKEN"]]
    let mut ref_ids_full = vec![BOS_TOKEN as u32];
    ref_ids_full.extend_from_slice(ref_text_ids);
    ref_ids_full.push(EOS_TOKEN as u32);

    let pad_emb = assets.get_codec_embedding(0, PAD as i32);

    for &tid in &ref_ids_full {
        let t_emb = assets.get_text_embedding(tid as usize);
        let summed: Vec<f32> = t_emb
            .iter()
            .zip(pad_emb.iter())
            .map(|(a, b)| a + b)
            .collect();
        mid_embeds.push(summed);
    }

    // 2. Inject Audio Codes (Codec BOS -> Codes -> PAD)
    let marker_emb = assets.get_text_embedding(TEXT_AUDIO_MARKER);
    let codec_bos_emb = assets.get_codec_embedding(0, 2160); // Codec BOS
    let start_sum: Vec<f32> = marker_emb
        .iter()
        .zip(codec_bos_emb.iter())
        .map(|(a, b)| a + b)
        .collect();
    mid_embeds.push(start_sum);

    // Codes Loop
    let n_steps = ref_codes.len() / 16;
    for step in 0..n_steps {
        let mut summed_c = vec![0.0; 2048];
        for q in 0..16 {
            let c = ref_codes[step * 16 + q];
            let emb = assets.get_codec_embedding(q, c);
            for i in 0..2048 {
                summed_c[i] += emb[i];
            }
        }
        let final_vec: Vec<f32> = marker_emb
            .iter()
            .zip(summed_c.iter())
            .map(|(a, b)| a + b)
            .collect();
        mid_embeds.push(final_vec);
    }

    // Add Pad at end of audio
    let pad_0 = assets.get_codec_embedding(0, PAD as i32);
    let end_sum: Vec<f32> = marker_emb
        .iter()
        .zip(pad_0.iter())
        .map(|(a, b)| a + b)
        .collect();
    mid_embeds.push(end_sum);

    mid_embeds
}

fn run_inference(
    pdata: &qwen3_tts::prompt_builder::PromptData,
    assets: &Assets,
    talker_ctx: &mut LlamaContext,
    predictor_ctx: &mut LlamaContext,
    talker_model: &LlamaModel,
    predictor_model: &LlamaModel,
    decoder: &mut AudioDecoder,
    temp: f32,
    seed: u64,
) -> (Vec<f32>, f32) {
    let start = Instant::now();
    let prompt_embeds_flat: Vec<f32> = pdata.embd.iter().flatten().copied().collect();
    let n_tokens = pdata.embd.len();

    let talker_embd = talker_model.n_embd;
    let predictor_embd = predictor_model.n_embd;

    // Clear KV Cache for new run
    talker_ctx.clear_kv_cache();
    predictor_ctx.clear_kv_cache();

    // Prefill
    let mut talker_batch = LlamaBatch::new(4096, talker_embd, 1, 4);
    let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);
    let predictor_sampler = qwen3_tts::llama::LlamaSampler::greedy(predictor_model.n_vocab);

    let pos_arr = qwen3_position(0, n_tokens);
    talker_batch.set_embd(&prompt_embeds_flat, &pos_arr, 0);
    talker_ctx.decode(&mut talker_batch).unwrap();

    // Loop
    let mut all_codes = Vec::new();
    let talker_sampler =
        qwen3_tts::llama::LlamaSampler::new(talker_model.n_vocab, temp, 50, 1.0, seed);

    let mut cur_pos = n_tokens;

    for step in 0..100 {
        if step == 0 {
            // Fix: Read from offset 110 (Last Token)
            let logits = talker_ctx.get_logits_ith(n_tokens - 1);
            let mut logits_pairs: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            logits_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!(
                "      [Step 0 Debug] Top 10 Logits (Index {}):",
                n_tokens - 1
            );
            for i in 0..10 {
                let (idx, val) = logits_pairs[i];
                println!("        Code {}: {:.5}", idx, val);
            }
        }

        // Sample Code 0
        // Fix: Sample from Last Token (Index 110) for Step 0
        let sample_idx = if step == 0 { (n_tokens - 1) as i32 } else { 0 };
        let code_0 = talker_sampler.sample(&talker_ctx, sample_idx, None, Some(2160));
        println!("  [Step {}] Code 0: {}", step, code_0);

        if code_0 == talker_model.eos_token || code_0 == 2150 || code_0 == 151673 {
            println!("      [Debug] Stop at step {}: Token={}", step, code_0);
            break;
        }

        all_codes.push(code_0 as i32);

        // Predictor Loop
        let emb_idx = if step == 0 { n_tokens - 1 } else { 0 };

        if step == 0 {
            let emb_0 = talker_ctx.get_embedding_at(0).to_vec();
            let emb_last = talker_ctx.get_embedding_at(n_tokens - 1).to_vec();
            println!("      [Step 0 Debug] Embeddings Check:");
            println!("        Index 0 Sum: {:.5}", emb_0.iter().sum::<f32>());
            println!(
                "        Index {} Sum: {:.5}",
                n_tokens - 1,
                emb_last.iter().sum::<f32>()
            );
            if emb_0 == emb_last {
                println!(
                    "        WARNING: Embeddings at 0 and {} are IDENTICAL!",
                    n_tokens - 1
                );
            } else {
                println!("        Embeddings are DISTINCT.");
            }
        }

        let m_hidden = talker_ctx.get_embedding_at(emb_idx).to_vec();
        // Check m_hidden sum
        let sum_hidden: f32 = m_hidden.iter().sum();
        println!("      [Debug] m_hidden sum: {:.5}", sum_hidden);

        let m_h_1024 = assets.project(&m_hidden);
        let code_0_1024 = assets.get_codec_embedding_1024(0, code_0 as i32);

        let mut predictor_input = Vec::with_capacity(2048);
        predictor_input.extend_from_slice(&m_h_1024);
        predictor_input.extend_from_slice(&code_0_1024);

        predictor_ctx.clear_kv_cache();
        predictor_batch.clear();
        let pred_pos = normal_position(0, 2);
        predictor_batch.set_embd(&predictor_input, &pred_pos, 0);
        predictor_ctx.decode(&mut predictor_batch).unwrap();

        let mut step_embeds = Vec::new();
        step_embeds.push(assets.get_codec_embedding(0, code_0 as i32));

        for q in 1..16 {
            let start = (q - 1) * 2048;
            let end = q * 2048;

            // Fix: Predictor batch is always small. Use idx=0.
            let idx = 0;

            let sampled = predictor_sampler.sample(&predictor_ctx, idx, Some(start), Some(end));
            let c = sampled - start as i32;
            all_codes.push(c);
            step_embeds.push(assets.get_codec_embedding(q, c));

            if q < 15 {
                let next = assets.get_codec_embedding_1024(q, c);
                let pos = normal_position(q + 1, 1);
                predictor_batch.clear();
                predictor_batch.set_embd(&next, &pos, 0);
                predictor_ctx.decode(&mut predictor_batch).unwrap();
            }
        }

        // Feedback
        // Check step_embeds sum
        let mut feedback_sum = 0.0;
        let mut feedback = vec![0.0; 2048];
        for emb in &step_embeds {
            for (i, v) in emb.iter().enumerate() {
                feedback[i] += v;
            }
        }
        for (i, v) in assets.tts_pad.iter().enumerate() {
            feedback[i] += v;
        }

        feedback_sum = feedback.iter().sum();
        println!("      [Debug] Feedback sum: {:.5}", feedback_sum);

        let t_pos = qwen3_position(cur_pos as i32, 1);
        talker_batch.clear();
        talker_batch.set_embd(&feedback, &t_pos, 0);
        talker_ctx.decode(&mut talker_batch).unwrap();

        cur_pos += 1;
    }

    let infer_duration = start.elapsed().as_secs_f32();

    // Quick Decode
    let mut state = AudioDecoder::create_state();
    // Fix: Clamp codes to 2047
    let codes_i64: Vec<i64> = all_codes.iter().map(|&c| (c.min(2047)) as i64).collect();
    let audio = decoder.decode(&codes_i64, &mut state, true).unwrap();

    (audio, infer_duration)
}
