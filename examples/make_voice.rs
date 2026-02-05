//! Tool to Extract Voice Features and Save as .qvoice File

use clap::Parser;
use qwen3_tts::models::llama;
use qwen3_tts::models::onnx::{AudioEncoder, SpeakerEncoder};
use qwen3_tts::AudioSample;
use qwen3_tts::VoiceFile;
use std::path::Path;
use std::process::exit;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = ".")]
    model_dir: String,

    #[arg(short, long, default_value = "clone.wav")]
    input: String,

    #[arg(short, long, default_value = "my_voice.qvoice")]
    output: String,

    #[arg(short, long, default_value = "这是参考音频文本")]
    text: String,

    #[arg(long)]
    name: Option<String>,

    #[arg(long)]
    gender: Option<String>,

    #[arg(long)]
    age: Option<String>,

    #[arg(long)]
    description: Option<String>,
}

fn main() {
    let args = Args::parse();

    println!("===========================================================");
    println!("  Qwen3-TTS Voice Extractor");
    println!("===========================================================");
    println!("  Model Dir: {}", args.model_dir);
    println!("  Input:     {}", args.input);
    println!("  Output:    {}", args.output);
    println!("  Text:      {}", args.text);
    if let Some(n) = &args.name {
        println!("  Name:      {}", n);
    }

    let model_dir = Path::new(&args.model_dir);

    // 1. Load Audio
    println!("\n[1/4] Loading Audio...");
    let audio = match AudioSample::load_wav(&args.input) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading audio: {}", e);
            exit(1);
        }
    };
    println!("      Complete. {} samples", audio.samples.len());

    // 2. Load Encoders
    println!("\n[2/4] Loading Models...");
    // Need to initialize backend for some reason? Maybe not for pure ONNX if we use ort direct?
    // But our models might use ort which needs init.
    // Let's call load_backends just in case.
    llama::load_backends(); // Actually initializes basic env if needed

    let encoder_path = model_dir.join("qwen3_tts_codec_encoder.onnx");
    let spk_path = model_dir.join("qwen3_tts_speaker_encoder.onnx");

    let mut encoder = match AudioEncoder::load(encoder_path.to_str().unwrap()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading AudioEncoder: {}", e);
            exit(1);
        }
    };

    let mut spk_encoder = match SpeakerEncoder::load(spk_path.to_str().unwrap()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading SpeakerEncoder: {}", e);
            exit(1);
        }
    };
    println!("      Models loaded.");

    // 3. Extract Features
    println!("\n[3/4] Extracting Features...");
    let codes = match encoder.encode(&audio.samples) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error encoding audio: {}", e);
            exit(1);
        }
    };
    println!("      Audio Codes: {} items", codes.len());

    let spk_emb = match spk_encoder.encode(&audio.samples) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error extracting speaker embedding: {}", e);
            exit(1);
        }
    };
    println!("      Speaker Embedding: {} dims", spk_emb.len());

    // 4. Save Voice File
    println!("\n[4/4] Saving to .qvoice...");
    let voice_file = VoiceFile::new(args.text, codes, spk_emb).with_metadata(
        args.name,
        args.gender,
        args.age,
        args.description,
    );

    if let Err(e) = voice_file.save(&args.output) {
        eprintln!("Error saving voice file: {}", e);
        exit(1);
    }

    println!("\nSuccess! Saved voice to: {}", args.output);
}
