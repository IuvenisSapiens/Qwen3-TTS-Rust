//! Qwen3-TTS Rust Implementation - Library Usage Example

use clap::Parser;
use qwen3_tts::{AudioSample, TtsEngine};
use std::path::Path;
use std::process::exit;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = ".")]
    model_dir: String,

    #[arg(short, long, default_value = "clone.wav")]
    input: String,

    #[arg(short, long, default_value = "output.wav")]
    output: String,

    #[arg(
        short,
        long,
        default_value = "这是克隆音色参考音频，很高兴在这再次与你相见"
    )]
    ref_text: String,

    #[arg(short, long, default_value = "欢迎使用Rust推理引擎")]
    text: String,

    #[arg(long)]
    voice: Option<String>,
}

fn main() {
    let args = Args::parse();

    println!("===========================================================");
    println!("  Qwen3-TTS Rust Library Example");
    println!("===========================================================");

    // 1. Load Engine
    println!("Loading TTS Engine...");
    let start_load = Instant::now();
    let mut engine = match TtsEngine::load(&args.model_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading engine: {}", e);
            exit(1);
        }
    };
    println!(
        "Engine loaded in {:.2} s",
        start_load.elapsed().as_secs_f32()
    );

    // 2. Generate Audio
    println!("\nGenerating Audio...");
    let start_gen = Instant::now();

    let audio = if let Some(voice_path) = args.voice {
        println!("  Using Voice File: {}", voice_path);
        println!("  Target Text:      {}", args.text);

        let voice = match qwen3_tts::VoiceFile::load(&voice_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error loading voice file: {}", e);
                exit(1);
            }
        };

        if let Some(name) = &voice.name {
            println!("  Voice Name:   {}", name);
        }
        if let Some(gender) = &voice.gender {
            println!("  Gender:       {}", gender);
        }
        if let Some(age) = &voice.age {
            println!("  Age Group:    {}", age);
        }
        if let Some(desc) = &voice.description {
            println!("  Description:  {}", desc);
        }

        match engine.generate_with_voice(&args.text, &voice) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error generating audio: {}", e);
                exit(1);
            }
        }
    } else {
        println!("  Input Audio: {}", args.input);
        println!("  Ref Text:    {}", args.ref_text);
        println!("  Target Text: {}", args.text);

        match engine.generate(&args.text, &args.input, &args.ref_text) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error generating audio: {}", e);
                exit(1);
            }
        }
    };
    let gen_duration = start_gen.elapsed().as_secs_f32();
    let audio_duration = audio.duration();

    println!("Generation check: {} samples", audio.samples.len());
    println!("  Time: {:.2} s", gen_duration);
    println!("  RTF:  {:.2}", gen_duration / audio_duration);

    // 3. Save Output
    if let Err(e) = audio.save_wav(&args.output) {
        eprintln!("Error saving output: {}", e);
        exit(1);
    }
    println!("\nSaved to: {}", args.output);

    // qwen3_tts::cleanup(); // Cleanup is optional/automatic via Drop typically, but we have explicit fn
    qwen3_tts::cleanup();
}
