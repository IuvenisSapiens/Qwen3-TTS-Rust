//! Qwen3-TTS Rust Library
//! Optimized and organized structure for open source.

pub mod assets_manager;
pub mod models;
pub mod tts;
pub mod utils;

// Re-export core types for convenience
pub use tts::engine::TtsEngine;
pub use tts::prompt::PromptBuilder;
pub use utils::audio::AudioSample;
pub use utils::tokenizer::Tokenizer;
pub use utils::voice_file::VoiceFile;

pub fn load_backends() {
    models::llama::load_backends();
}

pub fn init_backend() {
    models::llama::init_backend();
}

pub fn cleanup() {
    models::llama::cleanup();
}
