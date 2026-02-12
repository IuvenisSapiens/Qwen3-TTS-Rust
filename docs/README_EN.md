# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md)

This project is a high-performance Rust implementation of Qwen3-TTS. The core breakthroughs are **"Instruction-Driven"** synthesis and **"Zero-shot Custom Speakers (Cloning)"**. Leveraging Rust's memory safety and the efficient inference of llama.cpp/ONNX, it provides an industrial-grade text-to-speech solution.

## 🚀 Key Leap: Instructions & Customization

Unlike traditional TTS, Qwen3-TTS Rust allows you to control speech style through simple text instructions and clone any voice in seconds.

### 1. Instruction-Driven
You can include emotional, speed, or style instructions directly in the text. The Large Language Model (LLM) uses its semantic understanding to "know" how to read.
> **Example**: `cargo run --example qwen3-tts -- --text "[Happily] Hello! The weather today is absolutely fantastic!" --voice-file "speaker.json"`

### 2. Custom Speakers (Cloning)
No longer restricted to preset voices. With just a **24kHz WAV reference audio**, you can create a unique voice pack.
-   **One-click Extraction**: Automatically extract Speaker Embeddings and acoustic features (Codec Codes).
-   **Permanent Saving**: Saved as `.json` after extraction, no original audio needed for future use.

## 🌟 Technical Advantages

-   **Cross-Platform/Backend**: Deeply adapted for **Windows / Linux / macOS**, supporting **CPU / CUDA / Vulkan / Metal**.
-   **Zero-Config Runtime**: Automatically manages `llama.cpp` (b7885) and `onnxruntime` binaries, featuring cross-platform asset mapping and dynamic loading.
-   **Hybrid Engine**: 
    -   **LLM Inference**: Uses llama.cpp for text-to-acoustic feature conversion, with **Vulkan** hardware acceleration enabled by default.
    -   **Audio Decoding**: Uses ONNX Runtime (CPU) for efficient streaming decoding, ensuring extremely low first-token latency.

## 🛠️ Quick Operation Guide

### Create and Save a Custom Voice
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "The text content I said during recording" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Excitedly] Hey! My voice has now been cloned into the Rust engine!"
```

### Use an Existing Voice Pack
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --voice-file "models/presets/my_voice.json" `
    --text "Welcome to the Qwen3-TTS Rust inference engine." `
    --max-steps 512
```

## 📂 Automated Management
The program has built-in **Auto-download** logic for both models and runtimes. On the first run, it will automatically download models from HuggingFace and the appropriate `llama.cpp` official binaries to the `runtime/` directory based on your OS.

### Recommended Directory Structure
```text
models/
├── onnx/      (Codec/Speaker/Decoder ONNX)
├── tokenizer/ (Tokenizer Config)
└── gguf/      (Talker/Predictor/Assets GGUF)
```

## 📜 License & Acknowledgements
- Based on the **MIT / Apache 2.0** license.
- Thanks to the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) official repository for the models and technical foundation.
- Thanks to [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) for the inspiration on the GGUF inference flow.
