# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [Deutsch](README_DE.md)

Dieses Projekt ist eine leistungsstarke Rust-Implementierung von Qwen3-TTS. Die wichtigsten Durchbrüche sind die **"Anweisungsgesteuerte (Instruction-Driven)"** Synthese und das **"Zero-Shot Custom Speaker Cloning"**. Durch die Kombination der Speichersicherheit von Rust mit der effizienten Inferenz von llama.cpp/ONNX bietet es eine Text-to-Speech-Lösung auf Industrieniveau.

## 🚀 Kernmerkmale: Anweisungen & Anpassung

Im Gegensatz zu herkömmlichen TTS-Systemen ermöglicht Qwen3-TTS Rust die Steuerung des Sprechstils durch einfache Textanweisungen und das Klonen jeder Stimme in Sekundenschnelle.

### 1. Anweisungsgesteuerter (Instruction-Driven) TTS
Sie können Anweisungen zu Emotionen, Geschwindigkeit oder Stil direkt in den Text einfügen. Das Sprachmodell (LLM) nutzt sein semantisches Verständnis, um zu "wissen", wie der Text gelesen werden soll.
> **Beispiel**: `cargo run --example qwen3-tts -- --text "[Fröhlich] Hallo! Das Wetter heute ist einfach fantastisch!" --voice-file "speaker.json"`

### 2. Benutzerdefinierte Stimmen (Custom Speakers)
Nicht mehr auf voreingestellte Stimmen beschränkt. Mit nur einem **24kHz WAV-Referenzaudio** können Sie ein einzigartiges Voice-Pack erstellen.
-   **One-Click-Extraktion**: Extrahiert automatisch Sprecher-Embeddings und akustische Merkmale (Codec-Codes).
-   **Dauerhafte Speicherung**: Nach der Extraktion als `.json` gespeichert, kein Original-Audio für die zukünftige Verwendung erforderlich.

## 🌟 Technische Vorteile

-   **Plattformübergreifend/Backends**: Tiefe Anpassung für **Windows / Linux / macOS**, unterstützt **CPU / CUDA / Vulkan / Metal**.
-   **Zero-Config Runtime**: Automatische Verwaltung von `llama.cpp` (b7885) und `onnxruntime` Binärdateien, mit plattformübergreifendem Asset-Mapping und dynamischem Laden.
-   **Hybrid-Engine**: 
    -   **LLM-Inferenz**: Verwendet llama.cpp für die Konvertierung von Text in akustische Merkmale, standardmäßig mit **Vulkan** Hardwarebeschleunigung.
    -   **Audio-Dekodierung**: Verwendet ONNX Runtime (CPU) für effizientes Streaming-Dekodieren mit minimaler Latenz.

## 🛠️ Kurzanleitung

### Benutzerdefinierte Stimme erstellen und speichern
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "Der Text, den ich während der Aufnahme gesprochen habe" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Aufgeregt] Hey! Meine Stimme wurde gerade in die Rust-Engine geklont!" `
    --max-steps 512
```

## 📂 Automatisierte Verwaltung
Das Programm verfügt über eine integrierte Logik zum **automatischen Download von Modellen und Runtimes**. Beim ersten Start werden die Modelle von HuggingFace und die entsprechenden offiziellen `llama.cpp` Binärdateien je nach Betriebssystem automatisch in das Verzeichnis `runtime/` heruntergeladen.

## 📜 Lizenz & Danksagung
- **MIT / Apache 2.0** Lizenz.
- Dank an das offizielle [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Repository für die Modelle und technische Basis.
- Dank an [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) für die Inspiration zum GGUF-Inferenzfluss.
