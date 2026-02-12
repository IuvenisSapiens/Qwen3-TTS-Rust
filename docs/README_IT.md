# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [Italiano](README_IT.md)

Questo progetto è un'implementazione Rust ad alte prestazioni di Qwen3-TTS. Le innovazioni principali sono la sintesi **"Guidata da Istruzioni (Instruction-Driven)"** e la **"Clonazione Vocale Personalizzata (Custom Speakers)"**. Sfruttando la sicurezza della memoria di Rust e l'inferenza efficiente di llama.cpp/ONNX, offre una soluzione di sintesi vocale di livello industriale.

## 🚀 Grande Salto: Istruzioni e Personalizzazione

A differenza dei sistemi TTS tradizionali, Qwen3-TTS Rust consente di controllare lo stile del parlato tramite semplici istruzioni testuali e di clonare qualsiasi voce in pochi secondi.

### 1. Guidato da Istruzioni (Instruction-Driven)
È possibile includere istruzioni su emozione, velocità o stile direttamente nel testo. Il modello linguistico (LLM) utilizza la sua comprensione semantica per "sapere" come leggere.
> **Esempio**: `cargo run --example qwen3-tts -- --text "[Goiamente] Ciao! Il tempo oggi è assolutamente fantastico!" --voice-file "speaker.json"`

### 2. Voci Personalizzate (Custom Speakers)
Non sei più limitato alle voci predefinite. Con un solo **audio di riferimento WAV a 24kHz**, puoi creare un pacchetto vocale unico.
-   **Estrazione in un click**: Estrae automaticamente i vettori del parlante (Speaker Embedding) e le caratteristiche acustiche (Codec Codes).
-   **Salvataggio Permanente**: Salvato come `.json` dopo l'estrazione, l'audio originale non è più necessario per l'uso futuro.

## 🌟 Vantaggi Tecnici

-   **Multi-Piattaforma/Backends**: Adattamento profondo per **Windows / Linux / macOS**, supportando **CPU / CUDA / Vulkan / Metal**.
-   **Runtime Senza Configurazione**: Gestisce automaticamente le dipendenze binarie di `llama.cpp` (b7885) e `onnxruntime`, con mappatura degli asset multi-piattaforma e caricamento dinamico.
-   **Motore Ibrido**: 
    -   **Inferenza LLM**: Utilizza llama.cpp per la conversione da testo a caratteristiche acustiche, con accelerazione hardware **Vulkan** attivata per impostazione predefinita.
    -   **Decodifica Audio**: Utilizza ONNX Runtime (CPU) per una decodifica fluida, garantendo una latenza minima.

## 🛠️ Guida Rapida

### Creare e Salvare una Voce Personalizzata
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "Il testo pronunciato durante la registrazione" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Emozionato] Ehi! La mia voce è stata clonata nel motore Rust!" `
    --max-steps 512
```

## 📂 Gestione Automatizzata
Il programma include una logica di **auto-download di modelli e runtime**. Al primo avvio, scaricherà automaticamente i modelli da HuggingFace e i binari ufficiali di `llama.cpp` appropriati nella cartella `runtime/` in base al proprio sistema operativo.

## 📜 Licenza e Ringraziamenti
- Licenza **MIT / Apache 2.0**.
- Grazie al repository ufficiale [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) per i modelli e la base tecnica.
- Grazie a [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) per l'ispirazione sul flusso di inferenza GGUF.
