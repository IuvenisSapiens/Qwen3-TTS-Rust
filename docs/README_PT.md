# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [Português](README_PT.md)

Este projeto é uma implementação em Rust de alto desempenho do Qwen3-TTS. Os principais avanços são a síntese **"Guiada por Instruções (Instruction-Driven)"** e a **"Clonagem de Voz Personalizada (Custom Speakers)"**. Aproveitando a segurança de memória do Rust e a inferência eficiente do llama.cpp/ONNX, ele oferece uma solução de texto para fala de nível industrial.

## 🚀 Grande Salto: Instruções e Personalização

Ao contrário dos sistemas TTS tradicionais, o Qwen3-TTS Rust permite controlar o estilo de fala através de simples instruções de texto e clonar qualquer voz em segundos.

### 1. Guiado por Instruções (Instruction-Driven)
Você pode incluir instruções de emoção, velocidade ou estilo diretamente no texto. O modelo de linguagem (LLM) usa sua compreensão semântica para "saber" como ler.
> **Exemplo**: `cargo run --example qwen3-tts -- --text "[Alegremente] Olá! O tempo hoje está absolutamente fantástico!" --voice-file "speaker.json"`

### 2. Vozes Personalizadas (Custom Speakers)
Não está mais limitado a vozes predefinidas. Com apenas um **áudio de referência WAV de 24kHz**, você pode criar um pacote de voz exclusivo.
-   **Extração em um clique**: Extrai automaticamente os vetores do falante (Speaker Embedding) e as características acústicas (Codec Codes).
-   **Salvamento Permanente**: Salvo como `.json` após a extração, não é necessário o áudio original para uso futuro.

## 🌟 Vantagens Técnicas

-   **Multiplataforma/Backends**: Adaptação profunda para **Windows / Linux / macOS**, suportando **CPU / CUDA / Vulkan / Metal**.
-   **Runtime Sem Configuração**: Gere automaticamente as dependências binárias do `llama.cpp` (b7885) e `onnxruntime`, com mapeamento de ativos multiplataforma e carregamento dinâmico.
-   **Motor Híbrido**: 
    -   **Inferência LLM**: Usa llama.cpp para a conversão de texto em características acústicas, com aceleração de hardware **Vulkan** ativada por padrão.
    -   **Decodificação de Áudio**: Usa ONNX Runtime (CPU) para uma decodificação fluida, garantindo latência mínima.

## 🛠️ Guia Rápido

### Criar e Salvar uma Voz Personalizada
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "O texto que eu disse durante a gravação" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Animado] Ei! Minha voz foi clonada no motor Rust!" `
    --max-steps 512
```

## 📂 Gestão Automatizada
O programa possui uma lógica integrada de **autodownload de modelos e runtimes**. Na primeira execução, ele baixará automaticamente os modelos do HuggingFace e os binários oficiais do `llama.cpp` adequados na pasta `runtime/` de acordo com o seu sistema operacional.

## 📜 Licença e Agradecimentos
- Licença **MIT / Apache 2.0**.
- Obrigado ao repositório oficial do [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) pelos modelos e base técnica.
- Obrigado ao [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) pela inspiração no fluxo de inferência GGUF.
