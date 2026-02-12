# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [Español](README_ES.md)

Este proyecto es una implementación en Rust de alto rendimiento de Qwen3-TTS. Los avances clave son la síntesis **"Guiada por Instrucciones (Instruction-Driven)"** y la **"Clonación de Voz Personalizada (Custom Speakers)"**. Aprovechando la seguridad de memoria de Rust y la inferencia eficiente de llama.cpp/ONNX, ofrece una solución de texto a voz de grado industrial.

## 🚀 Gran Salto: Instrucciones y Personalización

A diferencia de los sistemas TTS tradicionales, Qwen3-TTS Rust le permite controlar el estilo de habla mediante simples instrucciones de texto y clonar cualquier voz en segundos.

### 1. Guiado por Instrucciones (Instruction-Driven)
Puede incluir instrucciones de emoción, velocidad o estilo directamente en el texto. El modelo de lenguaje (LLM) utiliza su comprensión semántica para "saber" cómo leer.
> **Ejemplo**: `cargo run --example qwen3-tts -- --text "[Alegremente] ¡Hola! ¡El clima de hoy es absolutamente fantástico!" --voice-file "speaker.json"`

### 2. Voces Personalizadas (Custom Speakers)
Ya no está limitado a voces preestablecidas. Con solo un **audio de referencia WAV de 24kHz**, puede crear un paquete de voz único.
-   **Extracción en un clic**: Extrae automáticamente los vectores del hablante (Speaker Embedding) y las características acústicas (Codec Codes).
-   **Guardado Permanente**: Se guarda como `.json` después de la extracción, no se necesita el audio original para su uso futuro.

## 🌟 Ventajas Técnicas

-   **Multiplataforma/Backends**: Adaptación profunda para **Windows / Linux / macOS**, soportando **CPU / CUDA / Vulkan / Metal**.
-   **Runtime Sin Configuración**: Gestiona automáticamente las dependencias binarias de `llama.cpp` (b7885) y `onnxruntime`, con mapeo de activos multiplataforma y carga dinámica.
-   **Motor Híbrido**: 
    -   **Inferencia LLM**: Utiliza llama.cpp para la conversión de texto a características acústicas, con aceleración de hardware **Vulkan** activada por defecto.
    -   **Decodificación de Audio**: Utiliza ONNX Runtime (CPU) para una decodificación fluida, asegurando una latencia mínima.

## 🛠️ Guía Rápida

### Crear y Guardar una Voz Personalizada
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "El texto que dije durante la grabación" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Emocionado] ¡Oye! ¡Mi voz ha sido clonada en el motor Rust!" `
    --max-steps 512
```

## 📂 Gestión Automatizada
El programa tiene incorporada una lógica de **autodescarga de modelos y runtimes**. En la primera ejecución, descargará automáticamente los modelos de HuggingFace y los binarios oficiales de `llama.cpp` adecuados en la carpeta `runtime/` según su sistema operativo.

## 📜 Licencia y Agradecimientos
- Licencia **MIT / Apache 2.0**.
- Gracias al repositorio oficial de [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) por los modelos y la base técnica.
- Gracias a [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) por la inspiración en el flujo de inferencia GGUF.
