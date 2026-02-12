# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md)

이 프로젝트는 Qwen3-TTS의 고성능 Rust 구현체입니다. 핵심적인 돌파구는 **"지시어 주도(Instruction-Driven)"** 합성 및 **"제로샷 사용자 정의 음색(Custom Speakers)"**의 깊은 통합입니다. Rust의 메모리 안전성과 llama.cpp/ONNX의 효율적인 추론을 결합하여 산업급 텍스트 음성 변환 솔루션을 제공합니다.

## 🚀 핵심 도약: 지시어 및 사용자 정의

기존의 TTS와 달리, Qwen3-TTS Rust는 간단한 텍스트 지시어로 음성 스타일을 제어할 수 있으며, 어떤 음색이든 몇 초 만에 복제할 수 있습니다.

### 1. 지시어 주도 (Instruction-Driven)
텍스트에 감정, 속도 또는 스타일 지시어를 직접 포함할 수 있습니다. 대규모 언어 모델(LLM)의 의미 이해 능력을 통해 AI가 "어떻게 읽어야 할지"를 스스로 판단합니다.
> **예시**: `cargo run --example qwen3-tts -- --text "[기쁘게] 안녕하세요! 오늘 날씨가 정말 최고네요!" --voice-file "speaker.json"`

### 2. 사용자 정의 음색 (Custom Speakers)
더 이상 프리셋 음색에 얽매일 필요가 없습니다. 단 **24kHz WAV 참조 오디오**만 있으면 나만의 전용 음색 팩을 만들 수 있습니다.
-   **원클릭 추출**: 화자 벡터(Speaker Embedding)와 음향 특징(Codec Codes)을 자동으로 추출합니다.
-   **영구 저장**: 추출 후 `.json`으로 저장되며, 다음 사용 시 원본 오디오 파일이 필요하지 않습니다.

## 🌟 기술적 장점

-   **전체 플랫폼/전체 백엔드 지원**: **Windows / Linux / macOS**에 깊이 최적화되어 있으며, **CPU / CUDA / Vulkan / Metal**을 지원합니다.
-   **제로 구성 런타임**: `llama.cpp` (b7885) 및 `onnxruntime` 바이너리 종속성을 자동 관리하며, 플랫폼 간 어셋 매핑 및 동적 로드를 지원합니다.
-   **하이브리드 엔진**: 
    -   **LLM 추론**: llama.cpp를 사용하여 텍스트에서 음향 특징으로의 변환을 처리하며, 기본적으로 **Vulkan** 하드웨어 가속을 사용합니다.
    -   **오디오 디코딩**: ONNX Runtime(CPU)을 사용하여 효율적인 스트리밍 디코딩을 수행하며, 매우 낮은 첫 토큰 지연 시간을 실현합니다.

## 🛠️ 빠른 조작 가이드

### 사용자 정의 음색 생성 및 저장
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "녹음 시 말했던 텍스트 내용" `
    --save-voice "models/presets/my_voice.json" `
    --text "[흥분해서] 우와! 내 목소리가 이제 Rust 엔진으로 복제되었어!"
```

### 기존 음색 팩 사용
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --voice-file "models/presets/my_voice.json" `
    --text "Qwen3-TTS Rust 추론 엔진에 오신 것을 환영합니다." `
    --max-steps 512
```

## 📂 자동 관리
프로그램에는 **모델 및 런타임 자동 다운로드** 로직이 내장되어 있습니다. 첫 실행 시 HuggingFace에서 모델 파일을 자동으로 다운로드하고, 운영체제에 맞는 `llama.cpp` 공식 바이너리를 `runtime/` 디렉토리에 다운로드합니다.

### 권장 디렉토리 구조
```text
models/
├── onnx/      (Codec/Speaker/Decoder ONNX)
├── tokenizer/ (Tokenizer Config)
└── gguf/      (Talker/Predictor/Assets GGUF)
```

## 📜 라이선스 및 감사
- **MIT / Apache 2.0** 라이선스를 따릅니다.
- 모델과 기술 기반을 제공해 준 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 공식 저장소에 감사드립니다.
- GGUF 추론 흐름에 대한 영감을 준 [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF)에 감사드립니다.
