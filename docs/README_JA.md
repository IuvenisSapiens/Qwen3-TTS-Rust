# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md)

本プロジェクトは Qwen3-TTS の極致性能を実現した Rust 実装です。核心的な進歩は **「指示駆動 (Instruction-Driven)」** と **「ゼロショット・カスタム音色 (Custom Speakers)」** の深い統合にあります。Rust のメモリ安全特性と llama.cpp/ONNX の効率的な推論を組み合わせ、工業レベルのテキスト読み上げソリューションを提供します。

## 🚀 核心的な飛躍：指示とカスタマイズ

従来の TTS とは異なり、Qwen3-TTS Rust では単純なテキスト指示で音声スタイルを制御でき、あらゆる音色を数秒でクローンできます。

### 1. 指示駆動 (Instruction-Driven)
テキストの中に感情、速度、スタイルの指示を直接含めることができます。大規模言語モデル (LLM) の意味理解能力により、AI が「どう読むべきか」を判断します。
> **例**: `cargo run --example qwen3-tts -- --text "[嬉しそうに] こんにちは！今日の天気は本当に最高ですね！" --voice-file "speaker.json"`

### 2. カスタム音色 (Custom Speakers)
プリセットの音色に縛られることはもうありません。わずか **24kHz の WAV 参照オーディオ** があれば、専用の音色パックを作成できます。
-   **ワンクリック抽出**: 話者ベクトル (Speaker Embedding) と音響特徴 (Codec Codes) を自動的に抽出します。
-   **永久保存**: 抽出後は `.json` として保存され、次回の使用時に元の音声ファイルは不要です。

## 🌟 技術的優位性

-   **全プラットフォーム/全バックエンド対応**: **Windows / Linux / macOS** に深く適応し、 **CPU / CUDA / Vulkan / Metal** をサポートします。
-   **ゼロ構成ランタイム**: `llama.cpp` (b7885) と `onnxruntime` のバイナリ依存関係を自動管理し、プラットフォーム間のアセットマッピングと動的ロードをサポートします。
-   **ハイブリッド・エンジン**: 
    -   **LLM 推論**: llama.cpp を使用してテキストから音響特徴への変換を処理し、デフォルトで **Vulkan** ハードウェアアクセラレーションを有効にします。
    -   **音声デコード**: ONNX Runtime (CPU) を使用して効率的なストリーミングデコードを行い、極めて低い初回トークン遅延を実現します。

## 🛠️ クイック操作ガイド

### カスタム音色を作成して保存する
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "録音時に話した内容" `
    --save-voice "models/presets/my_voice.json" `
    --text "[興奮気味に] ねえ！私の声が Rust エンジンにクローンされたよ！"
```

### 既存の音色パックを使用する
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --voice-file "models/presets/my_voice.json" `
    --text "Qwen3-TTS Rust 推論エンジンへようこそ。" `
    --max-steps 512
```

## 📂 自動管理
プログラムには、**モデルとランタイムの自動ダウンロード**ロジックが組み込まれています。初回実行時に HuggingFace からモデルファイルを自動的にダウンロードし、OS に応じた適切な `llama.cpp` 公式バイナリを `runtime/` ディレクトリにダウンロードします。

### 推奨ディレクトリ構造
```text
models/
├── onnx/      (Codec/Speaker/Decoder ONNX)
├── tokenizer/ (Tokenizer Config)
└── gguf/      (Talker/Predictor/Assets GGUF)
```

## 📜 ライセンスと謝辞
- **MIT / Apache 2.0** ライセンスに基づきます。
- モデルと技術基盤を提供してくれた [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 公式リポジトリに感謝します。
- GGUF 推論フローのインスピレーションを与えてくれた [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) に感謝します。
