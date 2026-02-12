# Qwen3-TTS Rust

[中文](README.md) | [English](docs/README_EN.md)

本项目是 Qwen3-TTS 的极致性能实现，核心突破在于 **“指令驱动 (Instruction-Driven)”** 与 **“零样本自定义音色 (Custom Speakers)”** 的深度集成。通过 Rust 的内存安全特性与 llama.cpp/ONNX 的高效推理，为您提供工业级的文本转语音解决方案。

## 🚀 核心特性

### 1. 极致性能与流式响应
- **并发流式解码**：采用 4 帧 (64 codes) 粒度的并发解码策略，首字延迟低至 300ms，实现“边想边说”的流畅体验。
- **硬件加速**：默认启用 **Vulkan** (Windows/Linux) 和 **Metal** (macOS) 加速，显著提升推理速度。
- **自动运行时管理**：零配置环境，自动下载并配置 `llama.cpp` (b7885) 和 `onnxruntime`，开箱即用。

### 2. 灵活的说话人管理
- **自动扫描与缓存**：启动时自动加载 `speakers/` 和 `preset_speakers/` 目录下的音色文件。
- **多种选择方式**：支持通过 CLI 参数 `--speaker <name>` 或 `--voice-file <path>` 灵活选择说话人。
- **智能回退**：若指定说话人不存在，自动回退至默认音色 (vivian)，确保系统稳定性。

### 3. 精准的指令控制
- **指令驱动**：支持在文本中嵌入 `[高兴]`、`[悲伤]` 等情感指令，实时调整演绎风格。
- **EOS 对齐**：完美对齐 Qwen3 的停止逻辑，支持多种 EOS token 检测，杜绝生成末尾的静音或乱码。

## 🛠️ 快速上手

### 1. 基础生成
使用默认说话人生成语音：
```powershell
cargo run --example qwen3-tts -- --text "你好，欢迎使用 Qwen3-TTS Rust！"
```

### 2. 指定说话人
使用预设或自定义说话人：
```powershell
# 使用名称 (需在 speakers/ 目录下存在对应的 .json 文件)
cargo run --example qwen3-tts -- --text "今天天气不错。" --speaker dylan

# 使用指定文件路径
cargo run --example qwen3-tts -- --text "我是自定义音色。" --voice-file "path/to/my_voice.json"
```

### 3. 克隆新音色
只需 3-10 秒的参考音频即可克隆音色：
```powershell
cargo run --example qwen3-tts -- `
    --ref-audio "ref.wav" `
    --ref-text "参考音频对应的文本内容" `
    --save-voice "speakers/my_voice.json" `
    --text "新音色已保存，现在可以直接使用了！"
```

### 4. 高级配置
```powershell
cargo run --example qwen3-tts -- `
    --text "长文本生成测试。" `
    --max-steps 1024 `    # 调整最大生成长度
    --output "output.wav" # 指定输出文件名
```

## 📂 目录结构

系统首次运行会自动构建如下结构：

```text
.
├── models/             # 模型文件 (GGUF, ONNX, Tokenizer)
├── runtime/            # 自动下载的依赖库 (dll, so, dylib)
├── speakers/           # 用户自定义音色
└── preset_speakers/    # 系统预设音色
```

## 📜 许可证与致谢

- 基于 **MIT / Apache 2.0** 许可证。
- 感谢 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 官方仓库提供的模型与技术基座。
- 感谢 [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) 提供的推理流程启发。
