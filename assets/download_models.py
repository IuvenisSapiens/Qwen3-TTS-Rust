import os
import requests
from tqdm import tqdm

MODELS = {
    # 基础模型 (ONNX Encoders & Decoders, GGUF Models)
    # 假设托管在 HuggingFace 或 ModelScope
    # 这里使用示例 URL，实际需要替换为真实 URL
    "qwen3_tts_codec_encoder.onnx": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/qwen3_tts_codec_encoder.onnx",
    "qwen3_tts_speaker_encoder.onnx": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/qwen3_tts_speaker_encoder.onnx",
    "qwen3_tts_decoder.onnx": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/qwen3_tts_decoder.onnx",
    "qwen3_tts_talker-q4km.gguf": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/qwen3_tts_talker-q4km.gguf",
    "qwen3_tts_predictor-q4km.gguf": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/qwen3_tts_predictor-q4km.gguf",
    "tokenizer.json": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/tokenizer.json",
    "vocab.json": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/kevinwang676/Qwen3-TTS-GGUF/resolve/main/merges.txt",
}

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    # Ensure current directory is project root or assets?
    # We will download to CURRENT directory or specific 'models' directory.
    # Default to 'models' subdirectory.
    if not os.path.exists("models"):
        os.makedirs("models")
    
    os.chdir("models")
    
    for filename, url in MODELS.items():
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
