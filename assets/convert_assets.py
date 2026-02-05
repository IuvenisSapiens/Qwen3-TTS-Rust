
import argparse
import os
import numpy as np
import gguf

def convert(input_dir, output_file):
    print(f"Loading NPY files from {input_dir}...")
    
    writer = gguf.GGUFWriter(output_file, "qwen3-assets")
    
    # 1. Proj Weight & Bias
    proj_weight_path = os.path.join(input_dir, "proj_weight.npy")
    proj_bias_path = os.path.join(input_dir, "proj_bias.npy")
    
    if os.path.exists(proj_weight_path):
        data = np.load(proj_weight_path).astype(np.float32)
        print(f"  Adding proj.weight: {data.shape}")
        writer.add_tensor("proj.weight", data)
    else:
        print(f"Warning: {proj_weight_path} not found")

    if os.path.exists(proj_bias_path):
        data = np.load(proj_bias_path).astype(np.float32)
        print(f"  Adding proj.bias: {data.shape}")
        writer.add_tensor("proj.bias", data)
    else:
        print(f"Warning: {proj_bias_path} not found")

    # 2. Text Embeddings
    text_emb_path = os.path.join(input_dir, "text_embedding_projected.npy")
    if os.path.exists(text_emb_path):
        data = np.load(text_emb_path).astype(np.float32)
        print(f"  Adding text_embd: {data.shape}")
        writer.add_tensor("text_embd", data)
    else:
        print(f"Warning: {text_emb_path} not found")

    # 3. Codec Embeddings (0-15)
    for i in range(16):
        name = f"codec_embedding_{i}.npy"
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            data = np.load(path).astype(np.float32)
            print(f"  Adding codec_embd.{i}: {data.shape}")
            writer.add_tensor(f"codec_embd.{i}", data)
        else:
            print(f"Warning: {path} not found")

    print(f"Writing GGUF to {output_file}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS NPY assets to GGUF")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing .npy files")
    parser.add_argument("--output_file", type=str, default="qwen3_assets.gguf", help="Output GGUF file path")
    
    args = parser.parse_args()
    convert(args.input_dir, args.output_file)
