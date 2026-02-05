use qwen3_tts::tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    let tokenizer = Tokenizer::load(Path::new(
        r"c:\work\ai00-tts\thrd\Qwen3-TTS-GGUF\model-base",
    ))
    .unwrap();
    let user_ids = tokenizer.encode("user");
    let assistant_ids = tokenizer.encode("assistant");
    println!("'user' ids: {:?}", user_ids);
    println!("'assistant' ids: {:?}", assistant_ids);

    let im_start_ids = tokenizer.encode("<|im_start|>");
    println!("'<|im_start|>' ids: {:?}", im_start_ids);

    let im_end_ids = tokenizer.encode("<|im_end|>");
    println!("'<|im_end|>' ids: {:?}", im_end_ids);

    let nl_ids = tokenizer.encode("\n");
    println!("'\\n' ids: {:?}", nl_ids);
}
