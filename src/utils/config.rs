//! Config module
use std::path::PathBuf;
use structopt::StructOpt;
#[derive(Debug, StructOpt)]
#[structopt(name = "qwen3-tts")]
pub struct Args {
    #[structopt(short, long, default_value = "model-base")]
    pub model: PathBuf,
    #[structopt(short, long)]
    pub text: Option<String>,
    #[structopt(short, long)]
    pub voice: Option<PathBuf>,
    #[structopt(short, long, default_value = "chinese")]
    pub language: String,
    #[structopt(short, long, default_value = "0.8")]
    pub temperature: f32,
    #[structopt(short, long)]
    pub stream: bool,
    #[structopt(short, long)]
    pub output: Option<PathBuf>,
    #[structopt(short, long)]
    pub verbose: bool,
}
