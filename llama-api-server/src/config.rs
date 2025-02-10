use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub chat: ChatConfig,
    pub embedding: EmbeddingConfig,
    pub tts: TtsConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub socket_addr: String,
    pub port: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatConfig {
    pub model_name: String,
    pub model_alias: String,
    pub ctx_size: i32,
    pub batch_size: i32,
    pub ubatch_size: i32,
    pub prompt_template: String,
    #[serde(default)]
    pub reverse_prompt: Option<String>,
    pub n_predict: i32,
    pub n_gpu_layers: i32,
    pub split_mode: String,
    #[serde(default)]
    pub main_gpu: Option<i32>,
    #[serde(default)]
    pub tensor_split: Option<String>,
    pub threads: i32,
    pub no_mmap: bool,
    pub temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub json_schema: Option<String>,
    #[serde(default)]
    pub llava_mmproj: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub model_alias: String,
    pub ctx_size: i32,
    pub batch_size: i32,
    pub ubatch_size: i32,
    pub prompt_template: String,
}

#[derive(Debug, Deserialize)]
pub struct TtsConfig {
    pub model_name: String,
    pub model_alias: String,
    pub codec_model: String,
    pub output_file: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
