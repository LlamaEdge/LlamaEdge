use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) server: ServerConfig,
    pub(crate) chat: ChatConfig,
    pub(crate) embedding: EmbeddingConfig,
    pub(crate) tts: TtsConfig,
}
impl Config {
    pub(crate) fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ServerConfig {
    pub(crate) socket_addr: String,
    pub(crate) port: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) ctx_size: i32,
    pub(crate) batch_size: i32,
    pub(crate) ubatch_size: i32,
    pub(crate) prompt_template: String,
    #[serde(default)]
    pub(crate) reverse_prompt: Option<String>,
    pub(crate) n_predict: i32,
    pub(crate) n_gpu_layers: i32,
    pub(crate) split_mode: String,
    #[serde(default)]
    pub(crate) main_gpu: Option<i32>,
    #[serde(default)]
    pub(crate) tensor_split: Option<String>,
    pub(crate) threads: i32,
    pub(crate) no_mmap: bool,
    pub(crate) temp: f32,
    pub(crate) top_p: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) frequency_penalty: f32,
    #[serde(default)]
    pub(crate) grammar: Option<String>,
    #[serde(default)]
    pub(crate) json_schema: Option<String>,
    #[serde(default)]
    pub(crate) llava_mmproj: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) ctx_size: i32,
    pub(crate) batch_size: i32,
    pub(crate) ubatch_size: i32,
    pub(crate) prompt_template: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TtsConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) codec_model: String,
    pub(crate) output_file: String,
}
