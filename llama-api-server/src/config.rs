use crate::ServerError;
use chat_prompts::PromptTemplateType;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Config {
    pub(crate) server: ServerConfig,
    pub(crate) chat: ChatConfig,
    pub(crate) embedding: EmbeddingConfig,
    pub(crate) tts: TtsConfig,
}
impl Config {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, ServerError> {
        let content =
            fs::read_to_string(path.as_ref()).map_err(|e| ServerError::Operation(e.to_string()))?;
        let config: Config =
            toml::from_str(&content).map_err(|e| ServerError::Operation(e.to_string()))?;
        Ok(config)
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ServerConfig {
    pub(crate) socket_addr: SocketAddr,
}
impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            socket_addr: "0.0.0.0:8080".parse().unwrap(),
        }
    }
}
impl<'de> Deserialize<'de> for ServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        #[serde(rename = "server")]
        struct Helper {
            socket_addr: String,
        }

        let helper = Helper::deserialize(deserializer)?;
        let socket_addr = helper.socket_addr.parse().map_err(|e| {
            Error::custom(format!(
                "Failed to parse socket address from config file: {}",
                e
            ))
        })?;

        Ok(ServerConfig { socket_addr })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) ctx_size: i32,
    pub(crate) batch_size: i32,
    pub(crate) ubatch_size: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_template: Option<PromptTemplateType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reverse_prompt: Option<String>,
    pub(crate) n_predict: i32,
    pub(crate) n_gpu_layers: i32,
    pub(crate) split_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) main_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tensor_split: Option<String>,
    pub(crate) threads: i32,
    pub(crate) no_mmap: bool,
    pub(crate) temp: f32,
    pub(crate) top_p: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) grammar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) json_schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) llava_mmproj: Option<PathBuf>,
}
impl Default for ChatConfig {
    fn default() -> Self {
        ChatConfig {
            model_name: "default".to_string(),
            model_alias: "chat".to_string(),
            ctx_size: 4096,
            batch_size: 512,
            ubatch_size: 512,
            prompt_template: None,
            reverse_prompt: None,
            n_predict: -1,
            n_gpu_layers: 100,
            split_mode: "layer".to_string(),
            main_gpu: None,
            tensor_split: None,
            threads: 2,
            no_mmap: true,
            temp: 1.0,
            top_p: 1.0,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            grammar: None,
            json_schema: None,
            llava_mmproj: None,
        }
    }
}
impl<'de> Deserialize<'de> for ChatConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        #[serde(rename = "chat")]
        struct Helper {
            model_name: String,
            model_alias: String,
            ctx_size: i32,
            batch_size: i32,
            ubatch_size: i32,
            prompt_template: Option<String>,
            reverse_prompt: Option<String>,
            n_predict: i32,
            n_gpu_layers: i32,
            split_mode: String,
            main_gpu: Option<i32>,
            tensor_split: Option<String>,
            threads: i32,
            no_mmap: bool,
            temp: f32,
            top_p: f32,
            repeat_penalty: f32,
            presence_penalty: f32,
            frequency_penalty: f32,
            grammar: Option<String>,
            json_schema: Option<String>,
            llava_mmproj: Option<String>,
        }

        let helper = Helper::deserialize(deserializer)?;

        // prompt_template
        let prompt_template = if let Some(template) = helper.prompt_template {
            let prompt_template = template.parse::<PromptTemplateType>().map_err(|e| {
                Error::custom(format!(
                    "Failed to parse prompt_template from config file: {}",
                    e
                ))
            })?;

            Some(prompt_template)
        } else {
            None
        };

        // grammar
        let grammar = helper.grammar.filter(|grammar| !grammar.is_empty());

        // json_schema
        let json_schema = helper
            .json_schema
            .filter(|json_schema| !json_schema.is_empty());

        // llava_mmproj
        let llava_mmproj = if let Some(llava_mmproj) = helper.llava_mmproj {
            if !llava_mmproj.is_empty() {
                Some(PathBuf::from(llava_mmproj))
            } else {
                None
            }
        } else {
            None
        };

        Ok(ChatConfig {
            model_name: helper.model_name,
            model_alias: helper.model_alias,
            ctx_size: helper.ctx_size,
            batch_size: helper.batch_size,
            ubatch_size: helper.ubatch_size,
            prompt_template,
            reverse_prompt: helper.reverse_prompt,
            n_predict: helper.n_predict,
            n_gpu_layers: helper.n_gpu_layers,
            split_mode: helper.split_mode,
            main_gpu: helper.main_gpu,
            tensor_split: helper.tensor_split,
            threads: helper.threads,
            no_mmap: helper.no_mmap,
            temp: helper.temp,
            top_p: helper.top_p,
            repeat_penalty: helper.repeat_penalty,
            presence_penalty: helper.presence_penalty,
            frequency_penalty: helper.frequency_penalty,
            grammar,
            json_schema,
            llava_mmproj,
        })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) ctx_size: i32,
    pub(crate) batch_size: i32,
    pub(crate) ubatch_size: i32,
    pub(crate) prompt_template: PromptTemplateType,
}
impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            model_name: "default".to_string(),
            model_alias: "embedding".to_string(),
            ctx_size: 384,
            batch_size: 512,
            ubatch_size: 512,
            prompt_template: PromptTemplateType::Embedding,
        }
    }
}
impl<'de> Deserialize<'de> for EmbeddingConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        #[serde(rename = "embedding")]
        struct Helper {
            model_name: String,
            model_alias: String,
            ctx_size: i32,
            batch_size: i32,
            ubatch_size: i32,
            prompt_template: String,
        }

        let helper = Helper::deserialize(deserializer)?;

        let prompt_template = helper
            .prompt_template
            .parse::<PromptTemplateType>()
            .map_err(|e| {
                Error::custom(format!(
                    "Failed to parse prompt_template from config file: {}",
                    e
                ))
            })?;

        Ok(EmbeddingConfig {
            model_name: helper.model_name,
            model_alias: helper.model_alias,
            ctx_size: helper.ctx_size,
            batch_size: helper.batch_size,
            ubatch_size: helper.ubatch_size,
            prompt_template,
        })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct TtsConfig {
    pub(crate) model_name: String,
    pub(crate) model_alias: String,
    pub(crate) codec_model: PathBuf,
    pub(crate) output_file: String,
}
impl Default for TtsConfig {
    fn default() -> Self {
        TtsConfig {
            model_name: "default".to_string(),
            model_alias: "tts".to_string(),
            codec_model: PathBuf::from(""),
            output_file: "output.wav".to_string(),
        }
    }
}
impl<'de> Deserialize<'de> for TtsConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename = "tts")]
        struct Helper {
            model_name: String,
            model_alias: String,
            codec_model: String,
            output_file: String,
        }

        let helper = Helper::deserialize(deserializer)?;
        let codec_model = PathBuf::from(helper.codec_model);

        Ok(TtsConfig {
            model_name: helper.model_name,
            model_alias: helper.model_alias,
            codec_model,
            output_file: helper.output_file,
        })
    }
}
