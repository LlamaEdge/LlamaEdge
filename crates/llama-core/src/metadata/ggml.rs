//! Define metadata for the ggml model.

use super::BaseMetadata;
use chat_prompts::PromptTemplateType;
use serde::{Deserialize, Serialize};

/// Builder for creating a ggml metadata
#[derive(Debug)]
pub struct GgmlMetadataBuilder {
    metadata: GgmlMetadata,
}
impl GgmlMetadataBuilder {
    pub fn new<S: Into<String>>(model_name: S, model_alias: S, pt: PromptTemplateType) -> Self {
        let metadata = GgmlMetadata {
            model_name: model_name.into(),
            model_alias: model_alias.into(),
            prompt_template: pt,
            ..Default::default()
        };

        Self { metadata }
    }

    pub fn with_prompt_template(mut self, template: PromptTemplateType) -> Self {
        self.metadata.prompt_template = template;
        self
    }

    pub fn enable_plugin_log(mut self, enable: bool) -> Self {
        self.metadata.log_enable = enable;
        self
    }

    pub fn enable_debug_log(mut self, enable: bool) -> Self {
        self.metadata.debug_log = enable;
        self
    }

    pub fn enable_prompts_log(mut self, enable: bool) -> Self {
        self.metadata.log_prompts = enable;
        self
    }

    pub fn enable_embeddings(mut self, enable: bool) -> Self {
        self.metadata.embeddings = enable;
        self
    }

    pub fn with_n_predict(mut self, n: i32) -> Self {
        self.metadata.n_predict = n;
        self
    }

    pub fn with_main_gpu(mut self, gpu: Option<u64>) -> Self {
        self.metadata.main_gpu = gpu;
        self
    }

    pub fn with_tensor_split(mut self, split: Option<String>) -> Self {
        self.metadata.tensor_split = split;
        self
    }

    pub fn with_threads(mut self, threads: u64) -> Self {
        self.metadata.threads = threads;
        self
    }

    pub fn with_reverse_prompt(mut self, prompt: Option<String>) -> Self {
        self.metadata.reverse_prompt = prompt;
        self
    }

    pub fn with_mmproj(mut self, path: Option<String>) -> Self {
        self.metadata.mmproj = path;
        self
    }

    pub fn with_image(mut self, path: impl Into<String>) -> Self {
        self.metadata.image = Some(path.into());
        self
    }

    pub fn with_n_gpu_layers(mut self, n: u64) -> Self {
        self.metadata.n_gpu_layers = n;
        self
    }

    pub fn disable_mmap(mut self, disable: Option<bool>) -> Self {
        self.metadata.use_mmap = disable.map(|v| !v);
        self
    }

    pub fn with_split_mode(mut self, mode: String) -> Self {
        self.metadata.split_mode = mode;
        self
    }

    pub fn with_ctx_size(mut self, size: u64) -> Self {
        self.metadata.ctx_size = size;
        self
    }

    pub fn with_batch_size(mut self, size: u64) -> Self {
        self.metadata.batch_size = size;
        self
    }

    pub fn with_ubatch_size(mut self, size: u64) -> Self {
        self.metadata.ubatch_size = size;
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.metadata.temperature = temp;
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.metadata.top_p = top_p;
        self
    }

    pub fn with_repeat_penalty(mut self, penalty: f64) -> Self {
        self.metadata.repeat_penalty = penalty;
        self
    }

    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.metadata.presence_penalty = penalty;
        self
    }

    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.metadata.frequency_penalty = penalty;
        self
    }

    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.metadata.grammar = grammar.into();
        self
    }

    pub fn with_json_schema(mut self, schema: Option<String>) -> Self {
        self.metadata.json_schema = schema;
        self
    }

    pub fn build(self) -> GgmlMetadata {
        self.metadata
    }
}

/// Metadata for chat and embeddings models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GgmlMetadata {
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_name: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_alias: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub log_prompts: bool,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub prompt_template: PromptTemplateType,

    // * Plugin parameters (used by this plugin):
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    #[serde(rename = "enable-debug-log")]
    pub debug_log: bool,
    // #[serde(rename = "stream-stdout")]
    // pub stream_stdout: bool,
    #[serde(rename = "embedding")]
    pub embeddings: bool,
    /// Number of tokens to predict, -1 = infinity, -2 = until context filled. Defaults to -1.
    #[serde(rename = "n-predict")]
    pub n_predict: i32,
    /// Halt generation at PROMPT, return control in interactive mode.
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    pub reverse_prompt: Option<String>,
    /// path to the multimodal projector file for llava
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// Path to the image file for llava
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    // * Model parameters (need to reload the model if updated):
    #[serde(rename = "n-gpu-layers")]
    pub n_gpu_layers: u64,
    /// The main GPU to use. Defaults to None.
    #[serde(rename = "main-gpu")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u64>,
    /// How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1. Defaults to None.
    #[serde(rename = "tensor-split")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<String>,
    /// Whether to use memory-mapped files for the model. Defaults to `true`.
    #[serde(skip_serializing_if = "Option::is_none", rename = "use-mmap")]
    pub use_mmap: Option<bool>,
    /// How to split the model across multiple GPUs. Possible values:
    /// - `none`: use one GPU only
    /// - `layer`: split layers and KV across GPUs (default)
    /// - `row`: split rows across GPUs
    #[serde(rename = "split-mode")]
    pub split_mode: String,

    // * Context parameters (used by the llama context):
    /// Size of the prompt context. 0 means loaded from model. Defaults to 4096.
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    /// Logical maximum batch size. Defaults to 2048.
    #[serde(rename = "batch-size")]
    pub batch_size: u64,
    /// Physical maximum batch size. Defaults to 512.
    #[serde(rename = "ubatch-size")]
    pub ubatch_size: u64,
    /// Number of threads to use during generation. Defaults to 2.
    #[serde(rename = "threads")]
    pub threads: u64,

    // * Sampling parameters (used by the llama sampling context).
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Defaults to 0.8.
    #[serde(rename = "temp")]
    pub temperature: f64,
    /// Top-p sampling. Between 0.0 and 1.0. Defaults to 0.9.
    #[serde(rename = "top-p")]
    pub top_p: f64,
    /// Penalize repeat sequence of tokens. Defaults to 1.0.
    #[serde(rename = "repeat-penalty")]
    pub repeat_penalty: f64,
    /// Repeat alpha presence penalty. Defaults to 0.0.
    #[serde(rename = "presence-penalty")]
    pub presence_penalty: f64,
    /// Repeat alpha frequency penalty. Defaults to 0.0.
    #[serde(rename = "frequency-penalty")]
    pub frequency_penalty: f64,

    // * grammar parameters
    /// BNF-like grammar to constrain generations (see samples in grammars/ dir). Defaults to empty string.
    pub grammar: String,
    /// JSON schema to constrain generations (<https://json-schema.org/>), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
}
impl Default for GgmlMetadata {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_alias: String::new(),
            log_prompts: false,
            debug_log: false,
            prompt_template: PromptTemplateType::Llama2Chat,
            log_enable: false,
            embeddings: false,
            n_predict: -1,
            reverse_prompt: None,
            mmproj: None,
            image: None,
            n_gpu_layers: 100,
            main_gpu: None,
            tensor_split: None,
            use_mmap: Some(true),
            split_mode: "layer".to_string(),
            ctx_size: 4096,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 2,
            temperature: 0.8,
            top_p: 0.9,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            grammar: String::new(),
            json_schema: None,
        }
    }
}
impl BaseMetadata for GgmlMetadata {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_alias(&self) -> &str {
        &self.model_alias
    }
}
impl GgmlMetadata {
    pub fn prompt_template(&self) -> PromptTemplateType {
        self.prompt_template
    }
}
