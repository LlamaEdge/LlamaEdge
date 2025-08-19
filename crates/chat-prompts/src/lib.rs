//! `chat-prompts` is part of [LlamaEdge API Server](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) project. It provides a collection of prompt templates that are used to generate prompts for the LLMs (See models in [huggingface.co/second-state](https://huggingface.co/second-state)).
//!
//! For the details of available prompt templates, see [README.md](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server/chat-prompts).

pub mod agent;
pub mod chat;
pub mod error;
pub mod utils;

use agent::*;
use chat::*;
use clap::ValueEnum;
use endpoints::chat::{ChatCompletionRequestMessage, Tool};
use error::Result;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Define the chat prompt template types.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum PromptTemplateType {
    #[value(name = "llama-2-chat")]
    Llama2Chat,
    #[value(name = "llama-3-chat")]
    Llama3Chat,
    #[value(name = "llama-3-tool")]
    Llama3Tool,
    #[value(name = "llama-4-chat")]
    Llama4Chat,
    #[value(name = "mistral-instruct")]
    MistralInstruct,
    #[value(name = "mistral-tool")]
    MistralTool,
    #[value(name = "mistrallite")]
    MistralLite,
    #[value(name = "mistral-small-chat")]
    MistralSmallChat,
    #[value(name = "mistral-small-tool")]
    MistralSmallTool,
    #[value(name = "openchat")]
    OpenChat,
    #[value(name = "codellama-instruct")]
    CodeLlama,
    #[value(name = "codellama-super-instruct")]
    CodeLlamaSuper,
    #[value(name = "human-assistant")]
    HumanAssistant,
    #[value(name = "vicuna-1.0-chat")]
    VicunaChat,
    #[value(name = "vicuna-1.1-chat")]
    Vicuna11Chat,
    #[value(name = "vicuna-llava")]
    VicunaLlava,
    #[value(name = "chatml")]
    ChatML,
    #[value(name = "chatml-tool")]
    ChatMLTool,
    #[value(name = "chatml-think")]
    ChatMLThink,
    #[value(name = "internlm-2-tool")]
    InternLM2Tool,
    #[value(name = "baichuan-2")]
    Baichuan2,
    #[value(name = "wizard-coder")]
    WizardCoder,
    #[value(name = "zephyr")]
    Zephyr,
    #[value(name = "stablelm-zephyr")]
    StableLMZephyr,
    #[value(name = "intel-neural")]
    IntelNeural,
    #[value(name = "deepseek-chat")]
    DeepseekChat,
    #[value(name = "deepseek-coder")]
    DeepseekCoder,
    #[value(name = "deepseek-chat-2")]
    DeepseekChat2,
    #[value(name = "deepseek-chat-25")]
    DeepseekChat25,
    #[value(name = "deepseek-chat-3")]
    DeepseekChat3,
    #[value(name = "solar-instruct")]
    SolarInstruct,
    #[value(name = "phi-2-chat")]
    Phi2Chat,
    #[value(name = "phi-2-instruct")]
    Phi2Instruct,
    #[value(name = "phi-3-chat")]
    Phi3Chat,
    #[value(name = "phi-3-instruct")]
    Phi3Instruct,
    #[value(name = "phi-4-chat")]
    Phi4Chat,
    #[value(name = "gemma-instruct")]
    GemmaInstruct,
    #[value(name = "gemma-3")]
    Gemma3,
    #[value(name = "octopus")]
    Octopus,
    #[value(name = "glm-4-chat")]
    Glm4Chat,
    #[value(name = "groq-llama3-tool")]
    GroqLlama3Tool,
    #[value(name = "mediatek-breeze")]
    BreezeInstruct,
    #[value(name = "nemotron-chat")]
    NemotronChat,
    #[value(name = "nemotron-tool")]
    NemotronTool,
    #[value(name = "functionary-32")]
    FunctionaryV32,
    #[value(name = "functionary-31")]
    FunctionaryV31,
    #[value(name = "minicpmv")]
    MiniCPMV,
    #[value(name = "moxin-chat")]
    MoxinChat,
    #[value(name = "moxin-instruct")]
    MoxinInstruct,
    #[value(name = "falcon3")]
    Falcon3,
    #[value(name = "megrez")]
    Megrez,
    #[value(name = "qwen2-vision")]
    Qwen2vl,
    #[value(name = "qwen3-no-think")]
    Qwen3NoThink,
    #[value(name = "qwen3-agent")]
    Qwen3Agent,
    #[value(name = "exaone-deep-chat")]
    ExaoneDeepChat,
    #[value(name = "exaone-chat")]
    ExaoneChat,
    #[value(name = "seed-instruct")]
    SeedInstruct,
    #[value(name = "seed-reasoning")]
    SeedReasoning,
    #[value(name = "smol-vision")]
    Smolvl,
    #[value(name = "smol3-no-think")]
    Smol3NoThink,
    #[value(name = "gpt-oss")]
    GptOss,
    #[value(name = "embedding")]
    Embedding,
    #[value(name = "tts")]
    Tts,
    #[value(name = "none")]
    Null,
}
impl PromptTemplateType {
    /// Check if the prompt template has a system prompt.
    pub fn has_system_prompt(&self) -> bool {
        match self {
            PromptTemplateType::Llama2Chat
            | PromptTemplateType::Llama3Chat
            | PromptTemplateType::Llama3Tool
            | PromptTemplateType::CodeLlama
            | PromptTemplateType::CodeLlamaSuper
            | PromptTemplateType::VicunaChat
            | PromptTemplateType::VicunaLlava
            | PromptTemplateType::ChatML
            | PromptTemplateType::ChatMLTool
            | PromptTemplateType::InternLM2Tool
            | PromptTemplateType::Baichuan2
            | PromptTemplateType::WizardCoder
            | PromptTemplateType::Zephyr
            | PromptTemplateType::IntelNeural
            | PromptTemplateType::DeepseekCoder
            | PromptTemplateType::DeepseekChat2
            | PromptTemplateType::DeepseekChat3
            | PromptTemplateType::Octopus
            | PromptTemplateType::Phi3Chat
            | PromptTemplateType::Phi4Chat
            | PromptTemplateType::Glm4Chat
            | PromptTemplateType::GroqLlama3Tool
            | PromptTemplateType::BreezeInstruct
            | PromptTemplateType::DeepseekChat25
            | PromptTemplateType::NemotronChat
            | PromptTemplateType::NemotronTool
            | PromptTemplateType::MiniCPMV
            | PromptTemplateType::MoxinChat
            | PromptTemplateType::Falcon3
            | PromptTemplateType::Megrez
            | PromptTemplateType::Qwen2vl
            | PromptTemplateType::Qwen3NoThink
            | PromptTemplateType::Qwen3Agent
            | PromptTemplateType::MistralSmallChat
            | PromptTemplateType::MistralSmallTool
            | PromptTemplateType::ExaoneDeepChat
            | PromptTemplateType::ExaoneChat
            | PromptTemplateType::ChatMLThink
            | PromptTemplateType::Llama4Chat
            | PromptTemplateType::SeedInstruct
            | PromptTemplateType::Smol3NoThink
            | PromptTemplateType::GptOss => true,
            PromptTemplateType::MistralInstruct
            | PromptTemplateType::MistralTool
            | PromptTemplateType::MistralLite
            | PromptTemplateType::HumanAssistant
            | PromptTemplateType::DeepseekChat
            | PromptTemplateType::GemmaInstruct
            | PromptTemplateType::Gemma3
            | PromptTemplateType::OpenChat
            | PromptTemplateType::Phi2Chat
            | PromptTemplateType::Phi2Instruct
            | PromptTemplateType::Phi3Instruct
            | PromptTemplateType::SolarInstruct
            | PromptTemplateType::Vicuna11Chat
            | PromptTemplateType::StableLMZephyr
            | PromptTemplateType::FunctionaryV32
            | PromptTemplateType::FunctionaryV31
            | PromptTemplateType::SeedReasoning
            | PromptTemplateType::MoxinInstruct
            | PromptTemplateType::Smolvl
            | PromptTemplateType::Embedding
            | PromptTemplateType::Tts
            | PromptTemplateType::Null => false,
        }
    }

    /// Check if the prompt template supports image input.
    pub fn is_image_supported(&self) -> bool {
        matches!(
            self,
            PromptTemplateType::MiniCPMV
                | PromptTemplateType::Qwen2vl
                | PromptTemplateType::VicunaLlava
                | PromptTemplateType::Gemma3
                | PromptTemplateType::Smolvl
        )
    }
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "llama-3-chat" => Ok(PromptTemplateType::Llama3Chat),
            "llama-3-tool" => Ok(PromptTemplateType::Llama3Tool),
            "llama-4-chat" => Ok(PromptTemplateType::Llama4Chat),
            "mistral-instruct" => Ok(PromptTemplateType::MistralInstruct),
            "mistral-tool" => Ok(PromptTemplateType::MistralTool),
            "mistrallite" => Ok(PromptTemplateType::MistralLite),
            "mistral-small-chat" => Ok(PromptTemplateType::MistralSmallChat),
            "mistral-small-tool" => Ok(PromptTemplateType::MistralSmallTool),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "codellama-super-instruct" => Ok(PromptTemplateType::CodeLlamaSuper),
            "belle-llama-2-chat" => Ok(PromptTemplateType::HumanAssistant),
            "human-assistant" => Ok(PromptTemplateType::HumanAssistant),
            "vicuna-1.0-chat" => Ok(PromptTemplateType::VicunaChat),
            "vicuna-1.1-chat" => Ok(PromptTemplateType::Vicuna11Chat),
            "vicuna-llava" => Ok(PromptTemplateType::VicunaLlava),
            "chatml" => Ok(PromptTemplateType::ChatML),
            "chatml-tool" => Ok(PromptTemplateType::ChatMLTool),
            "chatml-think" => Ok(PromptTemplateType::ChatMLThink),
            "internlm-2-tool" => Ok(PromptTemplateType::InternLM2Tool),
            "openchat" => Ok(PromptTemplateType::OpenChat),
            "baichuan-2" => Ok(PromptTemplateType::Baichuan2),
            "wizard-coder" => Ok(PromptTemplateType::WizardCoder),
            "zephyr" => Ok(PromptTemplateType::Zephyr),
            "stablelm-zephyr" => Ok(PromptTemplateType::StableLMZephyr),
            "intel-neural" => Ok(PromptTemplateType::IntelNeural),
            "deepseek-chat" => Ok(PromptTemplateType::DeepseekChat),
            "deepseek-coder" => Ok(PromptTemplateType::DeepseekCoder),
            "deepseek-chat-2" => Ok(PromptTemplateType::DeepseekChat2),
            "deepseek-chat-25" => Ok(PromptTemplateType::DeepseekChat25),
            "deepseek-chat-3" => Ok(PromptTemplateType::DeepseekChat3),
            "solar-instruct" => Ok(PromptTemplateType::SolarInstruct),
            "phi-2-chat" => Ok(PromptTemplateType::Phi2Chat),
            "phi-2-instruct" => Ok(PromptTemplateType::Phi2Instruct),
            "phi-3-chat" => Ok(PromptTemplateType::Phi3Chat),
            "phi-3-instruct" => Ok(PromptTemplateType::Phi3Instruct),
            "phi-4-chat" => Ok(PromptTemplateType::Phi4Chat),
            "gemma-instruct" => Ok(PromptTemplateType::GemmaInstruct),
            "gemma-3" => Ok(PromptTemplateType::Gemma3),
            "octopus" => Ok(PromptTemplateType::Octopus),
            "glm-4-chat" => Ok(PromptTemplateType::Glm4Chat),
            "groq-llama3-tool" => Ok(PromptTemplateType::GroqLlama3Tool),
            "mediatek-breeze" => Ok(PromptTemplateType::BreezeInstruct),
            "nemotron-chat" => Ok(PromptTemplateType::NemotronChat),
            "nemotron-tool" => Ok(PromptTemplateType::NemotronTool),
            "functionary-32" => Ok(PromptTemplateType::FunctionaryV32),
            "functionary-31" => Ok(PromptTemplateType::FunctionaryV31),
            "minicpmv" => Ok(PromptTemplateType::MiniCPMV),
            "moxin-chat" => Ok(PromptTemplateType::MoxinChat),
            "moxin-instruct" => Ok(PromptTemplateType::MoxinInstruct),
            "falcon3" => Ok(PromptTemplateType::Falcon3),
            "megrez" => Ok(PromptTemplateType::Megrez),
            "qwen2-vision" => Ok(PromptTemplateType::Qwen2vl),
            "qwen3-no-think" => Ok(PromptTemplateType::Qwen3NoThink),
            "qwen3-agent" => Ok(PromptTemplateType::Qwen3Agent),
            "exaone-deep-chat" => Ok(PromptTemplateType::ExaoneDeepChat),
            "exaone-chat" => Ok(PromptTemplateType::ExaoneChat),
            "seed-instruct" => Ok(PromptTemplateType::SeedInstruct),
            "seed-reasoning" => Ok(PromptTemplateType::SeedReasoning),
            "smol-vision" => Ok(PromptTemplateType::Smolvl),
            "smol3-no-think" => Ok(PromptTemplateType::Smol3NoThink),
            "gpt-oss" => Ok(PromptTemplateType::GptOss),
            "embedding" => Ok(PromptTemplateType::Embedding),
            "tts" => Ok(PromptTemplateType::Tts),
            "none" => Ok(PromptTemplateType::Null),
            _ => Err(error::PromptError::UnknownPromptTemplateType(
                template.to_string(),
            )),
        }
    }
}
impl std::fmt::Display for PromptTemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptTemplateType::Llama2Chat => write!(f, "llama-2-chat"),
            PromptTemplateType::Llama3Chat => write!(f, "llama-3-chat"),
            PromptTemplateType::Llama3Tool => write!(f, "llama-3-tool"),
            PromptTemplateType::Llama4Chat => write!(f, "llama-4-chat"),
            PromptTemplateType::MistralInstruct => write!(f, "mistral-instruct"),
            PromptTemplateType::MistralTool => write!(f, "mistral-tool"),
            PromptTemplateType::MistralLite => write!(f, "mistrallite"),
            PromptTemplateType::MistralSmallChat => write!(f, "mistral-small-chat"),
            PromptTemplateType::MistralSmallTool => write!(f, "mistral-small-tool"),
            PromptTemplateType::OpenChat => write!(f, "openchat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::HumanAssistant => write!(f, "human-asistant"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-1.0-chat"),
            PromptTemplateType::Vicuna11Chat => write!(f, "vicuna-1.1-chat"),
            PromptTemplateType::VicunaLlava => write!(f, "vicuna-llava"),
            PromptTemplateType::ChatML => write!(f, "chatml"),
            PromptTemplateType::ChatMLTool => write!(f, "chatml-tool"),
            PromptTemplateType::ChatMLThink => write!(f, "chatml-think"),
            PromptTemplateType::InternLM2Tool => write!(f, "internlm-2-tool"),
            PromptTemplateType::Baichuan2 => write!(f, "baichuan-2"),
            PromptTemplateType::WizardCoder => write!(f, "wizard-coder"),
            PromptTemplateType::Zephyr => write!(f, "zephyr"),
            PromptTemplateType::StableLMZephyr => write!(f, "stablelm-zephyr"),
            PromptTemplateType::IntelNeural => write!(f, "intel-neural"),
            PromptTemplateType::DeepseekChat => write!(f, "deepseek-chat"),
            PromptTemplateType::DeepseekCoder => write!(f, "deepseek-coder"),
            PromptTemplateType::DeepseekChat2 => write!(f, "deepseek-chat-2"),
            PromptTemplateType::DeepseekChat25 => write!(f, "deepseek-chat-25"),
            PromptTemplateType::DeepseekChat3 => write!(f, "deepseek-chat-3"),
            PromptTemplateType::SolarInstruct => write!(f, "solar-instruct"),
            PromptTemplateType::Phi2Chat => write!(f, "phi-2-chat"),
            PromptTemplateType::Phi2Instruct => write!(f, "phi-2-instruct"),
            PromptTemplateType::Phi3Chat => write!(f, "phi-3-chat"),
            PromptTemplateType::Phi3Instruct => write!(f, "phi-3-instruct"),
            PromptTemplateType::Phi4Chat => write!(f, "phi-4-chat"),
            PromptTemplateType::CodeLlamaSuper => write!(f, "codellama-super-instruct"),
            PromptTemplateType::GemmaInstruct => write!(f, "gemma-instruct"),
            PromptTemplateType::Gemma3 => write!(f, "gemma-3"),
            PromptTemplateType::Octopus => write!(f, "octopus"),
            PromptTemplateType::Glm4Chat => write!(f, "glm-4-chat"),
            PromptTemplateType::GroqLlama3Tool => write!(f, "groq-llama3-tool"),
            PromptTemplateType::BreezeInstruct => write!(f, "mediatek-breeze"),
            PromptTemplateType::NemotronChat => write!(f, "nemotron-chat"),
            PromptTemplateType::NemotronTool => write!(f, "nemotron-tool"),
            PromptTemplateType::FunctionaryV32 => write!(f, "functionary-32"),
            PromptTemplateType::FunctionaryV31 => write!(f, "functionary-31"),
            PromptTemplateType::MiniCPMV => write!(f, "minicpmv"),
            PromptTemplateType::MoxinChat => write!(f, "moxin-chat"),
            PromptTemplateType::MoxinInstruct => write!(f, "moxin-instruct"),
            PromptTemplateType::Falcon3 => write!(f, "falcon3"),
            PromptTemplateType::Megrez => write!(f, "megrez"),
            PromptTemplateType::Qwen2vl => write!(f, "qwen2-vision"),
            PromptTemplateType::Qwen3NoThink => write!(f, "qwen3-no-think"),
            PromptTemplateType::Qwen3Agent => write!(f, "qwen3-agent"),
            PromptTemplateType::ExaoneDeepChat => write!(f, "exaone-deep-chat"),
            PromptTemplateType::ExaoneChat => write!(f, "exaone-chat"),
            PromptTemplateType::SeedInstruct => write!(f, "seed-instruct"),
            PromptTemplateType::SeedReasoning => write!(f, "seed-reasoning"),
            PromptTemplateType::Smolvl => write!(f, "smol-vision"),
            PromptTemplateType::Smol3NoThink => write!(f, "smol3-no-think"),
            PromptTemplateType::GptOss => write!(f, "gpt-oss"),
            PromptTemplateType::Embedding => write!(f, "embedding"),
            PromptTemplateType::Tts => write!(f, "tts"),
            PromptTemplateType::Null => write!(f, "none"),
        }
    }
}

#[enum_dispatch::enum_dispatch(BuildChatPrompt)]
pub enum ChatPrompt {
    Llama2ChatPrompt,
    Llama3ChatPrompt,
    Llama3ToolPrompt,
    Llama4ChatPrompt,
    MistralInstructPrompt,
    MistralToolPrompt,
    MistralLitePrompt,
    MistralSmallChatPrompt,
    MistralSmallToolPrompt,
    OpenChatPrompt,
    CodeLlamaInstructPrompt,
    CodeLlamaSuperInstructPrompt,
    HumanAssistantChatPrompt,
    /// Vicuna 1.0
    VicunaChatPrompt,
    /// Vicuna 1.1
    Vicuna11ChatPrompt,
    VicunaLlavaPrompt,
    ChatMLPrompt,
    ChatMLToolPrompt,
    ChatMLThinkPrompt,
    InternLM2ToolPrompt,
    Baichuan2ChatPrompt,
    WizardCoderPrompt,
    ZephyrChatPrompt,
    StableLMZephyrChatPrompt,
    NeuralChatPrompt,
    DeepseekChatPrompt,
    DeepseekCoderPrompt,
    DeepseekChat2Prompt,
    DeepseekChat25Prompt,
    SolarInstructPrompt,
    Phi2ChatPrompt,
    Phi2InstructPrompt,
    Phi3ChatPrompt,
    Phi3InstructPrompt,
    Phi4ChatPrompt,
    GemmaInstructPrompt,
    Gemma3Prompt,
    OctopusPrompt,
    Glm4ChatPrompt,
    GroqLlama3ToolPrompt,
    BreezeInstructPrompt,
    NemotronChatPrompt,
    NemotronToolPrompt,
    FunctionaryV32ToolPrompt,
    FunctionaryV31ToolPrompt,
    MiniCPMVPrompt,
    MoxinChatPrompt,
    MoxinInstructPrompt,
    FalconChatPrompt,
    MegrezPrompt,
    Qwen2vlPrompt,
    Qwen3NoThinkPrompt,
    Qwen3AgentPrompt,
    ExaoneDeepChatPrompt,
    ExaoneChatPrompt,
    SeedInstructPrompt,
    SeedReasoningPrompt,
    SmolvlPrompt,
    Smol3NoThinkPrompt,
    GptOssPrompt,
}
impl From<PromptTemplateType> for ChatPrompt {
    fn from(ty: PromptTemplateType) -> Self {
        match ty {
            PromptTemplateType::Llama2Chat => ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt),
            PromptTemplateType::Llama3Chat => ChatPrompt::Llama3ChatPrompt(Llama3ChatPrompt),
            PromptTemplateType::Llama3Tool => ChatPrompt::Llama3ToolPrompt(Llama3ToolPrompt),
            PromptTemplateType::Llama4Chat => ChatPrompt::Llama4ChatPrompt(Llama4ChatPrompt),
            PromptTemplateType::MistralInstruct => {
                ChatPrompt::MistralInstructPrompt(MistralInstructPrompt)
            }
            PromptTemplateType::MistralTool => ChatPrompt::MistralToolPrompt(MistralToolPrompt),
            PromptTemplateType::MistralLite => ChatPrompt::MistralLitePrompt(MistralLitePrompt),
            PromptTemplateType::MistralSmallChat => {
                ChatPrompt::MistralSmallChatPrompt(MistralSmallChatPrompt)
            }
            PromptTemplateType::MistralSmallTool => {
                ChatPrompt::MistralSmallToolPrompt(MistralSmallToolPrompt)
            }
            PromptTemplateType::OpenChat => ChatPrompt::OpenChatPrompt(OpenChatPrompt),
            PromptTemplateType::CodeLlama => {
                ChatPrompt::CodeLlamaInstructPrompt(CodeLlamaInstructPrompt)
            }
            PromptTemplateType::CodeLlamaSuper => {
                ChatPrompt::CodeLlamaSuperInstructPrompt(CodeLlamaSuperInstructPrompt)
            }
            PromptTemplateType::HumanAssistant => {
                ChatPrompt::HumanAssistantChatPrompt(HumanAssistantChatPrompt)
            }
            PromptTemplateType::VicunaChat => ChatPrompt::VicunaChatPrompt(VicunaChatPrompt),
            PromptTemplateType::Vicuna11Chat => ChatPrompt::Vicuna11ChatPrompt(Vicuna11ChatPrompt),
            PromptTemplateType::VicunaLlava => ChatPrompt::VicunaLlavaPrompt(VicunaLlavaPrompt),
            PromptTemplateType::ChatML => ChatPrompt::ChatMLPrompt(ChatMLPrompt),
            PromptTemplateType::ChatMLTool => ChatPrompt::ChatMLToolPrompt(ChatMLToolPrompt),
            PromptTemplateType::ChatMLThink => ChatPrompt::ChatMLThinkPrompt(ChatMLThinkPrompt),
            PromptTemplateType::InternLM2Tool => {
                ChatPrompt::InternLM2ToolPrompt(InternLM2ToolPrompt)
            }
            PromptTemplateType::Baichuan2 => ChatPrompt::Baichuan2ChatPrompt(Baichuan2ChatPrompt),
            PromptTemplateType::WizardCoder => ChatPrompt::WizardCoderPrompt(WizardCoderPrompt),
            PromptTemplateType::Zephyr => ChatPrompt::ZephyrChatPrompt(ZephyrChatPrompt),
            PromptTemplateType::StableLMZephyr => {
                ChatPrompt::StableLMZephyrChatPrompt(StableLMZephyrChatPrompt)
            }
            PromptTemplateType::IntelNeural => ChatPrompt::NeuralChatPrompt(NeuralChatPrompt),
            PromptTemplateType::DeepseekChat => ChatPrompt::DeepseekChatPrompt(DeepseekChatPrompt),
            PromptTemplateType::DeepseekCoder => {
                ChatPrompt::DeepseekCoderPrompt(DeepseekCoderPrompt)
            }
            PromptTemplateType::DeepseekChat2 => {
                ChatPrompt::DeepseekChat2Prompt(DeepseekChat2Prompt)
            }
            PromptTemplateType::DeepseekChat25 => {
                ChatPrompt::DeepseekChat25Prompt(DeepseekChat25Prompt)
            }
            PromptTemplateType::DeepseekChat3 => {
                ChatPrompt::DeepseekChat25Prompt(DeepseekChat25Prompt)
            }
            PromptTemplateType::SolarInstruct => {
                ChatPrompt::SolarInstructPrompt(SolarInstructPrompt)
            }
            PromptTemplateType::Phi2Chat => ChatPrompt::Phi2ChatPrompt(Phi2ChatPrompt),
            PromptTemplateType::Phi2Instruct => ChatPrompt::Phi2InstructPrompt(Phi2InstructPrompt),
            PromptTemplateType::Phi3Chat => ChatPrompt::Phi3ChatPrompt(Phi3ChatPrompt),
            PromptTemplateType::Phi3Instruct => ChatPrompt::Phi3InstructPrompt(Phi3InstructPrompt),
            PromptTemplateType::Phi4Chat => ChatPrompt::Phi4ChatPrompt(Phi4ChatPrompt),
            PromptTemplateType::GemmaInstruct => {
                ChatPrompt::GemmaInstructPrompt(GemmaInstructPrompt)
            }
            PromptTemplateType::Gemma3 => ChatPrompt::Gemma3Prompt(Gemma3Prompt),
            PromptTemplateType::Octopus => ChatPrompt::OctopusPrompt(OctopusPrompt),
            PromptTemplateType::Glm4Chat => ChatPrompt::Glm4ChatPrompt(Glm4ChatPrompt),
            PromptTemplateType::GroqLlama3Tool => {
                ChatPrompt::GroqLlama3ToolPrompt(GroqLlama3ToolPrompt)
            }
            PromptTemplateType::BreezeInstruct => {
                ChatPrompt::BreezeInstructPrompt(BreezeInstructPrompt)
            }
            PromptTemplateType::NemotronChat => ChatPrompt::NemotronChatPrompt(NemotronChatPrompt),
            PromptTemplateType::NemotronTool => ChatPrompt::NemotronToolPrompt(NemotronToolPrompt),
            PromptTemplateType::FunctionaryV32 => {
                ChatPrompt::FunctionaryV32ToolPrompt(FunctionaryV32ToolPrompt)
            }
            PromptTemplateType::FunctionaryV31 => {
                ChatPrompt::FunctionaryV31ToolPrompt(FunctionaryV31ToolPrompt)
            }
            PromptTemplateType::MiniCPMV => ChatPrompt::MiniCPMVPrompt(MiniCPMVPrompt),
            PromptTemplateType::MoxinChat => ChatPrompt::MoxinChatPrompt(MoxinChatPrompt),
            PromptTemplateType::MoxinInstruct => {
                ChatPrompt::MoxinInstructPrompt(MoxinInstructPrompt)
            }
            PromptTemplateType::Falcon3 => ChatPrompt::FalconChatPrompt(FalconChatPrompt),
            PromptTemplateType::Megrez => ChatPrompt::MegrezPrompt(MegrezPrompt),
            PromptTemplateType::Qwen2vl => ChatPrompt::Qwen2vlPrompt(Qwen2vlPrompt),
            PromptTemplateType::Qwen3NoThink => ChatPrompt::Qwen3NoThinkPrompt(Qwen3NoThinkPrompt),
            PromptTemplateType::Qwen3Agent => ChatPrompt::Qwen3AgentPrompt(Qwen3AgentPrompt),
            PromptTemplateType::ExaoneDeepChat => {
                ChatPrompt::ExaoneDeepChatPrompt(ExaoneDeepChatPrompt)
            }
            PromptTemplateType::ExaoneChat => ChatPrompt::ExaoneChatPrompt(ExaoneChatPrompt),
            PromptTemplateType::SeedInstruct => ChatPrompt::SeedInstructPrompt(SeedInstructPrompt),
            PromptTemplateType::SeedReasoning => {
                ChatPrompt::SeedReasoningPrompt(SeedReasoningPrompt)
            }
            PromptTemplateType::Smolvl => ChatPrompt::SmolvlPrompt(SmolvlPrompt),
            PromptTemplateType::Smol3NoThink => ChatPrompt::Smol3NoThinkPrompt(Smol3NoThinkPrompt),
            PromptTemplateType::GptOss => ChatPrompt::GptOssPrompt(GptOssPrompt),
            PromptTemplateType::Embedding => {
                panic!("Embedding prompt template is not used for building chat prompts")
            }
            PromptTemplateType::Tts => {
                panic!("Tts prompt template is not used for building chat prompts")
            }
            PromptTemplateType::Null => {
                panic!("Null prompt template is not used for building chat prompts")
            }
        }
    }
}

/// Trait for merging RAG context into chat messages
pub trait MergeRagContext: Send {
    /// Merge RAG context into chat messages.
    ///
    /// Note that the default implementation simply merges the RAG context into the system message. That is, to use the default implementation, `has_system_prompt` should be set to `true` and `policy` set to `MergeRagContextPolicy::SystemMessage`.
    ///
    /// # Arguments
    ///
    /// * `messages` - The chat messages to merge the context into.
    ///
    /// * `context` - The RAG context to merge into the chat messages.
    ///
    /// * `has_system_prompt` - Whether the chat template has a system prompt.
    ///
    /// * `policy` - The policy for merging RAG context into chat messages.
    fn build(
        messages: &mut Vec<endpoints::chat::ChatCompletionRequestMessage>,
        context: &[String],
        has_system_prompt: bool,
        policy: MergeRagContextPolicy,
        rag_prompt: Option<String>,
    ) -> error::Result<()> {
        if (policy == MergeRagContextPolicy::SystemMessage) && has_system_prompt {
            if messages.is_empty() {
                return Err(error::PromptError::NoMessages);
            }

            if context.is_empty() {
                return Err(error::PromptError::Operation(
                    "No context provided.".to_string(),
                ));
            }

            let context = context[0].trim_end();

            // update or insert system message
            match messages[0] {
                ChatCompletionRequestMessage::System(ref message) => {
                    // compose new system message content
                    let content = match rag_prompt {
                        Some(rag_prompt) if !rag_prompt.is_empty() => {
                            format!(
                                "{original_system_message}\n{rag_prompt}\n{context}",
                                original_system_message = message.content().trim(),
                                rag_prompt = rag_prompt.trim(),
                                context = context.trim_end()
                            )
                        }
                        _ => {
                            format!("{original_system_message}\nUse the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}", original_system_message=message.content().trim(), context=context.trim_end())
                        }
                    };

                    // create system message
                    let system_message = ChatCompletionRequestMessage::new_system_message(
                        content,
                        messages[0].name().cloned(),
                    );

                    // replace the original system message
                    messages[0] = system_message;
                }
                _ => {
                    // compose new system message content
                    let content = match rag_prompt {
                        Some(rag_prompt) if !rag_prompt.is_empty() => {
                            format!(
                                "{rag_prompt}\n{context}",
                                rag_prompt = rag_prompt.trim(),
                                context = context.trim_end()
                            )
                        }
                        _ => {
                            format!("Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{}", context.trim_end())
                        }
                    };

                    // create system message
                    let system_message = ChatCompletionRequestMessage::new_system_message(
                        content,
                        messages[0].name().cloned(),
                    );
                    // insert system message
                    messages.insert(0, system_message);
                }
            };
        }

        Ok(())
    }
}

/// Define the strategy for merging RAG context into chat messages.
#[derive(Clone, Debug, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum MergeRagContextPolicy {
    /// Merge RAG context into the system message.
    ///
    /// Note that this policy is only applicable when the chat template has a system message.
    #[default]
    #[serde(rename = "system-message")]
    SystemMessage,
    /// Merge RAG context into the last user message.
    #[serde(rename = "last-user-message")]
    LastUserMessage,
}
impl std::fmt::Display for MergeRagContextPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeRagContextPolicy::SystemMessage => write!(f, "system-message"),
            MergeRagContextPolicy::LastUserMessage => write!(f, "last-user-message"),
        }
    }
}
impl FromStr for MergeRagContextPolicy {
    type Err = error::PromptError;

    fn from_str(policy: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match policy {
            "system-message" => MergeRagContextPolicy::SystemMessage,
            "last-user-message" => MergeRagContextPolicy::LastUserMessage,
            _ => {
                return Err(error::PromptError::UnknownMergeRagContextPolicy(
                    policy.to_string(),
                ))
            }
        })
    }
}

/// Trait for building prompts for chat completions.
#[enum_dispatch::enum_dispatch]
pub trait BuildChatPrompt: Send {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String>;

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        _tools: Option<&[Tool]>,
    ) -> Result<String> {
        self.build(messages)
    }
}
