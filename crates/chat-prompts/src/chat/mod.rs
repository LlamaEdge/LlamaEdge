pub mod baichuan;
pub mod belle;
pub mod chatml;
pub mod deepseek;
pub mod exaone;
pub mod falcon;
pub mod functionary;
pub mod gemma;
pub mod glm;
pub mod groq;
pub mod intel;
pub mod llama;
pub mod mediatek;
pub mod megrez;
pub mod minicpm;
pub mod mistral;
pub mod moxin;
pub mod nvidia;
pub mod octopus;
pub mod openchat;
pub mod phi;
pub mod qwen;
pub mod solar;
pub mod vicuna;
pub mod wizard;
pub mod zephyr;

use crate::{error::Result, PromptTemplateType};
use baichuan::*;
use belle::*;
use chatml::*;
use deepseek::*;
use endpoints::chat::{ChatCompletionRequestMessage, Tool};
use exaone::*;
use falcon::*;
use functionary::{FunctionaryV31ToolPrompt, FunctionaryV32ToolPrompt};
use gemma::*;
use glm::*;
use groq::*;
use intel::*;
use llama::*;
use mediatek::BreezeInstructPrompt;
use megrez::*;
use minicpm::*;
use mistral::*;
use moxin::*;
use nvidia::{NemotronChatPrompt, NemotronToolPrompt};
use octopus::*;
use openchat::*;
use phi::*;
use qwen::Qwen2vlPrompt;
use solar::*;
use vicuna::*;
use wizard::*;
use zephyr::*;

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

#[enum_dispatch::enum_dispatch(BuildChatPrompt)]
pub enum ChatPrompt {
    Llama2ChatPrompt,
    Llama3ChatPrompt,
    Llama3ToolPrompt,
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
    FalconChatPrompt,
    MegrezPrompt,
    Qwen2vlPrompt,
    ExaoneDeepChatPrompt,
}
impl From<PromptTemplateType> for ChatPrompt {
    fn from(ty: PromptTemplateType) -> Self {
        match ty {
            PromptTemplateType::Llama2Chat => ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt),
            PromptTemplateType::Llama3Chat => ChatPrompt::Llama3ChatPrompt(Llama3ChatPrompt),
            PromptTemplateType::Llama3Tool => ChatPrompt::Llama3ToolPrompt(Llama3ToolPrompt),
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
            PromptTemplateType::Falcon3 => ChatPrompt::FalconChatPrompt(FalconChatPrompt),
            PromptTemplateType::Megrez => ChatPrompt::MegrezPrompt(MegrezPrompt),
            PromptTemplateType::Qwen2vl => ChatPrompt::Qwen2vlPrompt(Qwen2vlPrompt),
            PromptTemplateType::ExaoneDeepChat => {
                ChatPrompt::ExaoneDeepChatPrompt(ExaoneDeepChatPrompt)
            }
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
