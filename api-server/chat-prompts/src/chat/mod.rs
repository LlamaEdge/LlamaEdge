pub mod baichuan;
pub mod belle;
pub mod chatml;
pub mod deepseek;
pub mod gemma;
pub mod intel;
pub mod llama;
pub mod mistral;
pub mod octopus;
pub mod openchat;
pub mod phi;
pub mod solar;
pub mod vicuna;
pub mod wizard;
pub mod zephyr;

use crate::{error::Result, PromptTemplateType};
use baichuan::*;
use belle::*;
use chatml::*;
use deepseek::*;
use endpoints::chat::ChatCompletionRequestMessage;
use gemma::*;
use intel::*;
use llama::*;
use mistral::*;
use octopus::*;
use openchat::*;
use phi::*;
use solar::*;
use vicuna::*;
use wizard::*;
use zephyr::*;

#[enum_dispatch::enum_dispatch]
pub trait BuildChatPrompt: Send {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String>;
}

#[enum_dispatch::enum_dispatch(BuildChatPrompt)]
pub enum ChatPrompt {
    Llama2ChatPrompt,
    MistralInstructPrompt,
    MistralLitePrompt,
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
    Baichuan2ChatPrompt,
    WizardCoderPrompt,
    ZephyrChatPrompt,
    StableLMZephyrChatPrompt,
    NeuralChatPrompt,
    DeepseekChatPrompt,
    DeepseekCoderPrompt,
    SolarInstructPrompt,
    Phi2ChatPrompt,
    Phi2InstructPrompt,
    GemmaInstructPrompt,
    OctopusPrompt,
}
impl From<PromptTemplateType> for ChatPrompt {
    fn from(ty: PromptTemplateType) -> Self {
        match ty {
            PromptTemplateType::Llama2Chat => ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt),
            PromptTemplateType::MistralInstruct => {
                ChatPrompt::MistralInstructPrompt(MistralInstructPrompt)
            }
            PromptTemplateType::MistralLite => ChatPrompt::MistralLitePrompt(MistralLitePrompt),
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
            PromptTemplateType::SolarInstruct => {
                ChatPrompt::SolarInstructPrompt(SolarInstructPrompt)
            }
            PromptTemplateType::Phi2Chat => ChatPrompt::Phi2ChatPrompt(Phi2ChatPrompt),
            PromptTemplateType::Phi2Instruct => ChatPrompt::Phi2InstructPrompt(Phi2InstructPrompt),
            PromptTemplateType::GemmaInstruct => {
                ChatPrompt::GemmaInstructPrompt(GemmaInstructPrompt)
            }
            PromptTemplateType::Octopus => ChatPrompt::OctopusPrompt(OctopusPrompt),
        }
    }
}
