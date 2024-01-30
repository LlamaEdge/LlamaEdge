pub mod chat;
pub mod error;

use std::str::FromStr;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum PromptTemplateType {
    Llama2Chat,
    MistralInstruct,
    MistralLite,
    OpenChat,
    CodeLlama,
    CodeLlamaSuper,
    HumanAssistant,
    VicunaChat,
    Vicuna11Chat,
    ChatML,
    Baichuan2,
    WizardCoder,
    Zephyr,
    StableLMZephyr,
    IntelNeural,
    DeepseekChat,
    DeepseekCoder,
    SolarInstruct,
    Phi2Chat,
    Phi2Instruct,
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "mistral-instruct-v0.1" => Ok(PromptTemplateType::MistralInstruct),
            "mistral-instruct" => Ok(PromptTemplateType::MistralInstruct),
            "mistrallite" => Ok(PromptTemplateType::MistralLite),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "codellama-super-instruct" => Ok(PromptTemplateType::CodeLlamaSuper),
            "belle-llama-2-chat" => Ok(PromptTemplateType::HumanAssistant),
            "human-assistant" => Ok(PromptTemplateType::HumanAssistant),
            "vicuna-chat" => Ok(PromptTemplateType::VicunaChat),
            "vicuna-1.0-chat" => Ok(PromptTemplateType::VicunaChat),
            "vicuna-1.1-chat" => Ok(PromptTemplateType::Vicuna11Chat),
            "chatml" => Ok(PromptTemplateType::ChatML),
            "openchat" => Ok(PromptTemplateType::OpenChat),
            "baichuan-2" => Ok(PromptTemplateType::Baichuan2),
            "wizard-coder" => Ok(PromptTemplateType::WizardCoder),
            "zephyr" => Ok(PromptTemplateType::Zephyr),
            "stablelm-zephyr" => Ok(PromptTemplateType::StableLMZephyr),
            "intel-neural" => Ok(PromptTemplateType::IntelNeural),
            "deepseek-chat" => Ok(PromptTemplateType::DeepseekChat),
            "deepseek-coder" => Ok(PromptTemplateType::DeepseekCoder),
            "solar-instruct" => Ok(PromptTemplateType::SolarInstruct),
            "phi-2-chat" => Ok(PromptTemplateType::Phi2Chat),
            "phi-2-instruct" => Ok(PromptTemplateType::Phi2Instruct),
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
            PromptTemplateType::MistralInstruct => write!(f, "mistral-instruct"),
            PromptTemplateType::MistralLite => write!(f, "mistrallite"),
            PromptTemplateType::OpenChat => write!(f, "openchat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::HumanAssistant => write!(f, "human-asistant"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-1.0-chat"),
            PromptTemplateType::Vicuna11Chat => write!(f, "vicuna-1.1-chat"),
            PromptTemplateType::ChatML => write!(f, "chatml"),
            PromptTemplateType::Baichuan2 => write!(f, "baichuan-2"),
            PromptTemplateType::WizardCoder => write!(f, "wizard-coder"),
            PromptTemplateType::Zephyr => write!(f, "zephyr"),
            PromptTemplateType::StableLMZephyr => write!(f, "stablelm-zephyr"),
            PromptTemplateType::IntelNeural => write!(f, "intel-neural"),
            PromptTemplateType::DeepseekChat => write!(f, "deepseek-chat"),
            PromptTemplateType::DeepseekCoder => write!(f, "deepseek-coder"),
            PromptTemplateType::SolarInstruct => write!(f, "solar-instruct"),
            PromptTemplateType::Phi2Chat => write!(f, "phi-2-chat"),
            PromptTemplateType::Phi2Instruct => write!(f, "phi-2-instruct"),
            PromptTemplateType::CodeLlamaSuper => write!(f, "codellama-super-instruct"),
        }
    }
}
