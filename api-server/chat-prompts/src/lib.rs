pub mod chat;
pub mod error;

use std::str::FromStr;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum PromptTemplateType {
    Llama2Chat,
    MistralInstructV01,
    MistralLite,
    OpenChat,
    CodeLlama,
    BelleLlama2Chat,
    VicunaChat,
    ChatML,
    Baichuan2,
    WizardCoder,
    Zephyr,
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "mistral-instruct-v0.1" => Ok(PromptTemplateType::MistralInstructV01),
            "mistrallite" => Ok(PromptTemplateType::MistralLite),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "belle-llama-2-chat" => Ok(PromptTemplateType::BelleLlama2Chat),
            "vicuna-chat" => Ok(PromptTemplateType::VicunaChat),
            "chatml" => Ok(PromptTemplateType::ChatML),
            "openchat" => Ok(PromptTemplateType::OpenChat),
            "baichuan-2" => Ok(PromptTemplateType::Baichuan2),
            "wizard-coder" => Ok(PromptTemplateType::WizardCoder),
            "zephyr" => Ok(PromptTemplateType::Zephyr),
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
            PromptTemplateType::MistralInstructV01 => write!(f, "mistral-instruct-v0.1"),
            PromptTemplateType::MistralLite => write!(f, "mistrallite"),
            PromptTemplateType::OpenChat => write!(f, "openchat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::BelleLlama2Chat => write!(f, "belle-llama-2-chat"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-chat"),
            PromptTemplateType::ChatML => write!(f, "chatml"),
            PromptTemplateType::Baichuan2 => write!(f, "baichuan-2"),
            PromptTemplateType::WizardCoder => write!(f, "wizard-coder"),
            PromptTemplateType::Zephyr => write!(f, "zephyr"),
        }
    }
}
