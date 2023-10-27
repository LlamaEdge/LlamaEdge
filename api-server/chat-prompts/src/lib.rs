pub mod chat;
pub mod error;

use std::str::FromStr;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum PromptTemplateType {
    Llama2Chat,
    MistralChat,
    CodeLlama,
    BelleLlama2Chat,
    VicunaChat,
    ChatML,
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "mistral-chat" => Ok(PromptTemplateType::MistralChat),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "belle-llama-2-chat" => Ok(PromptTemplateType::BelleLlama2Chat),
            "vicuna-chat" => Ok(PromptTemplateType::VicunaChat),
            "chatml" => Ok(PromptTemplateType::ChatML),
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
            PromptTemplateType::MistralChat => write!(f, "mistral-chat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::BelleLlama2Chat => write!(f, "belle-llama-2-chat"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-chat"),
            PromptTemplateType::ChatML => write!(f, "chatml"),
        }
    }
}
