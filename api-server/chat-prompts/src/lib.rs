pub mod chat;
pub mod error;

use clap::ValueEnum;
use endpoints::chat::ChatCompletionRequestMessage;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum PromptTemplateType {
    #[value(name = "llama-2-chat")]
    Llama2Chat,
    #[value(name = "llama-3-chat")]
    Llama3Chat,
    #[value(name = "mistral-instruct")]
    MistralInstruct,
    #[value(name = "mistrallite")]
    MistralLite,
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
    #[value(name = "solar-instruct")]
    SolarInstruct,
    #[value(name = "phi-2-chat")]
    Phi2Chat,
    #[value(name = "phi-2-instruct")]
    Phi2Instruct,
    #[value(name = "gemma-instruct")]
    GemmaInstruct,
    #[value(name = "octopus")]
    Octopus,
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "llama-3-chat" => Ok(PromptTemplateType::Llama3Chat),
            "mistral-instruct" => Ok(PromptTemplateType::MistralInstruct),
            "mistrallite" => Ok(PromptTemplateType::MistralLite),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "codellama-super-instruct" => Ok(PromptTemplateType::CodeLlamaSuper),
            "belle-llama-2-chat" => Ok(PromptTemplateType::HumanAssistant),
            "human-assistant" => Ok(PromptTemplateType::HumanAssistant),
            "vicuna-1.0-chat" => Ok(PromptTemplateType::VicunaChat),
            "vicuna-1.1-chat" => Ok(PromptTemplateType::Vicuna11Chat),
            "vicuna-llava" => Ok(PromptTemplateType::VicunaLlava),
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
            "gemma-instruct" => Ok(PromptTemplateType::GemmaInstruct),
            "octopus" => Ok(PromptTemplateType::Octopus),
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
            PromptTemplateType::MistralInstruct => write!(f, "mistral-instruct"),
            PromptTemplateType::MistralLite => write!(f, "mistrallite"),
            PromptTemplateType::OpenChat => write!(f, "openchat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::HumanAssistant => write!(f, "human-asistant"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-1.0-chat"),
            PromptTemplateType::Vicuna11Chat => write!(f, "vicuna-1.1-chat"),
            PromptTemplateType::VicunaLlava => write!(f, "vicuna-llava"),
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
            PromptTemplateType::GemmaInstruct => write!(f, "gemma-instruct"),
            PromptTemplateType::Octopus => write!(f, "octopus"),
        }
    }
}

/// Trait for inserting RAG context into chat messages
pub trait MergeRagContext: Send {
    fn build(
        messages: &mut Vec<endpoints::chat::ChatCompletionRequestMessage>,
        context: &[String],
    ) -> error::Result<()> {
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
                let content = format!("{original_system_message}\nUse the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}", original_system_message=message.content().trim(), context=context.trim_end());
                // create system message
                let system_message = ChatCompletionRequestMessage::new_system_message(
                    content,
                    messages[0].name().cloned(),
                );
                // replace the original system message
                messages[0] = system_message;
            }
            _ => {
                // prepare system message
                let content = format!("Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{}", context.trim_end());

                // create system message
                let system_message = ChatCompletionRequestMessage::new_system_message(
                    content,
                    messages[0].name().cloned(),
                );
                // insert system message
                messages.insert(0, system_message);
            }
        };

        Ok(())
    }
}
