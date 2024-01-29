pub mod baichuan;
pub mod belle;
pub mod chatml;
pub mod deepseek;
pub mod intel;
pub mod llama;
pub mod mistral;
pub mod openchat;
pub mod phi;
pub mod solar;
pub mod vicuna;
pub mod wizard;
pub mod zephyr;

use crate::error::Result;
use baichuan::*;
use belle::*;
use chatml::*;
use deepseek::*;
use endpoints::chat::ChatCompletionRequestMessage;
use intel::*;
use llama::*;
use mistral::*;
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
    HumanAssistantChatPrompt,
    /// Vicuna 1.0
    VicunaChatPrompt,
    /// Vicuna 1.1
    Vicuna11ChatPrompt,
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
}
